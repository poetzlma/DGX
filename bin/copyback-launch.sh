#!/bin/bash
# copyback-launch.sh — on-demand weight fetch wrapper for dormant llama-swap slots.
#
# Weights for dormant (eval/rollback) models live on codeserver (COPYBACK_REMOTE,
# default 192.168.1.16), NOT on the Spark's NVMe. This wrapper is placed in front
# of a model's real launch script in llama-swap. When llama-swap starts the model it:
#   1. evicts every OTHER copy-back-managed model from local disk
#      (enforces "at most one dormant model resident" + self-heals after SIGKILL)
#   2. rsyncs THIS model's weights from codeserver unless a completion sentinel
#      proves a prior pull finished (guards against poisoned partial copies)
#   3. runs the real launch script, forwarding termination signals, and waits for
#      the child to FULLY exit (so the container frees the GPU) before evicting
#   4. on stop, evicts this model's local weights (evict-immediately-after-use)
#
# Usage in llama-swap cmd:
#   /home/max/llm-stack/bin/copyback-launch.sh <local_path> <remote_relpath> <real_launch_script>
#     local_path      absolute path that must exist before launch
#                     (e.g. ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B-FP8)
#     remote_relpath  path under codeserver:~/$COPYBACK_ARCHIVE_ROOT/
#                     (e.g. hub/models--Qwen--Qwen3.6-27B-FP8)
#     real_launch_script  the existing launch-vllm-*.sh / launch-*.sh.
#                     NOTE: it MUST `exec` its engine (docker run / binary) so the
#                     forwarded SIGTERM reaches the container, not an orphaned shell.
#
# Env overrides: COPYBACK_REMOTE, COPYBACK_ARCHIVE_ROOT, COPYBACK_MANIFEST.
# healthCheckTimeout in llama-swap.yaml MUST exceed the pull time
# (~1-3 min for 20-30G models, ~13 min for ds4's 85G GGUF). Currently 1200s.
set -uo pipefail

REMOTE="${COPYBACK_REMOTE:-192.168.1.16}"   # codeserver host (override via COPYBACK_REMOTE)
ARCHIVE_ROOT="${COPYBACK_ARCHIVE_ROOT:-llm-weights-archive}"   # codeserver:~/llm-weights-archive/
MANIFEST="${COPYBACK_MANIFEST:-$HOME/llm-stack/etc/copyback-models.txt}"   # one local_path per line
RSYNC_OPTS=(-a --partial --info=progress2,stats2 --timeout=120)

LOCAL_PATH="${1:?need local_path}"
REMOTE_REL="${2:?need remote_relpath}"
REAL="${3:?need real launch script}"

SENTINEL="$LOCAL_PATH/.copyback-complete"   # written only after a fully-successful pull

log() { echo "[copyback $(basename "$LOCAL_PATH")] $*" >&2; }

# --- safety: only ever rm -rf paths under a known weights root ----------------
: "${HOME:?HOME must be set}"
_evict_ok() {   # 0 = safe to rm, 1 = refuse
  local p="$1"
  case "$p" in
    ""|/|"$HOME"|"$HOME"/) return 1 ;;   # never the root or home itself
    *..*) return 1 ;;                    # no path traversal
  esac
  case "$p" in
    "$HOME"/.cache/huggingface/*|"$HOME"/models/*|"$HOME"/llm-stack/models/*|"$HOME"/ds4/*|"$HOME"/cosmos3/*) return 0 ;;
    *) return 1 ;;
  esac
}
safe_rm() {
  local p="$1"
  if _evict_ok "$p"; then
    rm -rf "$p" || log "WARN: rm -rf failed for $p"
  else
    log "REFUSING to evict unsafe path: '$p' (not under a known weights root)"
  fi
}

# 1. Evict every OTHER managed model (keep disk to one dormant model at a time).
if [[ -f "$MANIFEST" ]]; then
  while IFS= read -r m; do
    m="${m%$'\r'}"                    # strip trailing CR (CRLF-edited manifest)
    m="${m#"${m%%[![:space:]]*}"}"    # ltrim whitespace
    m="${m%"${m##*[![:space:]]}"}"    # rtrim whitespace
    [[ -z "$m" || "$m" == \#* ]] && continue
    m="${m/#\~/$HOME}"
    [[ "$m" == "$LOCAL_PATH" ]] && continue
    if [[ -e "$m" ]]; then
      log "evicting stale managed model: $m"
      safe_rm "$m"
    fi
  done < "$MANIFEST"
fi

# 2. Pull this model unless a prior pull COMPLETED (sentinel). A bare directory
#    without the sentinel is a poisoned partial (SIGKILL'd mid-rsync) → re-pull.
if [[ ! -e "$SENTINEL" ]]; then
  [[ -e "$LOCAL_PATH" ]] && { log "incomplete copy present — removing before re-pull"; safe_rm "$LOCAL_PATH"; }
  log "weights absent — pulling from $REMOTE:$ARCHIVE_ROOT/$REMOTE_REL"
  if ! mkdir -p "$LOCAL_PATH"; then log "ERROR: mkdir failed"; exit 1; fi
  if ! rsync "${RSYNC_OPTS[@]}" \
        "$REMOTE:$ARCHIVE_ROOT/$REMOTE_REL/" "$LOCAL_PATH/"; then
    log "ERROR: pull failed — removing partial copy"
    safe_rm "$LOCAL_PATH"
    exit 1
  fi
  : > "$SENTINEL" || log "WARN: could not write completion sentinel"
  log "pull complete"
else
  log "weights already local — skipping pull"
fi

# 3. Run the real launch script, forwarding signals; evict AFTER the child fully exits.
evict() {
  log "stopping — evicting local weights"
  safe_rm "$LOCAL_PATH"
}
"$REAL" &
CHILD=$!
forward() { kill -TERM "$CHILD" 2>/dev/null; }
trap forward TERM INT
wait "$CHILD"; RC=$?
# `wait` returns early (128+signum) when the trap fires; keep waiting until the
# child is really gone so the container tears down and frees the GPU BEFORE evict.
while kill -0 "$CHILD" 2>/dev/null; do wait "$CHILD"; RC=$?; done
trap - TERM INT
evict
exit "$RC"
