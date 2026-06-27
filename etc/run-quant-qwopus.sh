#!/bin/bash
# AutoRound INT4 quantization of Qwopus3.6-27B-v2, replicating Intel's recipe:
#   bits=4, group_size=128, sym, data_type=int, auto_round:auto_gptq packing,
#   linear_attn.in_proj_a/b + mtp.fc kept in original precision (fp16),
#   only model.language_model.layers quantized (vision tower untouched).
#
# iters=0 => RTN (round-to-nearest, no calibration). Calibration quality
# affects ACCURACY, not inference speed / kernel path, and speed is the
# deliverable for this run. A calibrated pass (iters>=200, ~hours) can follow
# if quality numbers are wanted.
#
# Runs in the derived quant image (auto_round 0.12.3 on the aeon torch/tx).
set -e
OUT=/models/Qwopus3.6-27B-v2-int4-AutoRound
docker rm -f qwopus-quant 2>/dev/null || true
docker run --rm --name qwopus-quant \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/max/.cache/huggingface:/root/.cache/huggingface \
  -v /home/max/llm-stack/models:/models \
  -v /home/max/llm-stack/etc:/llm-stack-etc:ro \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  vllm-aeon-autoround:local \
  python3 /llm-stack-etc/quant_driver.py
echo "=== output dir ==="
ls -la /home/max/llm-stack/models/Qwopus3.6-27B-v2-int4-AutoRound/ 2>/dev/null | head
