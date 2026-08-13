#!/usr/bin/env python3
"""Dump GGUF metadata from the head of a file — works on a PARTIAL download.

The whole KV block lives in the first few MB, so this answers the risky
questions about a new checkpoint (chat template present? tokenizer changed?
architecture string our engine expects?) before an 80 GiB transfer finishes.

Usage: gguf-head.py FILE [--keys substr] [--diff OTHER]
"""
import json, struct, sys

# GGUF value type enum
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
FIXED = {T_U8: ("<B", 1), T_I8: ("<b", 1), T_U16: ("<H", 2), T_I16: ("<h", 2),
         T_U32: ("<I", 4), T_I32: ("<i", 4), T_F32: ("<f", 4), T_BOOL: ("<?", 1),
         T_U64: ("<Q", 8), T_I64: ("<q", 8), T_F64: ("<d", 8)}


class R:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n):
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError("ran past the downloaded region")
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def s(self):
        return self.raw(self.u64()).decode("utf-8", "replace")

    def val(self, t):
        if t in FIXED:
            f, n = FIXED[t]
            return struct.unpack(f, self.raw(n))[0]
        if t == T_STR:
            return self.s()
        if t == T_ARR:
            et = self.u32()
            n = self.u64()
            if et == T_STR:
                # token lists are huge; keep a sample and the count
                out = []
                for i in range(n):
                    v = self.s()
                    if i < 8:
                        out.append(v)
                return {"__array_str__": n, "sample": out}
            if et in FIXED:
                f, sz = FIXED[et]
                buf = self.raw(sz * n)
                if n > 16:
                    return {"__array__": n, "type": et,
                            "sample": list(struct.unpack("<" + f[1] * 16, buf[:sz * 16]))}
                return list(struct.unpack("<" + f[1] * n, buf))
            raise ValueError(f"array of type {et}")
        raise ValueError(f"unknown value type {t}")


def read_meta(path):
    fh = open(path, "rb")
    r = R(fh)
    if r.raw(4) != b"GGUF":
        sys.exit("not a GGUF file")
    ver = r.u32()
    ntensor = r.u64()
    nkv = r.u64()
    kv = {}
    for _ in range(nkv):
        k = r.s()
        t = r.u32()
        kv[k] = r.val(t)
    return {"version": ver, "n_tensors": ntensor, "n_kv": nkv, "kv": kv}


def summarize(m, label):
    kv = m["kv"]
    print(f"### {label}")
    print(f"  gguf v{m['version']} | {m['n_tensors']} tensors | {m['n_kv']} metadata keys")
    for k in ("general.architecture", "general.name", "general.basename",
              "general.quantization_version", "general.file_type", "general.size_label"):
        if k in kv:
            print(f"  {k:38} = {kv[k]}")
    tmpl = kv.get("tokenizer.chat_template")
    if tmpl is None:
        print("  tokenizer.chat_template              = *** ABSENT ***")
    else:
        print(f"  tokenizer.chat_template              = present, {len(tmpl)} chars")
        first = tmpl.strip().splitlines()[0][:100] if tmpl.strip() else ""
        print(f"      first line: {first!r}")
    for k in sorted(kv):
        if k.startswith("tokenizer.") and not k.startswith("tokenizer.ggml.tokens") \
                and not k.startswith("tokenizer.ggml.merges") and k != "tokenizer.chat_template":
            v = kv[k]
            if isinstance(v, dict):
                v = f"<array n={v.get('__array_str__') or v.get('__array__')}>"
            print(f"  {k:38} = {str(v)[:80]}")
    return kv


def main():
    path = sys.argv[1]
    m = read_meta(path)
    kv = summarize(m, path.split("/")[-1])

    if "--keys" in sys.argv:
        sub = sys.argv[sys.argv.index("--keys") + 1]
        print(f"\n--- keys matching {sub!r} ---")
        for k in sorted(kv):
            if sub in k:
                v = kv[k]
                if isinstance(v, dict):
                    v = f"<array n={v.get('__array_str__') or v.get('__array__')} sample={v.get('sample')}>"
                print(f"  {k:44} = {str(v)[:160]}")

    if "--diff" in sys.argv:
        other = sys.argv[sys.argv.index("--diff") + 1]
        m2 = read_meta(other)
        kv2 = m2["kv"]
        print(f"\n{'='*70}\nDIFF vs {other.split('/')[-1]}\n{'='*70}")
        print(f"  tensors: {m['n_tensors']} vs {m2['n_tensors']}"
              + ("  <-- DIFFERENT" if m["n_tensors"] != m2["n_tensors"] else "  (same)"))
        allk = sorted(set(kv) | set(kv2))
        for k in allk:
            a, b = kv.get(k, "<absent>"), kv2.get(k, "<absent>")
            if isinstance(a, dict):
                a = f"<array n={a.get('__array_str__') or a.get('__array__')}>"
            if isinstance(b, dict):
                b = f"<array n={b.get('__array_str__') or b.get('__array__')}>"
            if a != b:
                sa, sb = str(a), str(b)
                if len(sa) > 70 or len(sb) > 70:
                    print(f"  {k}:\n      A({len(sa)} ch) {sa[:70]}...\n      B({len(sb)} ch) {sb[:70]}...")
                else:
                    print(f"  {k}: {sa}  ->  {sb}")


if __name__ == "__main__":
    main()
