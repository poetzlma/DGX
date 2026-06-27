"""AutoRound INT4 quant of Qwopus, with the shard-writer meta-offload disabled.

auto_round 0.12.3's ShardWriter._offload_to_meta calls module.to("meta") on
modules whose state_dict contains non-Parameter tensors (the Qwen3.5 linear-
attention / fp16-kept in_proj modules), which trips
  AssertionError: param must be a Parameter
during save. The offload is only a RAM optimization; on a 119 GB box we don't
need it. No-op'ing it also makes finalize() see real tensors instead of meta.
"""
import sys
import auto_round.compressors.shard_writer as sw

sw.ShardWriter._offload_to_meta = lambda self, saved_params: None

from auto_round.__main__ import run

sys.argv = [
    "auto-round",
    "--model_name", "/root/.cache/huggingface/hub/models--Jackrong--Qwopus3.6-27B-v2/snapshots/66a4ee9d49dbcef3f83528a400ef6bec93684d6b",
    "--bits", "4",
    "--group_size", "128",
    "--data_type", "int",
    "--iters", "0",
    "--fp_layers", "in_proj_a,in_proj_b,mtp.fc",
    "--to_quant_block_names", "model.language_model.layers",
    "--format", "auto_round:auto_gptq",
    "--output_dir", "/models/Qwopus3.6-27B-v2-int4-AutoRound",
    "--device", "0",
]
sys.exit(run())
