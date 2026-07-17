"""Dump a golden batch + reference activations for the C++ RT implementation.

Reuses the synthetic-churn batch builder from rt/demo/run_rt.py, runs the real
RT-J classification checkpoint in fp32, and saves:
  - every input tensor of the batch (raw, PRE-sort order)
  - x_embed  : block-0 input  (post-sort order)  [B,S,512]
  - x_block0 : block-0 output (post-sort order)  [B,S,512]
  - yhat_number : final number-head output (post-sort order) [B,S]
  - sorted_is_targets [B,S] and sort_idxs [B,S]
as little-endian .bin files + manifest.json with shapes/dtypes.

Usage: /Users/henneberger/rt/.venv/bin/python dump_golden.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "testdata"
OUT.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "run_rt", "/Users/henneberger/rt/demo/run_rt.py")
run_rt = importlib.util.module_from_spec(spec)
sys.modules["run_rt"] = run_rt
spec.loader.exec_module(run_rt)          # loads MiniLM + patches flex to eager

contexts = [run_rt.build_context(c) for c in run_rt.TARGET_CUSTOMERS]
batch = run_rt.collate(contexts)

from rt.checkpoints import load_rt_model
model, config = load_rt_model(run_rt.CKPT, device="cpu")
model = model.float().eval()

captured = {}
def pre_hook(mod, args):
    captured["x_embed"] = args[0].detach().clone()
def post_hook(mod, args, out):
    captured["x_block0"] = out.detach().clone()
model.blocks[0].register_forward_pre_hook(pre_hook)
model.blocks[0].register_forward_hook(post_hook)

with torch.no_grad():
    _, yhat, _, sorted_is_targets = model(batch, return_embeddings=False)

# Recompute the model's internal sort to save sort_idxs (mirrors forward()).
col = batch["col_name_idxs"]
pad = batch["is_padding"]
sort_keys = col.masked_fill(pad, torch.iinfo(col.dtype).max)
sort_idxs = sort_keys.argsort(dim=-1, stable=True)

manifest = {}
def save(name, tensor, dtype):
    arr = np.ascontiguousarray(tensor.detach().to(torch.float32).numpy()
                               if tensor.is_floating_point()
                               else tensor.detach().numpy())
    arr = arr.astype(dtype)
    path = OUT / f"{name}.bin"
    arr.tofile(path)
    manifest[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}

# ---- inputs (pre-sort) ----
save("node_idxs", batch["node_idxs"], np.int64)
save("f2p_nbr_idxs", batch["f2p_nbr_idxs"], np.int64)
save("col_name_idxs", batch["col_name_idxs"], np.int64)
save("table_name_idxs", batch["table_name_idxs"], np.int64)
save("is_padding", batch["is_padding"].to(torch.uint8), np.uint8)
save("sem_types", batch["sem_types"], np.int64)
save("is_targets", batch["is_targets"].to(torch.uint8), np.uint8)
save("number_values", batch["number_values"].float(), np.float32)
save("datetime_values", batch["datetime_values"].float(), np.float32)
save("boolean_values", batch["boolean_values"].float(), np.float32)
save("text_values", batch["text_values"].float(), np.float32)
save("col_name_values", batch["col_name_values"].float(), np.float32)
# ---- references (post-sort order) ----
save("sort_idxs", sort_idxs, np.int64)
save("x_embed", captured["x_embed"], np.float32)
save("x_block0", captured["x_block0"], np.float32)
save("yhat_number", yhat["number"].squeeze(-1), np.float32)
save("sorted_is_targets", sorted_is_targets.to(torch.uint8), np.uint8)

import shutil
ckpt = Path(torch.hub.get_dir()).parent  # not used; find real path via HF
from huggingface_hub import hf_hub_download
st_path = hf_hub_download("stanford-star/rt-j", "classification/model.safetensors")
manifest["_checkpoint"] = st_path
manifest["_note"] = "inputs are PRE-sort; x_embed/x_block0/yhat are POST-sort"

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(f"wrote {len(manifest)-2} tensors to {OUT}")
print("checkpoint:", st_path)
tp = (yhat["number"].squeeze(-1) * sorted_is_targets.float()).sum(1)
print("target scores:", [round(float(v), 5) for v in tp])
