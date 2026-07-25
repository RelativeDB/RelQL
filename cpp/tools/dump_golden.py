"""Dump golden batches + reference activations for the C++ RT implementation.

Reuses the synthetic-churn batch builder from rt/demo/run_rt.py, runs the real
RT-J classification checkpoint in fp32, and saves:
  - every input tensor of the batch (raw, PRE-sort order)
  - x_embed  : block-0 input  (post-sort order)  [B,S,512]
  - x_block0 : block-0 output (post-sort order)  [B,S,512]
  - yhat_number : final number-head output (post-sort order) [B,S]
  - sorted_is_targets [B,S] and sort_idxs [B,S]
as little-endian .bin files + manifest.json with shapes/dtypes.

TWO goldens are written:

  cpp/testdata/         B=5  S=16  -- the small, fast gate (one context per row)
  cpp/testdata/large/   B=2  S>=64 -- the long-context gate

The large one exists because the small one is structurally blind to a whole
class of defect. At S=16 every attention group is tiny and the node ids are
0..N, so a bug that only bites when (a) sequences are long, (b) column groups
AND same-node (feat) groups both hold several tokens, or (c) node ids exceed
2**24 -- the point where a float32 round-trip starts merging distinct ids --
passes the small gate green. The large batch therefore concatenates every
target customer's context into each row (long sequences, column groups with
one token per customer, feat groups with several tokens per node) and remaps
node ids into a realistic global-id range around 4.49e7, where consecutive
integers are NOT representable in float32 and collapse in groups of four.

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

from rt.checkpoints import load_rt_model
model, config = load_rt_model(run_rt.CKPT, device="cpu")
model = model.float().eval()

captured = {}
model.blocks[0].register_forward_pre_hook(
    lambda mod, args: captured.__setitem__("x_embed", args[0].detach().clone()))
model.blocks[0].register_forward_hook(
    lambda mod, args, out: captured.__setitem__("x_block0", out.detach().clone()))

from huggingface_hub import hf_hub_download
st_path = hf_hub_download("stanford-star/rt-j", "classification/model.safetensors")


def dump(batch, out_dir, note):
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        _, yhat, _, sorted_is_targets = model(batch, return_embeddings=False)

    # Recompute the model's internal sort to save sort_idxs (mirrors forward()).
    col = batch["col_name_idxs"]
    sort_keys = col.masked_fill(batch["is_padding"], torch.iinfo(col.dtype).max)
    sort_idxs = sort_keys.argsort(dim=-1, stable=True)

    manifest = {}

    def save(name, tensor, dtype):
        arr = np.ascontiguousarray(tensor.detach().to(torch.float32).numpy()
                                   if tensor.is_floating_point()
                                   else tensor.detach().numpy())
        arr = arr.astype(dtype)
        arr.tofile(out_dir / f"{name}.bin")
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

    manifest["_checkpoint"] = st_path
    manifest["_note"] = ("inputs are PRE-sort; x_embed/x_block0/yhat are "
                         "POST-sort. " + note)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    scores = (yhat["number"].squeeze(-1) * sorted_is_targets.float()).sum(1)
    B, S = batch["node_idxs"].shape
    print(f"\n{out_dir}: B={B} S={S}, {len(manifest) - 2} tensors")
    print("  target scores:", [round(float(v), 5) for v in scores])
    return [round(float(v), 5) for v in scores]


# --------------------------------------------------------------------------- #
# small golden: one context per row (B=5, S=16)
# --------------------------------------------------------------------------- #
contexts = [run_rt.build_context(c) for c in run_rt.TARGET_CUSTOMERS]
dump(run_rt.collate(contexts), OUT, "one context per row")


# --------------------------------------------------------------------------- #
# large golden: every context concatenated into each row (B=2, S>=64)
# --------------------------------------------------------------------------- #
# Node ids are remapped to NODE_ID_BASE + slot*SLOT_STRIDE + local. The base sits
# above 2**24 where float32 spacing is 4, so the consecutive local ids inside one
# context are distinct int64s that a float32 round-trip would merge -- exactly
# the hazard the small golden cannot express.
NODE_ID_BASE = 44_903_100
SLOT_STRIDE = 4096
assert np.float32(NODE_ID_BASE) == np.float32(NODE_ID_BASE + 1), (
    "base must sit where consecutive ints collide in float32")


def concat_contexts(order):
    """One Ctx holding every listed context back to back, node ids remapped."""
    merged = run_rt.Ctx()
    for slot, name in enumerate(order):
        c = run_rt.build_context(name)
        off = NODE_ID_BASE + slot * SLOT_STRIDE
        merged.node += [n + off for n in c.node]
        merged.f2p += [[(p + off if p >= 0 else -1) for p in row] for row in c.f2p]
        for attr in ("col", "tab", "sem", "is_tgt", "num", "dt", "txt",
                     "boolv", "colvec"):
            getattr(merged, attr).extend(getattr(c, attr))
    return merged


targets = list(run_rt.TARGET_CUSTOMERS)
# Six contexts per row (the five targets plus one repeat in a fresh id slot) so
# the sequence clears the S=36 onset comfortably.
rows = [concat_contexts(targets + targets[:1]),
        concat_contexts(targets[::-1] + targets[-1:])]
large = run_rt.collate(rows)
S = large["node_idxs"].shape[1]
assert S >= 64, f"large golden must clear the S=36 onset, got S={S}"

# Assert the properties that make this gate non-blind, so a future regeneration
# cannot quietly produce another all-singletons batch.
node = large["node_idxs"][0][~large["is_padding"][0]].numpy()
colt = [(int(c) << 32) ^ int(t) for c, t in
        zip(large["col_name_idxs"][0][~large["is_padding"][0]],
            large["table_name_idxs"][0][~large["is_padding"][0]])]
_, node_counts = np.unique(node, return_counts=True)
_, col_counts = np.unique(colt, return_counts=True)
assert (node_counts > 1).sum() >= 5, "need multi-token same-node (feat) groups"
assert (col_counts > 1).sum() >= 5, "need multi-token column groups"
assert node.max() > 2 ** 24, "node ids must exceed the float32 integer limit"
assert len(np.unique(node.astype(np.float32))) < len(np.unique(node)), (
    "node ids must include a pair that a float32 round-trip would merge")
print(f"\nlarge batch: S={S}, "
      f"{int((node_counts > 1).sum())} multi-token feat groups, "
      f"{int((col_counts > 1).sum())} multi-token col groups, "
      f"node id max={node.max()}, "
      f"{len(np.unique(node)) - len(np.unique(node.astype(np.float32)))} ids "
      f"lost to a float32 round-trip")

dump(large, OUT / "large",
     "every target context concatenated per row; node ids remapped above 2**24")
print("\ncheckpoint:", st_path)
