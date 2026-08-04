"""End-to-end FP16 RT-J inference with Triton-only GPU computation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import triton
from safetensors import safe_open

from .triton_data import build_worklists, load_npz, sort_batch
from .triton_kernels import (
    combine_embeddings_kernel,
    linear,
    linear_gather,
    prepare_kv_kernel,
    relational_attention_kernel,
    rmsnorm,
    scalar_embeddings_kernel,
    swiglu_packed_kernel,
    zero_kernel,
)


D_MODEL = 512
D_TEXT = 384
D_FF = 2048
HEADS = 8
HEAD_DIM = 64
QUERY_TILE = 16
MAX_KEYS = 512


def roc_auc(labels, scores):
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = np.bincount(inverse, weights=ranks)[inverse] / counts[inverse]
    positive = labels.astype(bool)
    npos = int(positive.sum())
    nneg = len(labels) - npos
    return float((ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


class RTJTriton:
    def __init__(self, checkpoint):
        self.weights = {}
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                # As published. The reference trains and serves in bf16 (see
                # rt/embed.py and the checkpoint's own dtype), so narrowing or
                # widening here would be this implementation disagreeing with
                # the one the weights came from. The kernels meet the weights
                # at the dot instead.
                self.weights[name] = handle.get_tensor(name).cuda()
        masks = []
        for semantic in ("number", "text", "datetime", "boolean"):
            masks.append(self.weights[f"mask_embs.{semantic}"])
        self.mask_embeddings = torch.stack(masks).contiguous()
        self.qkvg_weights = {}
        for block in range(12):
            for kind in ("col", "feat", "nbr"):
                prefix = f"blocks.{block}.attns.{kind}"
                # Concatenated in torch, on the device. Going through numpy
                # here is what made a bf16 checkpoint unloadable: numpy has no
                # bfloat16, so the published weights could only be packed by
                # first widening or narrowing them.
                self.qkvg_weights[(block, kind)] = torch.cat([
                    self.weights[f"{prefix}.wq.weight"],
                    self.weights[f"{prefix}.wk.weight"],
                    self.weights[f"{prefix}.wv.weight"],
                    self.weights[f"{prefix}.wg.weight"],
                ]).contiguous()
                for projection in ("wq", "wk", "wv", "wg"):
                    del self.weights[f"{prefix}.{projection}.weight"]
            self.weights[f"blocks.{block}.ffn.w13.weight"] = torch.cat([
                self.weights[f"blocks.{block}.ffn.w1.weight"],
                self.weights[f"blocks.{block}.ffn.w3.weight"],
            ]).contiguous()
            del self.weights[f"blocks.{block}.ffn.w1.weight"]
            del self.weights[f"blocks.{block}.ffn.w3.weight"]
        self.capacity = 0
        self.key_capacity = 0

    def weight(self, name):
        return self.weights[name]

    def reserve(self, rows, keys):
        device = "cuda"
        if rows > self.capacity:
            self.capacity = rows
            self.x = torch.empty((rows, D_MODEL), dtype=torch.float32, device=device)
            self.xn = torch.empty((rows, D_MODEL), dtype=torch.float16, device=device)
            self.tmp = torch.empty((rows, D_MODEL), dtype=torch.float32, device=device)
            self.col = torch.empty((rows, D_MODEL), dtype=torch.float32, device=device)
            self.text = torch.empty((rows, D_MODEL), dtype=torch.float32, device=device)
            self.scalar = torch.empty((rows, D_MODEL), dtype=torch.float32, device=device)
            self.qkvg = torch.empty((rows, 4 * D_MODEL), dtype=torch.float16, device=device)
            self.att = torch.empty((rows, D_MODEL), dtype=torch.float16, device=device)
            self.ff13 = torch.empty((rows, 2 * D_FF), dtype=torch.float16, device=device)
            self.ffa = torch.empty((rows, D_FF), dtype=torch.float16, device=device)
            self.scores = torch.empty((rows, 1), dtype=torch.float32, device=device)
        if keys > self.key_capacity:
            self.key_capacity = keys
            self.packed_k = torch.empty((keys, D_MODEL), dtype=torch.float16, device=device)
            self.packed_v = torch.empty((keys, D_MODEL), dtype=torch.float16, device=device)

    @staticmethod
    def device_array(array, dtype=None):
        t = torch.from_numpy(np.ascontiguousarray(array)).cuda()
        return t if dtype is None else t.to(dtype)

    def prepare(self, raw):
        batch, order = sort_batch(raw)
        work = build_worklists(batch)
        gpu = {}
        for name in (
            "sem_types", "is_targets", "is_padding", "number_values",
            "datetime_values", "boolean_values", "text_values", "col_name_values",
        ):
            values = batch[name]
            gpu[name] = self.device_array(values.reshape((-1,) + values.shape[2:]))
        bsz, seq = order.shape
        absolute_order = (
            order + np.arange(bsz, dtype=np.int64)[:, None] * seq
        ).reshape(-1)
        gpu["row_order"] = self.device_array(absolute_order)
        gpu_work = []
        for item in work:
            entry = {
                "qidx": self.device_array(item.qidx),
                "kidx": self.device_array(item.kidx),
                "buckets": [],
            }
            lower = 0
            for limit in (32, 64, 128, 256, 512):
                selected = np.flatnonzero((item.nk > lower) & (item.nk <= limit))
                if selected.size:
                    entry["buckets"].append((limit, {
                        name: self.device_array(getattr(item, name)[selected])
                        for name in ("qstart", "kstart", "nq", "nk", "logkv")
                    }))
                lower = limit
            gpu_work.append(entry)
        return batch, order, gpu, gpu_work

    def embed(self, gpu, rows):
        col_raw = self.tmp[:rows]
        linear_gather(
            gpu["col_name_values"].reshape(rows, D_TEXT),
            gpu["row_order"],
            self.weight("enc_dict.col_name.weight"),
            self.weight("enc_dict.col_name.bias"),
            col_raw,
        )
        rmsnorm(col_raw, self.weight("norm_dict.col_name.scale"), self.col[:rows])
        text_raw = self.tmp[:rows]
        linear_gather(
            gpu["text_values"].reshape(rows, D_TEXT),
            gpu["row_order"],
            self.weight("enc_dict.text.weight"),
            self.weight("enc_dict.text.bias"),
            text_raw,
        )
        rmsnorm(text_raw, self.weight("norm_dict.text.scale"), self.text[:rows])
        scalar_embeddings_kernel[(rows,)](
            gpu["number_values"], gpu["datetime_values"], gpu["boolean_values"],
            self.weight("enc_dict.number.weight"),
            self.weight("enc_dict.datetime.weight"),
            self.weight("enc_dict.boolean.weight"),
            self.weight("enc_dict.number.bias"),
            self.weight("enc_dict.datetime.bias"),
            self.weight("enc_dict.boolean.bias"),
            self.weight("norm_dict.number.scale"),
            self.weight("norm_dict.datetime.scale"),
            self.weight("norm_dict.boolean.scale"),
            gpu["sem_types"], self.scalar,
            rows=rows, width=D_MODEL, block=512, num_warps=8,
        )
        combine_embeddings_kernel[(rows,)](
            self.col, self.text, self.scalar,
            gpu["sem_types"], gpu["is_targets"], gpu["is_padding"],
            self.mask_embeddings, self.x,
            rows=rows, width=D_MODEL, block=512, num_warps=8,
        )

    def attention(self, block, kind, work, rows):
        prefix = f"blocks.{block}.attns.{kind}"
        linear(self.xn[:rows], self.qkvg_weights[(block, kind)], self.qkvg[:rows])
        packed = kind == "col"
        if packed:
            nkeys = work["kidx"].numel()
            prepare_kv_kernel[(nkeys, HEADS)](
                self.qkvg, work["kidx"], self.weight(f"{prefix}.k_norm.scale"),
                self.packed_k, self.packed_v,
                n_keys=nkeys, n_heads=HEADS, head_dim=HEAD_DIM,
                num_warps=2,
            )
        zero_kernel[(triton.cdiv(rows * D_MODEL, 256),)](
            self.att, total=rows * D_MODEL, block=256, num_warps=4,
        )
        for max_keys, bucket in work["buckets"]:
            nwork = bucket["nq"].numel()
            key_tile = 64 if max_keys >= 128 else 32
            relational_attention_kernel[(nwork, HEADS)](
                self.qkvg, self.packed_k, self.packed_v,
                work["qidx"], work["kidx"], bucket["qstart"], bucket["kstart"],
                bucket["nq"], bucket["nk"], bucket["logkv"],
                self.weight(f"{prefix}.q_norm.scale"),
                self.weight(f"{prefix}.k_norm.scale"),
                self.weight(f"{prefix}.scale"),
                self.att,
                n_heads=HEADS, head_dim=HEAD_DIM, packed_kv=packed,
                block_m=QUERY_TILE, block_n=key_tile, max_keys=max_keys,
                num_warps=8, num_stages=3,
            )
        linear(
            self.att[:rows], self.weight(f"{prefix}.wo.weight"), self.x[:rows],
            residual=self.x[:rows],
        )

    def forward_prepared(self, gpu, work, rows):
        key_capacity = work[0]["kidx"].numel()
        self.reserve(rows, key_capacity)
        self.embed(gpu, rows)
        for block in range(12):
            for index, kind in enumerate(("col", "feat", "nbr")):
                rmsnorm(
                    self.x[:rows],
                    self.weight(f"blocks.{block}.norms.{kind}.scale"),
                    self.xn[:rows],
                )
                self.attention(block, kind, work[index], rows)
            rmsnorm(
                self.x[:rows],
                self.weight(f"blocks.{block}.norms.ffn.scale"),
                self.xn[:rows],
            )
            linear(
                self.xn[:rows], self.weight(f"blocks.{block}.ffn.w13.weight"),
                self.ff13[:rows],
            )
            swiglu_packed_kernel[(rows, triton.cdiv(D_FF, 256))](
                self.ff13, self.ffa,
                rows=rows, width=D_FF, block=256, num_warps=4,
            )
            linear(
                self.ffa[:rows], self.weight(f"blocks.{block}.ffn.w2.weight"),
                self.x[:rows], residual=self.x[:rows],
            )
        rmsnorm(self.x[:rows], self.weight("norm_out.scale"), self.tmp[:rows])
        linear(
            self.tmp[:rows], self.weight("dec_dict.number.weight"),
            self.scores[:rows], self.weight("dec_dict.number.bias"), ieee=True,
        )
        return self.scores[:rows]

    def predict(self, raw):
        batch, _, gpu, work = self.prepare(raw)
        rows = batch["node_idxs"].size
        scores = self.forward_prepared(gpu, work, rows)
        torch.cuda.synchronize()
        all_scores = scores.float().cpu().numpy().reshape(batch["node_idxs"].shape)
        targets = batch["is_targets"].astype(bool)
        return np.asarray([
            all_scores[b][targets[b]][0] for b in range(all_scores.shape[0])
        ], np.float32)


def main():
    checkpoint = sys.argv[1]
    corpus = Path(sys.argv[2])
    files = sorted(corpus.glob("batch-*.npz"))
    started = time.perf_counter()
    model = RTJTriton(checkpoint)
    load_seconds = time.perf_counter() - started

    raw, labels = load_npz(files[0])
    started = time.perf_counter()
    first = model.predict(raw)
    compile_seconds = time.perf_counter() - started
    print(json.dumps({
        "checkpoint_load_seconds": load_seconds,
        "first_compile_and_forward_seconds": compile_seconds,
        "first_logits": first.tolist(),
        "first_labels": labels.tolist(),
    }), flush=True)

    latencies = []
    for _ in range(10):
        started = time.perf_counter()
        model.predict(raw)
        latencies.append(time.perf_counter() - started)
    print(json.dumps({
        "warm_batch_ms": {
            "p50": float(np.percentile(latencies, 50) * 1000),
            "p95": float(np.percentile(latencies, 95) * 1000),
        }
    }), flush=True)

    all_labels = []
    all_scores = []
    inference = 0.0
    for path in files:
        raw, labels = load_npz(path)
        started = time.perf_counter()
        scores = model.predict(raw)
        inference += time.perf_counter() - started
        all_labels.extend(labels.tolist())
        all_scores.extend(scores.tolist())
    labels = np.asarray(all_labels, np.uint8)
    scores = np.asarray(all_scores, np.float64)
    print(json.dumps({
        "examples": len(labels),
        "roc_auc": roc_auc(labels, scores),
        "accuracy_at_0_5": float(((scores >= 0) == labels.astype(bool)).mean()),
        "corpus_seconds": inference,
        "throughput_requests_per_second": len(labels) / inference,
    }, indent=2))


if __name__ == "__main__":
    main()
