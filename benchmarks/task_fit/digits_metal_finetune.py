"""Frozen-backbone RT-J multiclass fine-tuning on sklearn Digits.

Digits is not present in RT-J's 485-database recipe.  This benchmark uses a
fixed stratified 80/20 split, computes normalization statistics on the train
split only, extracts RT-J target features on Metal, and trains the native C++
multiclass head on Metal.  No PyTorch operation participates in optimization.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
D_MODEL, D_TEXT, MAX_F2P = 512, 384, 5
SEM_NUMBER, SEM_TEXT = 0, 1
RT_DEVICE_MPS = 1
RT_FINETUNE_MULTICLASS = 2
CLASS_NAMES = ["zero", "one", "two", "three", "four",
               "five", "six", "seven", "eight", "nine"]


def parse_args():
    default_ckpt = (Path.home() / ".cache/huggingface/hub/"
                    "models--stanford-star--rt-j/snapshots/"
                    "1f552a738a0f8dada8af77db913b2b90511e2f00/"
                    "classification/model.safetensors")
    p = argparse.ArgumentParser()
    p.add_argument("--lib", type=Path, default=ROOT / "cpp/build/librt_c.dylib")
    p.add_argument("--checkpoint", type=Path, default=default_ckpt)
    p.add_argument("--adapter", type=Path,
                   default=ROOT / "benchmarks/task_fit/digits_head.safetensors")
    p.add_argument("--results", type=Path,
                   default=ROOT / "benchmarks/task_fit/digits_results.json")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64,
                   help="RT-J feature-extraction batch size")
    return p.parse_args()


def bind(path: Path):
    lib = ctypes.CDLL(str(path))
    f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    i64p = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
    u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
    lib.rt_device_available.argtypes = [ctypes.c_int32]
    lib.rt_device_available.restype = ctypes.c_int
    lib.rt_model_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.rt_model_load.restype = ctypes.c_void_p
    lib.rt_model_free.argtypes = [ctypes.c_void_p]
    lib.rt_encode_targets_device.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        i64p, i64p, i64p, i64p, u8p, i64p, u8p,
        f32p, f32p, f32p, f32p, f32p,
        ctypes.c_int32, ctypes.c_int32, f32p,
        ctypes.c_char_p, ctypes.c_size_t]
    lib.rt_encode_targets_device.restype = ctypes.c_int
    lib.rt_finetune_head_create.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, f32p,
        ctypes.c_char_p, ctypes.c_size_t]
    lib.rt_finetune_head_create.restype = ctypes.c_void_p
    lib.rt_finetune_head_free.argtypes = [ctypes.c_void_p]
    lib.rt_finetune_head_predict.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, f32p, f32p,
        ctypes.c_char_p, ctypes.c_size_t]
    lib.rt_finetune_head_predict.restype = ctypes.c_int
    # Nullable ranking offsets are represented as void* here; multiclass passes NULL.
    lib.rt_finetune_head_fit_metal.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, f32p, f32p,
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_size_t]
    lib.rt_finetune_head_fit_metal.restype = ctypes.c_int
    lib.rt_finetune_head_save.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.rt_finetune_head_save.restype = ctypes.c_int
    return lib


def check(rc: int, err, what: str):
    if rc:
        raise RuntimeError(f"{what}: {err.value.decode('utf-8', 'replace')}")


def make_schema_embeddings():
    from sentence_transformers import SentenceTransformer
    # The same pinned encoder as RT-J, cache-only so the benchmark never
    # silently downloads or changes its text representation.
    encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L12-v2", local_files_only=True)
    phrases = [f"pixel_{r}_{c} of digits" for r in range(8) for c in range(8)]
    phrases.append("label of task")
    col = np.asarray(encoder.encode(phrases, show_progress_bar=False),
                     dtype=np.float32)
    classes = np.asarray(encoder.encode(
        CLASS_NAMES, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32)
    return col, classes


def encode_features(lib, model, X: np.ndarray, col_emb: np.ndarray,
                    batch_size: int) -> np.ndarray:
    S = 65
    result = np.empty((len(X), D_MODEL), np.float32)
    for start in range(0, len(X), batch_size):
        xb = np.ascontiguousarray(X[start:start + batch_size], np.float32)
        B = len(xb)
        node = np.zeros((B, S), np.int64)
        node[:, -1] = 1
        f2p = np.full((B, S, MAX_F2P), -1, np.int64)
        f2p[:, -1, 0] = 0
        col = np.broadcast_to(np.arange(S, dtype=np.int64), (B, S)).copy()
        table = np.zeros((B, S), np.int64)
        table[:, -1] = 1
        padding = np.zeros((B, S), np.uint8)
        sem = np.full((B, S), SEM_NUMBER, np.int64)
        sem[:, -1] = SEM_TEXT
        target = np.zeros((B, S), np.uint8)
        target[:, -1] = 1
        number = np.zeros((B, S), np.float32)
        number[:, :64] = xb
        zeros = np.zeros((B, S), np.float32)
        text = np.zeros((B, S, D_TEXT), np.float32)
        col_names = np.ascontiguousarray(
            np.broadcast_to(col_emb, (B, S, D_TEXT)), np.float32)
        out = np.empty((B, D_MODEL), np.float32)
        err = ctypes.create_string_buffer(1024)
        rc = lib.rt_encode_targets_device(
            model, B, S, node, f2p, col, table, padding, sem, target,
            number, zeros, zeros, text, col_names, 0, RT_DEVICE_MPS, out,
            err, len(err))
        check(rc, err, "rt_encode_targets_device")
        result[start:start + B] = out
        print(f"feature extraction {start + B:4d}/{len(X)}", flush=True)
    return result


def predict(lib, head, features, n_classes=10):
    features = np.ascontiguousarray(features, np.float32)
    out = np.empty((len(features), n_classes), np.float32)
    err = ctypes.create_string_buffer(1024)
    check(lib.rt_finetune_head_predict(head, len(features), features, out,
                                       err, len(err)), err, "head predict")
    return out


def metrics(y, logits):
    pred = logits.argmax(axis=1)
    order = np.argsort(-logits, axis=1)
    ranks = np.argmax(order == y[:, None], axis=1) + 1
    z = logits.astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "cross_entropy": float(log_loss(y, p, labels=np.arange(10))),
        # Viewing the ten classes as a candidate set exercises the same score
        # ordering consumed by ranking inference.
        "ranking_mrr": float(np.mean(1.0 / ranks)),
        "ranking_recall_at_3": float(np.mean(ranks <= 3)),
    }


def main():
    a = parse_args()
    lib = bind(a.lib)
    if not lib.rt_device_available(RT_DEVICE_MPS):
        raise RuntimeError("Metal device unavailable (run outside a restricted sandbox)")
    err = ctypes.create_string_buffer(1024)
    model = lib.rt_model_load(os.fsencode(a.checkpoint), err, len(err))
    if not model:
        raise RuntimeError(f"model load: {err.value.decode()}")
    head = None
    try:
        digits = load_digits()
        indices = np.arange(len(digits.target))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.20, random_state=1729,
            stratify=digits.target)
        X_train = digits.data[train_idx].astype(np.float32)
        X_test = digits.data[test_idx].astype(np.float32)
        y_train = digits.target[train_idx].astype(np.float32)
        y_test = digits.target[test_idx].astype(np.int64)
        # Train-only normalization; constant pixels remain zero.
        mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
        sd[sd < 1e-6] = 1.0
        X_train = (X_train - mu) / sd
        X_test = (X_test - mu) / sd

        col_emb, class_emb = make_schema_embeddings()
        train_features = encode_features(lib, model, X_train, col_emb,
                                          a.batch_size)
        test_features = encode_features(lib, model, X_test, col_emb,
                                         a.batch_size)
        head = lib.rt_finetune_head_create(
            model, RT_FINETUNE_MULTICLASS, 10,
            np.ascontiguousarray(class_emb), err, len(err))
        if not head:
            raise RuntimeError(f"head create: {err.value.decode()}")

        before = metrics(y_test, predict(lib, head, test_features))
        initial, final = ctypes.c_float(), ctypes.c_float()
        seconds = ctypes.c_double()
        rc = lib.rt_finetune_head_fit_metal(
            head, len(train_features),
            np.ascontiguousarray(train_features),
            np.ascontiguousarray(y_train), None, 0, a.epochs,
            a.learning_rate, a.weight_decay,
            ctypes.byref(initial), ctypes.byref(final), ctypes.byref(seconds),
            err, len(err))
        check(rc, err, "Metal fine-tune")
        after = metrics(y_test, predict(lib, head, test_features))
        check(lib.rt_finetune_head_save(
            head, os.fsencode(a.adapter), err, len(err)), err, "head save")

        result = {
            "dataset": "sklearn.datasets.load_digits",
            "not_in_rt_j_recipe": True,
            "recipe_audit": "no configuration containing 'digits' in recipe_rt_j.txt",
            "checkpoint": "stanford-star/rt-j@1f552a738a0f8dada8af77db913b2b90511e2f00/classification",
            "split": {"seed": 1729, "train": len(train_idx),
                      "test": len(test_idx), "stratified": True},
            "device": "Metal",
            "method": "frozen RT-J backbone + trainable 10x512 linear head",
            "trainable_parameters": 10 * D_MODEL + 10,
            "epochs": a.epochs,
            "learning_rate": a.learning_rate,
            "weight_decay": a.weight_decay,
            "training_loss": {"before": float(initial.value),
                              "after": float(final.value)},
            "training_seconds": seconds.value,
            "before": before,
            "after": after,
            "adapter": str(a.adapter.relative_to(ROOT)),
        }
        a.results.write_text(json.dumps(result, indent=2) + "\n")
        print("\n" + json.dumps(result, indent=2))
    finally:
        if head:
            lib.rt_finetune_head_free(head)
        lib.rt_model_free(model)


if __name__ == "__main__":
    main()
