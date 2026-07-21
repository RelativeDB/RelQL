#!/usr/bin/env python
"""Framed-IPC worker for the reference SQL/XGBoost predictor."""

from __future__ import annotations

import pickle
import struct
import sys

import numpy as np


def _read():
    size = sys.stdin.buffer.read(8)
    if not size:
        return None
    return pickle.loads(sys.stdin.buffer.read(struct.unpack("!Q", size)[0]))


def _write(value):
    blob = pickle.dumps(value, protocol=5)
    sys.stdout.buffer.write(struct.pack("!Q", len(blob)))
    sys.stdout.buffer.write(blob)
    sys.stdout.buffer.flush()


# Exact SQL_TUNED_CLF/SQL_TUNED_REG values at reference commit
# eece04847de7b52d6fe7a718c277abec7bb18c83. Both sets are identical there and
# early_stopping_frac=0.0, so the reference fit path is a direct model.fit.
_TUNED_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=0.0,
    tree_method="hist",
    n_jobs=1,
    verbosity=0,
)


def _predict(train_features, train_labels, test_features, task_type):
    from xgboost import XGBClassifier, XGBRegressor

    if train_features is None or len(train_labels) < 2:
        return 0.5 if task_type == "clf" else 0.0
    x_train = np.asarray(train_features, dtype=np.float32)
    y_train = np.asarray(train_labels, dtype=np.float32)
    x_test = np.asarray(test_features, dtype=np.float32).reshape(1, -1)
    if task_type == "clf":
        y_int = (y_train > 0).astype(np.int32)
        if len(np.unique(y_int)) < 2:
            return float(y_int[0])
        n_pos = int(y_int.sum())
        n_neg = int(len(y_int) - n_pos)
        model = XGBClassifier(
            **_TUNED_PARAMS, objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=(n_neg / n_pos) if n_pos else 1.0,
        )
        model.fit(x_train, y_int)
        return float(model.predict_proba(x_test)[0, 1])
    model = XGBRegressor(
        **_TUNED_PARAMS, objective="reg:squarederror",
        eval_metric="mae",
    )
    model.fit(x_train, y_train)
    return float(model.predict(x_test)[0])


def main():
    while True:
        message = _read()
        if message is None:
            return
        try:
            _write((True, _predict(*message)))
        except BaseException as exc:
            _write((False, repr(exc)))


if __name__ == "__main__":
    main()
