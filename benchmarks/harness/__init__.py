"""Temporal-backtest evaluation harness for relativedb / RelQL.

Measures whether the engine's predictions carry real predictive signal on
*real, held-out future data*, using a strict point-in-time backtest:

    context  := rows with timestamp <= anchor T   (engine-enforced)
    target   := RelQL window (T, T+h]
    truth    := the actual outcome computed from real future rows
    metric   := prediction vs. truth (AUROC, PR-AUC, Brier, MAE, Recall@K, ...)

Nothing here reaches into engine internals: predictions come out of
``Engine.execute`` exactly as a user would call it. Ground truth is computed
independently from the raw frames so a leaky engine would be *caught*, not
flattered, by the comparison.
"""
