"""Walk-forward evaluation metrics and risk-filter comparison.

This module deliberately contains no model-training code.  It evaluates predictions
on an already-unseen, chronological sample so reported metrics cannot leak labels
from the future.

Usage:
    python backtest.py data/asset_alpha_training.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BUY_THRESHOLD = 0.55
DEFAULT_SELL_THRESHOLD = 0.45


def max_drawdown(equity: pd.Series) -> float:
    """Return maximum peak-to-trough drawdown as a positive fraction."""
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float(((peak - equity) / peak.replace(0, np.nan)).fillna(0).max())


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualised Sharpe using simple periodic returns."""
    clean = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2 or clean.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * clean.mean() / clean.std(ddof=1))


def _strategy_returns(
    probabilities: pd.Series,
    future_returns: pd.Series,
    volatility: pd.Series | None,
    filtered: bool,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.Series:
    """Create next-period strategy returns, optionally applying risk filters."""
    p = pd.Series(probabilities, dtype="float64").reset_index(drop=True)
    r = pd.Series(future_returns, dtype="float64").reset_index(drop=True)
    signal = np.where(p >= 0.5, 1.0, -1.0)

    if filtered:
        # Low-conviction HOLD zone, equivalent to the live signal thresholds.
        signal = np.where(p >= buy_threshold, 1.0,
                          np.where(p <= sell_threshold, -1.0, 0.0))
        # Volatility guard: skip abnormal test bars rather than entering into a
        # spike.  The threshold is learned from past bars only.
        if volatility is not None:
            vol = pd.Series(volatility, dtype="float64").reset_index(drop=True)
            baseline = vol.shift(1).rolling(30, min_periods=5).median()
            signal = np.where((vol > 3 * baseline.fillna(vol.median())) | ~np.isfinite(vol), 0.0, signal)

    return pd.Series(signal * r, name="strategy_return")


def evaluate_predictions(
    probabilities: pd.Series,
    actual_up: pd.Series,
    future_returns: pd.Series,
    volatility: pd.Series | None = None,
    periods_per_year: int = 252,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    sell_threshold: float = DEFAULT_SELL_THRESHOLD,
) -> dict[str, Any]:
    """Compare directional model performance before and after risk filters."""
    p = pd.Series(probabilities).reset_index(drop=True)
    y = pd.Series(actual_up).astype(int).reset_index(drop=True)
    raw_direction = (p >= 0.5).astype(int)
    raw = _strategy_returns(p, future_returns, volatility, False, buy_threshold, sell_threshold)
    filtered = _strategy_returns(p, future_returns, volatility, True, buy_threshold, sell_threshold)

    def report(returns: pd.Series) -> dict[str, float]:
        equity = (1.0 + returns.fillna(0)).cumprod()
        return {
            "sharpe": round(sharpe_ratio(returns, periods_per_year), 4),
            "max_drawdown_pct": round(max_drawdown(equity) * 100, 4),
            "total_return_pct": round((equity.iloc[-1] - 1) * 100, 4) if len(equity) else 0.0,
            "trades": int((returns != 0).sum()),
        }

    return {
        "directional_accuracy_pct": round(float((raw_direction == y).mean() * 100), 4),
        "before_filters": report(raw),
        "after_filters": report(filtered),
        "filter_config": {
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "volatility_guard": "skip bar when volatility > 3x trailing 30-bar median",
        },
    }


def run_csv_backtest(path: str, output: str = "backtest_results.json") -> dict[str, Any]:
    """Run a model-free baseline backtest from a CSV probability column.

    The CSV must contain ``probability`` and either ``close`` or ``future_return``.
    This is useful for evaluating saved prediction files without retraining.
    """
    df = pd.read_csv(path)
    if "probability" not in df:
        raise ValueError("CSV must contain a 'probability' column")
    if "future_return" not in df:
        if "close" not in df:
            raise ValueError("CSV must contain 'close' or 'future_return'")
        df["future_return"] = df["close"].shift(-1).div(df["close"]).sub(1)
    df = df.dropna(subset=["probability", "future_return"])
    actual_up = (df["future_return"] > 0).astype(int)
    result = evaluate_predictions(df["probability"], actual_up, df["future_return"], df.get("volatility"))
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate directional accuracy and risk-filtered backtest metrics")
    parser.add_argument("csv", help="CSV containing probability and close/future_return columns")
    parser.add_argument("--output", default="backtest_results.json")
    args = parser.parse_args()
    print(json.dumps(run_csv_backtest(args.csv, args.output), indent=2))
