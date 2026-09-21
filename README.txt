
## Evaluation and backtest metrics

Run `python train.py` after placing the OHLCV training file at
`data/asset_alpha_training.csv`. The chronological holdout evaluation now reports:

- **Directional accuracy** — percentage of next-bar up/down moves predicted correctly.
- **Sharpe ratio** — annualised risk-adjusted return from the test-period strategy.
- **Backtest result** — total return and trade count on unseen data.
- **Maximum drawdown before vs. after filters** — compares the raw directional
  strategy with the confidence thresholds (`BUY >= 0.55`, `SELL <= 0.45`) and the
  volatility-spike guard (skip bars above 3x the trailing 30-bar volatility median).

The same metrics are written to `backtest_results.json`; no numbers are hard-coded,
so results are generated from the supplied dataset. `python backtest.py` can also
evaluate a CSV containing `probability` and `close` (or `future_return`) columns.
