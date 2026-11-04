# Zied Strategy Lab

An independent research sandbox; separate from `Zied-`. It does not send live OKX orders.

## Implemented
- Exactly **10 candidate strategies per cycle**.
- Generated feature catalog with **1,000+ selectable definitions**; the runtime currently computes a smaller concrete set of feature columns and samples candidates from those computed features.
- Candle anatomy, common single-/multi-candle recognizers and a broad pattern-name catalog. For broader established pattern recognition, install TA-Lib; its Pattern Recognition outputs are integrated when available. Some uncommon pattern names remain conservative placeholders without TA-Lib.
- Persistent candidate/cycle memory in JSON.
- Chronological 60/20/20 train/validation/test evaluation with non-overlapping holding windows.
- Candidate fitness is calculated from validation data only; the final test segment is held out from ranking.
- Cost-aware out-of-sample metrics: net/gross returns, win rate, profit factor, and maximum drawdown.
- Persistent per-candidate out-of-sample performance summaries in `data/lab_memory.json`.
- Last input row is conservatively excluded as potentially unfinished.

## Run
```bash
pip install -r requirements.txt
python main.py --csv path/to/ohlcv.csv
```

CSV columns: `open, high, low, close, volume`; optional `timestamp` sorts the data. Results: `data/latest_cycle.json`; memory: `data/lab_memory.json`.

## Automated checks
GitHub Actions compiles the Python files and runs the test suite on pushes, pull requests, and manual dispatches.

## Evaluation caveats
Current costs are configurable constants (20 bps round-trip fee plus 5 bps slippage per side by default), not live exchange-specific fee tiers. The evaluator still uses fixed holding-period close-to-close returns; it does not yet model OHLC stop/target execution, partial fills, funding, liquidity constraints, or portfolio interactions. Use long, clean historical datasets (preferably thousands of candles) because short validation/test segments can produce too few trades. Passing code tests is not evidence of profitability; require repeated positive out-of-sample results before considering integration. Results are research metrics, not forecasts or trade recommendations.
