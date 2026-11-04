# Zied Strategy Lab

An independent research sandbox; separate from `Zied-`. It does not send live OKX orders.

## Implemented
- Exactly **10 candidate strategies per cycle**.
- Generated feature catalog with **1,000+ selectable definitions**; the runtime currently computes a smaller concrete set of feature columns and samples candidates from those computed features.
- Candle anatomy, common single-/multi-candle recognizers and a broad pattern-name catalog. For broader established pattern recognition, install TA-Lib; its Pattern Recognition outputs are integrated when available. Some uncommon pattern names remain conservative placeholders without TA-Lib.
- Persistent candidate/cycle memory in JSON.
- Chronological forward-return scoring with non-overlapping holding windows and estimated transaction costs.
- Last input row is conservatively excluded as potentially unfinished.

## Run
```bash
pip install -r requirements.txt
python main.py --csv path/to/ohlcv.csv
``
CSV columns: `open, high, low, close, volume`; optional `timestamp` sorts the data. Results: `data/latest_cycle.json`; memory: `data/lab_memory.json`.

## Evaluation caveats
Current costs are configurable constants (20 bps round-trip fee plus 5 bps slippage per side by default), not live exchange-specific fee tiers. The baseline evaluator uses fixed holding-period returns; it does not yet model OHLC stop/target execution, partial fills, funding, liquidity constraints, or portfolio interactions. Results are research metrics, not forecasts or trade recommendations.
