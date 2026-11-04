# Zied Strategy Lab

An independent research sandbox for adaptive strategy discovery. It is separate from `Zied-` and does not place live OKX orders.

## Current engine
- Generates **exactly 10 candidate strategies per discovery cycle**.
- Builds a selectable feature catalog from indicator families, periods, transforms, crossovers, candle geometry and context; the catalog is asserted to contain **more than 1,000 definitions**.
- Includes a broad catalog of classic Japanese candlestick patterns. If TA-Lib is installed, its full Pattern Recognition family is also extracted.
- Uses a persistent JSON memory file to retain prior candidates and cycle evaluations.
- Conservatively excludes the last input row as potentially unfinished and evaluates candidates using forward returns as a baseline research proxy.

## Run locally
```bash
pip install -r requirements.txt
python main.py --csv path/to/ohlcv.csv
```

CSV must contain `open, high, low, close, volume`; optional `timestamp` is used to sort rows. Outputs are written to `data/latest_cycle.json`; persistent memory is stored at `data/lab_memory.json`.

## Important limitations
This is an initial research baseline, not a production backtester. The present evaluation proxy does not yet model exchange fees, slippage, order execution, stop-loss/take-profit behavior or portfolio constraints. Do not treat candidate scores as live trade recommendations.