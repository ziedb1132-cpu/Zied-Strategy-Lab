# Zied Strategy Lab

An independent research sandbox; separate from `Zied-`. It does not send live OKX orders.

## Implemented
- Exactly **10 candidate strategies per research cycle**.
- Public OKX Spot USDT historical OHLCV download without API keys; confirmed candles only.
- Configurable instrument, candle interval, and history size.
- More than 1,000 feature definitions in the catalog; runtime selects from its computed feature columns.
- Candle anatomy, common single-/multi-candle recognizers, and optional TA-Lib pattern recognition.
- Chronological 60/20/20 train/validation/test evaluation. Candidate fitness uses validation only; test data remains held out.
- Cost-aware metrics: gross/net returns, win rate, profit factor, and maximum drawdown.
- Persistent per-candidate performance, cycle history, and latest-cycle JSON output.
- Hourly scheduled GitHub Actions research cycle plus manual dispatch; generated state is committed to `main`.

## Run locally
```bash
pip install -r requirements.txt
python main.py --inst-id BTC-USDT --bar 1H --candles 1500
```

You can also supply a local CSV instead:
```bash
python main.py --csv path/to/ohlcv.csv
```

Supported options: `--inst-id` (OKX Spot USDT pair), `--bar` (e.g. `5m`, `15m`, `1H`, `4H`, `1D`), `--candles` (120-10000), `--memory`, and `--out`.

## Automated research
The `.github/workflows/research.yml` workflow runs hourly and can also be started manually from the Actions tab. Manual inputs let you choose the instrument, interval, and number of candles. Each successful run commits `data/lab_memory.json` and `data/latest_cycle.json`, so the next cycle resumes from saved research history. The workflow uses only public market data; no OKX API key or trading permission is required.

## Evaluation caveats
The evaluator uses fixed holding-period close-to-close returns, not OHLC stop/target execution. Costs are configurable constants (20 bps round-trip fee plus 5 bps slippage per side by default), not exchange-specific account fee tiers. It does not model partial fills, liquidity constraints, or portfolio interactions. Results depend on the chosen instrument and timeframe; require substantial history and repeated positive out-of-sample performance before considering any integration. Passing tests is not evidence of profitability. Results are research metrics, not forecasts or trade recommendations.
