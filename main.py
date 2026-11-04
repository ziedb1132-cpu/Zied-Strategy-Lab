"""Zied Strategy Lab: fetch public OKX candles and evaluate exactly 10 candidates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from okx_data import fetch_okx_candles
from strategy_lab import discover_cycle

def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only strategy discovery using public OKX Spot candles.")
    parser.add_argument("--csv", help="Optional local OHLCV CSV; when omitted, fetch candles from OKX.")
    parser.add_argument("--inst-id", default="SOL-USDT", help="OKX Spot USDT instrument, e.g. SOL-USDT.")
    parser.add_argument("--bar", default="1H", help="OKX candle interval, e.g. 5m, 15m, 1H, 4H, 1D.")
    parser.add_argument("--candles", type=int, default=1500, help="Confirmed historical candles (120-10000).")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--memory", default="data/lab_memory.json")
    parser.add_argument("--out", default="data/latest_cycle.json")
    args = parser.parse_args()
    if args.csv:
        frame = pd.read_csv(args.csv)
        source = {"type": "csv", "file": args.csv}
    else:
        frame = fetch_okx_candles(args.inst_id, args.bar, args.candles, args.timeout)
        source = {"type": "okx_public_api", "exchange": "OKX", "market": "spot", "inst_id": args.inst_id,
                  "bar": args.bar, "candles_requested": args.candles, "candles_received": len(frame)}
    result = discover_cycle(frame, memory_path=args.memory, candidate_count=10)
    result["data_source"] = source
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Cycle {result['cycle']} complete: {len(result['candidates'])} candidates; "
          f"{source['type']} data; {len(frame)} candles -> {out}")

if __name__ == "__main__":
    main()
