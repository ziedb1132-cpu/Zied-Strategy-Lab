"""Zied Strategy Lab: discover BUY-only strategies across OKX Spot timeframes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from okx_data import fetch_okx_candles, SUPPORTED_BARS
from strategy_lab import discover_cycle
from strategy_report import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only BUY strategy discovery using public OKX Spot candles.")
    parser.add_argument("--csv", help="Optional local OHLCV CSV; when omitted, fetch candles from OKX.")
    parser.add_argument("--inst-id", default="SOL-USDT", help="OKX Spot USDT instrument, e.g. SOL-USDT.")
    parser.add_argument("--bar", default="ALL", help="Candle interval or ALL for every supported OKX interval.")
    parser.add_argument("--candles", type=int, default=1500, help="Confirmed historical candles (120-10000).")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--memory", default="data/lab_memory.json")
    parser.add_argument("--out", default="data/latest_cycle.json")
    args = parser.parse_args()

    bars = SUPPORTED_BARS if args.bar.upper() == "ALL" else [args.bar]
    if args.csv and len(bars) > 1:
        parser.error("--csv can only be used with one --bar value.")

    runs = []
    for bar in bars:
        if args.csv:
            frame = pd.read_csv(args.csv)
            source = {"type": "csv", "file": args.csv, "market": "spot",
                      "inst_id": args.inst_id, "bar": bar}
        else:
            frame = fetch_okx_candles(args.inst_id, bar, args.candles, args.timeout)
            source = {"type": "okx_public_api", "exchange": "OKX", "market": "spot",
                      "inst_id": args.inst_id, "bar": bar,
                      "candles_requested": args.candles, "candles_received": len(frame)}

        result = discover_cycle(frame, memory_path=args.memory, candidate_count=10, timeframe=bar)
        result["data_source"] = source
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        report = build_report(args.memory, args.out)
        runs.append({"bar": bar, "cycle": result["cycle"], "candles": len(frame),
                     "strategies_generated": len(result["candidates"])})
        print(f"Cycle {result['cycle']} ({bar}) complete: {len(result['candidates'])} BUY candidates; "
              f"{report['strategies_discovered']} unique strategies across timeframes; "
              f"{report['successful_buy_strategies']} successful.")

    latest = json.loads(Path(args.out).read_text(encoding="utf-8"))
    latest["timeframe_runs"] = runs
    latest["all_timeframes"] = bars
    latest["data_source"]["bar"] = "ALL" if len(bars) > 1 else bars[0]
    Path(args.out).write_text(json.dumps(latest, indent=2, allow_nan=False), encoding="utf-8")
    report = build_report(args.memory, args.out)
    print(f"Completed {len(bars)} timeframe(s); {report['strategies_discovered']} unique strategies in report.")


if __name__ == "__main__":
    main()
