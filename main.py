"""Zied Strategy Lab: discover BUY-only strategies across OKX Spot timeframes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from okx_data import fetch_okx_candles, SUPPORTED_BARS, default_candle_count
from strategy_lab import discover_cycle, build_chart_payload, evaluate_cross_asset_candidates, _confidence_score
from strategy_report import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only BUY strategy discovery using public OKX Spot candles.")
    parser.add_argument("--csv", help="Optional local OHLCV CSV; when omitted, fetch candles from OKX.")
    parser.add_argument("--inst-id", default="SOL-USDT", help="OKX Spot USDT instrument, e.g. SOL-USDT.")
    parser.add_argument("--bar", default="ALL", help="Candle interval or ALL for every supported OKX interval.")
    parser.add_argument("--candles", type=int, default=0, help="Confirmed historical candles; 0 selects the timeframe default.")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--cross-instruments", default="BTC-USDT,ETH-USDT,BNB-USDT", help="Comma-separated instruments used for independent cross-asset validation.")
    parser.add_argument("--memory", default="data/lab_memory.json")
    parser.add_argument("--out", default="data/latest_cycle.json")
    args = parser.parse_args()

    bars = SUPPORTED_BARS if args.bar.upper() == "ALL" else [args.bar]
    if args.csv and len(bars) > 1:
        parser.error("--csv can only be used with one --bar value.")

    runs = []
    chart_payload = None
    for bar in bars:
        try:
            if args.csv:
                frame = pd.read_csv(args.csv)
                source = {"type": "csv", "file": args.csv, "market": "spot",
                          "inst_id": args.inst_id, "bar": bar}
            else:
                requested_candles = args.candles if args.candles > 0 else default_candle_count(bar)
                frame = fetch_okx_candles(args.inst_id, bar, requested_candles, args.timeout)
                source = {"type": "okx_public_api", "exchange": "OKX", "market": "spot",
                          "inst_id": args.inst_id, "bar": bar,
                          "candles_requested": requested_candles, "candles_received": len(frame)}

            result = discover_cycle(frame, memory_path=args.memory, candidate_count=10, timeframe=bar)
            cross_ids = [x.strip().upper() for x in args.cross_instruments.split(",") if x.strip()]
            cross_frames = {}
            for cross_id in cross_ids:
                if cross_id == args.inst_id.upper():
                    continue
                try:
                    cross_frames[cross_id] = fetch_okx_candles(
                        cross_id, bar, requested_candles if not args.csv else default_candle_count(bar), args.timeout
                    )
                except Exception as cross_exc:
                    print(f"WARNING: cross-asset data skipped for {cross_id} ({bar}): {cross_exc}")
            cross_results = evaluate_cross_asset_candidates(result["candidates"], cross_frames) if cross_frames else {}
            if cross_results:
                memory_file = Path(args.memory)
                saved_memory = json.loads(memory_file.read_text(encoding="utf-8"))
                for candidate in result["candidates"]:
                    asset_result = cross_results.get(candidate["id"], {})
                    candidate["evaluation"]["cross_asset"] = asset_result
                    valid_assets = [v.get("validation", {}) for v in asset_result.values() if v.get("status") == "evaluated"]
                    positive_assets = sum(1 for v in valid_assets if v.get("net_total_return", 0) > 0)
                    candidate["evaluation"]["confidence_score"] = _confidence_score(
                        candidate["evaluation"].get("validation", {}),
                        candidate["evaluation"].get("walk_forward", {}),
                        candidate["evaluation"].get("robustness", {}),
                        candidate["evaluation"].get("monte_carlo", {}),
                        asset_result
                    )
                    stats = saved_memory.get("performance", {}).get(candidate["id"], {})
                    stats["cross_asset_validation"] = asset_result
                    stats["cross_asset_count"] = len(valid_assets)
                    stats["cross_asset_positive_count"] = positive_assets
                    stats["last_confidence_score"] = candidate["evaluation"]["confidence_score"]
                    saved_memory.get("performance", {})[candidate["id"]] = stats
                memory_file.write_text(json.dumps(saved_memory, indent=2, allow_nan=False), encoding="utf-8")
            result["cross_asset_instruments"] = list(cross_frames)
            result["data_source"] = source
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            chart_candidate = build_chart_payload(frame, args.memory, args.inst_id, bar)
            if bar == "1H" or chart_payload is None:
                chart_payload = chart_candidate
            report = build_report(args.memory, args.out, chart_payload=chart_payload)
            runs.append({"bar": bar, "status": "completed", "cycle": result["cycle"],
                         "candles": len(frame), "strategies_generated": len(result["candidates"])})
            print(f"Cycle {result['cycle']} ({bar}) complete: {len(result['candidates'])} BUY candidates; "
                  f"{report['strategies_discovered']} unique strategies across timeframes; "
                  f"{report['successful_buy_strategies']} successful.")
        except Exception as exc:
            runs.append({"bar": bar, "status": "skipped", "error": str(exc)})
            print(f"WARNING: timeframe {bar} skipped: {exc}")


    if not Path(args.out).exists():
        raise RuntimeError("No timeframe produced usable candles; no cycle report can be generated.")
    latest = json.loads(Path(args.out).read_text(encoding="utf-8"))
    latest["timeframe_runs"] = runs
    latest["all_timeframes"] = bars
    latest["data_source"]["bar"] = "ALL" if len(bars) > 1 else bars[0]
    Path(args.out).write_text(json.dumps(latest, indent=2, allow_nan=False), encoding="utf-8")
    report = build_report(args.memory, args.out, chart_payload=chart_payload)
    print(f"Completed {len(bars)} timeframe(s); {report['strategies_discovered']} unique strategies in report.")


if __name__ == "__main__":
    main()
