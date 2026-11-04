"""Zied Strategy Lab entry point: exactly 10 research candidates per cycle."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from strategy_lab import discover_cycle

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',required=True,help='OHLCV CSV with open/high/low/close/volume columns')
    parser.add_argument('--memory',default='data/lab_memory.json')
    parser.add_argument('--out',default='data/latest_cycle.json')
    args=parser.parse_args()
    result=discover_cycle(pd.read_csv(args.csv),memory_path=args.memory,candidate_count=10)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f"Cycle {result['cycle']} complete: {len(result['candidates'])} candidates -> {out}")

if __name__=='__main__': main()
