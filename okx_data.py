"""Public OKX Spot OHLCV downloader (no API key required)."""
from __future__ import annotations
import json
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import pandas as pd
from pathlib import Path

BASE_URL = "https://www.okx.com/api/v5"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
SUPPORTED_BARS = ["5m", "15m", "30m", "1H", "4H", "1D"]
DEFAULT_TEST_DAYS = {"5m": 90, "15m": 180, "30m": 365, "1H": 730, "4H": 1095, "1D": 1825}
BAR_SECONDS = {"5m": 300, "15m": 900, "30m": 1800, "1H": 3600, "4H": 14400, "1D": 86400}
MAX_CANDLES = 30000

def default_candle_count(bar: str) -> int:
    if bar not in DEFAULT_TEST_DAYS:
        raise ValueError(f"Unsupported OKX candle interval: {bar}")
    return min(MAX_CANDLES, int(DEFAULT_TEST_DAYS[bar] * 86400 / BAR_SECONDS[bar]))

def _request_candles(endpoint: str, params: dict[str, Any], timeout: int = 20) -> list[list[str]]:
    url = f"{BASE_URL}{endpoint}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "Zied-Strategy-Lab/1.0", "Accept": "application/json"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("code") != "0":
                raise RuntimeError(f"OKX API error {payload.get('code')}: {payload.get('msg')}")
            return payload.get("data", [])
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(float(attempt + 1))
    raise RuntimeError(f"Unable to retrieve OKX candles: {last_error}") from last_error

def fetch_okx_candles(inst_id: str = "BTC-USDT", bar: str = "1H", limit: int = 1500, timeout: int = 20) -> pd.DataFrame:
    """Fetch up to limit confirmed candles and return them oldest-first."""
    if not inst_id.endswith("-USDT") or not inst_id.replace("-", "").isalnum():
        raise ValueError("Only OKX Spot USDT instruments are supported, e.g. BTC-USDT.")
    if bar not in SUPPORTED_BARS:
        raise ValueError(f"Unsupported OKX candle interval: {bar}")
    if not 120 <= limit <= MAX_CANDLES:
        raise ValueError(f"Candle limit must be between 120 and {MAX_CANDLES}.")
    cache_dir = Path(".strategy_lab_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{inst_id}_{bar}_{limit}.csv"
    if cache_file.exists():
        try:
            cached = pd.read_csv(cache_file)
            if len(cached) >= limit:
                cached = cached.tail(limit).copy()
                cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
                return cached.reset_index(drop=True)
        except Exception:
            pass
    rows: dict[int, list[str]] = {}
    first = _request_candles("/market/candles", {"instId": inst_id, "bar": bar, "limit": min(300, limit)}, timeout)
    for row in first:
        if len(row) >= 6 and (len(row) < 9 or str(row[8]) == "1"):
            rows[int(row[0])] = row
    if not rows:
        raise RuntimeError(f"OKX returned no confirmed candles for {inst_id} ({bar}).")
    while len(rows) < limit:
        oldest = min(rows)
        page = _request_candles("/market/history-candles", {
            "instId": inst_id, "bar": bar, "after": str(oldest), "limit": min(100, limit-len(rows))
        }, timeout)
        if not page:
            break
        before = len(rows)
        for row in page:
            if len(row) >= 6 and (len(row) < 9 or str(row[8]) == "1"):
                rows[int(row[0])] = row
        if len(rows) == before:
            break
        time.sleep(0.12)
    ordered = [rows[key] for key in sorted(rows)][-limit:]
    frame = pd.DataFrame(
        [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in ordered],
        columns=COLUMNS,
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    if len(frame) < 120:
        raise RuntimeError(f"Only {len(frame)} confirmed candles available; at least 120 are required.")
    frame.to_csv(cache_file, index=False)
    return frame
