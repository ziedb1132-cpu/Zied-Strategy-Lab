"""Public OKX Spot OHLCV downloader (no API key required)."""
from __future__ import annotations
import json
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import pandas as pd

BASE_URL = "https://www.okx.com/api/v5"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
SUPPORTED_BARS = ["1s","1m","3m","5m","15m","30m","1H","2H","4H","6H","12H","1D","2D","3D","1W","1M","3M","6Hutc","12Hutc","1Dutc","2Dutc","3Dutc","1Wutc","1Mutc","3Mutc"]

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
    if not 120 <= limit <= 10000:
        raise ValueError("Candle limit must be between 120 and 10000.")
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
    return frame
