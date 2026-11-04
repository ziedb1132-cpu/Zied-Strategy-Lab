import json
import numpy as np
import pandas as pd
import pytest

from okx_data import fetch_okx_candles, SUPPORTED_BARS
from strategy_lab import FEATURE_CATALOG, CANDLE_PATTERNS, extract_features, discover_cycle


def sample_ohlcv(n=360):
    rng = np.random.default_rng(17)
    returns = rng.normal(0.0003, 0.012, n)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.006, 0.003, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(8, 0.4, n)
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
                         "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_feature_catalog_exceeds_one_thousand():
    assert len(FEATURE_CATALOG) > 1000


def test_pattern_catalog_is_present():
    assert "DOJI" in CANDLE_PATTERNS
    assert "THREE_WHITE_SOLDIERS" in CANDLE_PATTERNS


def test_features_are_finite_and_nonempty():
    features = extract_features(sample_ohlcv())
    assert len(features) > 0 and len(features.columns) > 0
    assert np.isfinite(features.to_numpy()).all()


def test_cycle_generates_exactly_ten_and_persists_memory(tmp_path):
    memory = tmp_path / "memory.json"
    result = discover_cycle(sample_ohlcv(), memory_path=str(memory))
    assert len(result["candidates"]) == 10 and result["cycle"] == 1
    assert result["research_only"] is True
    saved = json.loads(memory.read_text(encoding="utf-8"))
    assert saved["cycle"] == 1 and len(saved["candidates"]) == 10
    assert len({c["id"] for c in result["candidates"]}) == 10
    assert all(c["direction"] == "long" for c in result["candidates"])
    assert "performance" in saved and len(saved["performance"]) == 10
    for candidate in result["candidates"]:
        evaluation = candidate["evaluation"]
        assert evaluation["fitness_basis"] == "validation_only"
        assert set(evaluation) >= {"train", "validation", "test", "status"}
        assert evaluation["split_points"]["train_end"] < evaluation["split_points"]["validation_end"]


def test_test_segment_is_separate_from_validation():
    from strategy_lab import _evaluate_candidate
    frame = sample_ohlcv(900)
    features = extract_features(frame)
    candidate = {"features": ["ema_5_raw", "rsi_14_raw", "pattern_doji"],
                 "direction": "long", "threshold": 0.2, "holding_bars": 4}
    result = _evaluate_candidate(candidate, features, frame["close"].iloc[:-1].reset_index(drop=True))
    points = result["split_points"]
    assert points["train_end"] == int((len(frame)-1) * 0.60)
    assert points["validation_end"] == int((len(frame)-1) * 0.80)
    assert result["fitness_basis"] == "validation_only"
    assert result["test"]["trades"] >= 0


def test_candidate_count_must_be_ten(tmp_path):
    with pytest.raises(ValueError, match="Exactly 10"):
        discover_cycle(sample_ohlcv(), memory_path=str(tmp_path / "m.json"), candidate_count=9)


def test_okx_candle_fetcher_parses_confirmed_rows(monkeypatch):
    import okx_data
    rows = [[str(1735689600000 + i * 3600000), "100", "105", "99", str(103 + i * 0.01),
             "12", "12", "1236", "1"] for i in range(120)]
    rows.append(["1736121600000", "103", "106", "101", "104", "15", "15", "1560", "0"])
    monkeypatch.setattr(okx_data, "_request_candles", lambda endpoint, params, timeout=20: rows)
    frame = fetch_okx_candles("BTC-USDT", "1H", 120)
    assert len(frame) == 120 and frame.iloc[0]["close"] == 103.0
    assert str(frame.iloc[0]["timestamp"].tz) == "UTC"


def test_okx_candle_fetcher_rejects_non_usdt_pairs():
    with pytest.raises(ValueError, match="USDT"):
        fetch_okx_candles("BTC-USDC", "1H", 120)



def test_all_okx_timeframes_are_declared():
    assert {"5m", "15m", "30m", "1H", "4H", "1D"} == set(SUPPORTED_BARS)


def test_strategy_memory_keeps_timeframes_separate(tmp_path):
    memory = tmp_path / "memory.json"
    frame = sample_ohlcv()
    first = discover_cycle(frame, memory_path=str(memory), timeframe="1H")
    second = discover_cycle(frame, memory_path=str(memory), timeframe="4H")
    saved = json.loads(memory.read_text(encoding="utf-8"))
    assert all(item["bar"] == "1H" for item in first["candidates"])
    assert all(item["bar"] == "4H" for item in second["candidates"])
    assert saved["timeframe_cycles"] == {"1H": 1, "4H": 1}
    assert len(saved["performance"]) == 20
