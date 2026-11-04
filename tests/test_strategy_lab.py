import json
import numpy as np
import pandas as pd
import pytest

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
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


def test_feature_catalog_exceeds_one_thousand():
    assert len(FEATURE_CATALOG) > 1000


def test_pattern_catalog_is_present():
    assert "DOJI" in CANDLE_PATTERNS
    assert "THREE_WHITE_SOLDIERS" in CANDLE_PATTERNS


def test_features_are_finite_and_nonempty():
    features = extract_features(sample_ohlcv())
    assert len(features) > 0
    assert len(features.columns) > 0
    assert np.isfinite(features.to_numpy()).all()


def test_cycle_generates_exactly_ten_and_persists_memory(tmp_path):
    memory = tmp_path / "memory.json"
    result = discover_cycle(sample_ohlcv(), memory_path=str(memory))
    assert len(result["candidates"]) == 10
    assert result["cycle"] == 1
    assert result["research_only"] is True
    saved = json.loads(memory.read_text(encoding="utf-8"))
    assert saved["cycle"] == 1
    assert len(saved["candidates"]) == 10
    assert len({c["id"] for c in result["candidates"]}) == 10
    assert "performance" in saved
    assert len(saved["performance"]) == 10
    for candidate in result["candidates"]:
        evaluation = candidate["evaluation"]
        assert evaluation["fitness_basis"] == "validation_only"
        assert set(evaluation) >= {"train", "validation", "test", "status"}
        assert evaluation["split_points"]["train_end"] < evaluation["split_points"]["validation_end"]


def test_test_segment_is_separate_from_validation():
    from strategy_lab import _evaluate_candidate
    frame = sample_ohlcv(900)
    features = extract_features(frame)
    clean_close = frame["close"].iloc[:-1].reset_index(drop=True)
    candidate = {
        "features": ["ema_5_raw", "rsi_14_raw", "pattern_doji"],
        "direction": "long_short",
        "threshold": 0.2,
        "holding_bars": 4,
    }
    result = _evaluate_candidate(candidate, features, clean_close)
    points = result["split_points"]
    assert points["train_end"] == int(len(clean_close) * 0.60)
    assert points["validation_end"] == int(len(clean_close) * 0.80)
    assert result["fitness_basis"] == "validation_only"
    assert result["test"]["trades"] >= 0


def test_candidate_count_must_be_ten(tmp_path):
    with pytest.raises(ValueError, match="Exactly 10"):
        discover_cycle(sample_ohlcv(), memory_path=str(tmp_path / "m.json"), candidate_count=9)
