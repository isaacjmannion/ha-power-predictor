"""Unit tests for the recorder-statistics processing helpers (data_processing.py)."""

import datetime

import data_processing as dp
import pandas as pd
import pytest


def _stat(start, mean):
    return {"start": start, "mean": mean}


def test_parse_start_epoch_float():
    assert dp._parse_start(0.0) == pd.Timestamp("1970-01-01 00:00:00", tz="UTC")


def test_parse_start_naive_datetime_localized_utc():
    ts = dp._parse_start(datetime.datetime(2026, 1, 1, 12, 0, 0))
    assert ts == pd.Timestamp("2026-01-01 12:00:00", tz="UTC")


def test_parse_start_aware_unchanged():
    aware = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    assert dp._parse_start(aware) == aware


def test_process_inner_join_and_temporal_features():
    base = 1_700_000_000
    power = [_stat(base + 3600 * i, 1.0 + i) for i in range(3)]
    temp = [_stat(base + 3600 * i, 20.0 + i) for i in range(3)]
    df = dp.process_ha_statistics(power, temp)
    assert list(df.columns) == [
        "timestamp", "consumption", "temperature", "year", "month", "day_of_week", "hour",
    ]
    assert df["consumption"].tolist() == [1.0, 2.0, 3.0]
    assert df["temperature"].tolist() == [20.0, 21.0, 22.0]
    assert df["hour"].iloc[0] == df["timestamp"].iloc[0].hour


def test_process_drops_none_means():
    base = 1_700_000_000
    power = [_stat(base, 1.0), _stat(base + 3600, None), _stat(base + 7200, 3.0)]
    temp = [_stat(base + 3600 * i, 20.0 + i) for i in range(3)]
    assert dp.process_ha_statistics(power, temp)["consumption"].tolist() == [1.0, 3.0]


def test_process_inner_join_only_overlap():
    base = 1_700_000_000
    power = [_stat(base + 3600 * i, float(i)) for i in range(0, 3)]
    temp = [_stat(base + 3600 * i, float(i)) for i in range(1, 4)]
    assert len(dp.process_ha_statistics(power, temp)) == 2


def test_process_no_overlap_raises():
    power = [_stat(1_700_000_000, 1.0)]
    temp = [_stat(1_700_100_000, 20.0)]
    with pytest.raises(ValueError):
        dp.process_ha_statistics(power, temp)


def test_process_empty_power_raises():
    with pytest.raises(ValueError):
        dp.process_ha_statistics([], [_stat(1_700_000_000, 20.0)])


def test_add_lagged_features_zero_returns_unchanged():
    df = pd.DataFrame({"consumption": [1.0, 2.0], "temperature": [10.0, 11.0]})
    assert dp.add_lagged_features(df, 0, 0).equals(df)


def test_add_lagged_features_shifts_and_drops_leading_rows():
    df = pd.DataFrame({
        "consumption": [1.0, 2.0, 3.0, 4.0],
        "temperature": [10.0, 11.0, 12.0, 13.0],
    })
    out = dp.add_lagged_features(df, n_power_lags=2, n_temp_lags=1)
    assert len(out) == 2
    assert {"power_lag_1", "power_lag_2", "temp_lag_1"} <= set(out.columns)
    assert out["consumption"].iloc[0] == 3.0
    assert out["power_lag_1"].iloc[0] == 2.0
    assert out["power_lag_2"].iloc[0] == 1.0
    assert out["temp_lag_1"].iloc[0] == 11.0


def test_get_default_features_linear_hour_when_no_harmonics():
    assert dp.get_default_features(0) == ["year", "month", "day_of_week", "hour", "temperature"]


def test_get_default_features_cyclical_default_two_harmonics():
    assert dp.get_default_features() == [
        "year", "month", "day_of_week",
        "hour_sin_1", "hour_cos_1", "hour_sin_2", "hour_cos_2",
        "temperature",
    ]


def test_get_default_features_one_harmonic():
    assert dp.get_default_features(1) == [
        "year", "month", "day_of_week", "hour_sin_1", "hour_cos_1", "temperature",
    ]


def test_add_cyclical_features_values_and_period():
    df = pd.DataFrame({"hour": [0, 6, 12, 18]})
    out = dp.add_cyclical_features(df, hour_harmonics=2)
    assert {"hour_sin_1", "hour_cos_1", "hour_sin_2", "hour_cos_2"} <= set(out.columns)
    # First harmonic: sin(2*pi*hour/24), cos(2*pi*hour/24)
    assert out["hour_sin_1"].tolist() == pytest.approx([0.0, 1.0, 0.0, -1.0], abs=1e-9)
    assert out["hour_cos_1"].tolist() == pytest.approx([1.0, 0.0, -1.0, 0.0], abs=1e-9)
    # Raw hour column is preserved untouched (needed for peak/off-peak routing).
    assert out["hour"].tolist() == [0, 6, 12, 18]


def test_add_cyclical_features_hour_0_and_24_match():
    # The encoding is periodic: hour 0 and a notional hour 24 map identically.
    out = dp.add_cyclical_features(pd.DataFrame({"hour": [0, 24]}), hour_harmonics=2)
    assert out["hour_sin_1"].iloc[0] == pytest.approx(out["hour_sin_1"].iloc[1], abs=1e-9)
    assert out["hour_cos_2"].iloc[0] == pytest.approx(out["hour_cos_2"].iloc[1], abs=1e-9)


def test_add_cyclical_features_zero_harmonics_noop():
    df = pd.DataFrame({"hour": [1, 2, 3]})
    assert dp.add_cyclical_features(df, hour_harmonics=0) is df


def test_build_feature_weights_groups_and_order():
    features = [
        "year", "month", "day_of_week",
        "hour_sin_1", "hour_cos_1",
        "temperature",
        "power_lag_1", "power_lag_2",
        "temp_lag_1",
    ]
    w = dp.build_feature_weights(
        features, {"time": 3.0, "temperature": 0.5, "lags": 2.0}
    )
    assert w.tolist() == [1.0, 1.0, 1.0, 3.0, 3.0, 0.5, 2.0, 2.0, 0.5]


def test_build_feature_weights_linear_hour_in_time_group():
    w = dp.build_feature_weights(["hour", "temperature"], {"time": 4.0})
    assert w.tolist() == [4.0, 1.0]


def test_build_feature_weights_missing_groups_default_neutral():
    features = ["hour_sin_1", "temperature", "power_lag_1"]
    w = dp.build_feature_weights(features, {})
    assert w.tolist() == [1.0, 1.0, 1.0]


def test_normalize_hour_offsets_list_of_rows():
    raw = [{"hour": 18, "offset": 0.8}, {"hour": 3, "offset": -0.3}]
    assert dp.normalize_hour_offsets(raw) == {18: 0.8, 3: -0.3}


def test_normalize_hour_offsets_dict_form():
    assert dp.normalize_hour_offsets({"18": 0.8, 3: -0.3}) == {18: 0.8, 3: -0.3}


def test_normalize_hour_offsets_duplicate_hour_last_wins():
    raw = [{"hour": 5, "offset": 1.0}, {"hour": 5, "offset": 2.5}]
    assert dp.normalize_hour_offsets(raw) == {5: 2.5}


def test_normalize_hour_offsets_drops_out_of_range_and_malformed():
    raw = [
        {"hour": 24, "offset": 1.0},   # hour out of range
        {"hour": -1, "offset": 1.0},   # hour out of range
        {"hour": 9, "offset": "x"},    # non-numeric offset
        {"offset": 1.0},               # missing hour
        {"hour": 10, "offset": 2.0},   # valid
    ]
    assert dp.normalize_hour_offsets(raw) == {10: 2.0}


def test_normalize_hour_offsets_empty_and_none():
    assert dp.normalize_hour_offsets([]) == {}
    assert dp.normalize_hour_offsets(None) == {}
    assert dp.normalize_hour_offsets({}) == {}
