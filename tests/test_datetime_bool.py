"""Datetime and boolean columns: encoding, pipeline routing, and persistence."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import row2vec
from row2vec.datetime_encoding import DatetimeEncoder
from row2vec.pipeline_builder import PipelineBuilder
from row2vec.serialization import load_model, save_model
from row2vec.utils import boolean_columns, datetime_columns


def _frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "when": pd.date_range("2024-01-01", periods=n, freq="37h"),
            "flag": rng.random(n) > 0.5,
        }
    )


def test_predicates() -> None:
    df = _frame()
    assert datetime_columns(df) == ["when"]
    assert boolean_columns(df) == ["flag"]


def test_cyclical_wraparound() -> None:
    enc = DatetimeEncoder().fit(pd.DataFrame({"t": pd.date_range("2023-01-01", periods=400)}))
    out = enc.transform(
        pd.DataFrame({"t": pd.to_datetime(["2023-12-31", "2024-01-01", "2024-07-01"])})
    )
    names = list(enc.get_feature_names_out())
    month = [names.index("t_month_sin"), names.index("t_month_cos")]
    near = np.linalg.norm(out[0, month] - out[1, month])
    far = np.linalg.norm(out[0, month] - out[2, month])
    assert near < 0.2 < far


def test_hour_wraparound_and_constant_components_dropped() -> None:
    times = pd.to_datetime("2024-03-01") + pd.to_timedelta(np.arange(48), unit="h")
    enc = DatetimeEncoder().fit(pd.DataFrame({"t": times}))
    names = list(enc.get_feature_names_out())
    assert "t_hour_sin" in names
    assert "t_month_sin" not in names  # one month only: nothing to encode
    out = enc.transform(
        pd.DataFrame({"t": pd.to_datetime(["2024-03-01 23:00", "2024-03-01 00:00"])})
    )
    cols = [names.index("t_hour_sin"), names.index("t_hour_cos")]
    assert np.linalg.norm(out[0, cols] - out[1, cols]) < 0.3


def test_trend_is_standardised_and_monotonic() -> None:
    dates = pd.date_range("2019-01-01", "2024-01-01", freq="30D")
    enc = DatetimeEncoder().fit(pd.DataFrame({"t": dates}))
    out = enc.transform(pd.DataFrame({"t": dates}))
    trend = out[:, list(enc.get_feature_names_out()).index("t_elapsed")]
    assert abs(trend.mean()) < 1e-9
    assert np.all(np.diff(trend) > 0)


def test_missing_and_timezones_and_unseen() -> None:
    aware = pd.Series(pd.date_range("2024-01-01", periods=30, freq="7h", tz="Europe/Rome"))
    aware = aware.mask(aware.index == 3)
    enc = DatetimeEncoder().fit(pd.DataFrame({"t": aware}))
    out = enc.transform(pd.DataFrame({"t": aware}))
    assert np.isfinite(out).all()
    utc = DatetimeEncoder().fit(
        pd.DataFrame({"t": aware.dt.tz_convert("UTC").dt.tz_localize(None)})
    )
    assert np.allclose(
        utc.transform(pd.DataFrame({"t": aware.dt.tz_convert("UTC").dt.tz_localize(None)})), out
    )
    future = (
        DatetimeEncoder()
        .fit(pd.DataFrame({"t": aware}))
        .transform(pd.DataFrame({"t": pd.to_datetime(["2040-06-15 12:00"]).tz_localize("UTC")}))
    )
    assert np.isfinite(future).all()


def test_all_missing_datetime_column_raises() -> None:
    with pytest.raises(ValueError, match="no valid timestamps"):
        DatetimeEncoder().fit(
            pd.DataFrame({"t": pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]")})
        )


def test_pipeline_routes_columns_and_reports() -> None:
    df = _frame()
    builder = PipelineBuilder()
    pre, analysis = builder.build_preprocessing_pipeline(df)
    out = pre.fit_transform(df)
    assert {name for name, *_ in pre.transformers} == {"numeric", "boolean", "datetime"}
    assert analysis["datetime_columns"] == 1
    assert analysis["boolean_columns"] == 1
    assert builder.get_pipeline_description()["datetime_processing"]
    assert set(np.unique(out[:, [list(pre.get_feature_names_out()).index("boolean__flag")]])) <= {
        0.0,
        1.0,
    }


def test_nullable_boolean_missing_gets_mode() -> None:
    df = _frame()
    df["flag"] = pd.array([True] * 40 + [pd.NA] * 5 + [False] * 15, dtype="boolean")
    pre, _ = PipelineBuilder().build_preprocessing_pipeline(df)
    out = pre.fit_transform(df)
    col = out[:, list(pre.get_feature_names_out()).index("boolean__flag")]
    assert np.isfinite(col).all()
    assert col[40:45].tolist() == [1.0] * 5


def test_unsupported_dtype_warns() -> None:
    df = _frame()
    df["gap"] = pd.period_range("2024-01", periods=len(df), freq="M")
    with pytest.warns(UserWarning, match="gap"):
        PipelineBuilder().build_preprocessing_pipeline(df)


def test_no_warning_for_supported_columns() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        PipelineBuilder().build_preprocessing_pipeline(_frame())


def test_datetime_only_frame_embeds() -> None:
    df = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=40, freq="13h")})
    emb = row2vec.learn_embedding(df, mode="pca", embedding_dim=2, enable_logging=False)
    assert emb.shape == (40, 2)


def test_target_mode_excludes_datetime_target() -> None:
    df = _frame()
    pre, _ = PipelineBuilder().build_preprocessing_pipeline(df, target=df["flag"], mode="target")
    assert "boolean" not in {name for name, *_ in pre.transformers}


def test_save_load_round_trip(tmp_path: Path) -> None:
    df = _frame()
    _, model = row2vec.learn_embedding_with_model(
        df, mode="pca", embedding_dim=3, enable_logging=False
    )
    path = save_model(model, tmp_path / "dt")
    loaded = load_model(path)
    new = _frame(20).assign(when=pd.date_range("2030-05-05", periods=20, freq="5h"))
    expected = model.predict(new, validate_schema=False)
    got = loaded.predict(new, validate_schema=False)
    assert np.allclose(expected.to_numpy(), got.to_numpy())
