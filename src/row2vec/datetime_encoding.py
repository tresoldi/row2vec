"""Encoding of datetime columns.

A timestamp is not a number to scale and not a category to encode. Its calendar
parts are *cyclical* (23:00 is close to 00:00, December to January), so each
part that varies in the training data becomes a sine/cosine pair, and a single
standardised "elapsed time" feature carries the trend (2019 versus 2024).
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_TWO_PI = 2.0 * np.pi
_ONE_SECOND = pd.Timedelta(seconds=1)

#: Calendar parts that can become a sine/cosine pair, in output order.
_COMPONENTS = ("hour", "weekday", "day", "month")


def _to_naive_utc(series: pd.Series) -> pd.Series:
    """Return the column as timezone-naive datetimes expressed in UTC."""
    series = pd.to_datetime(series)
    if getattr(series.dt, "tz", None) is not None:
        series = series.dt.tz_convert("UTC").dt.tz_localize(None)
    return series


def _fractions(ts: pd.Series, component: str) -> np.ndarray:
    """Position of each timestamp within ``component``'s cycle, in [0, 1)."""
    dt = ts.dt
    if component == "hour":
        return np.asarray((dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0)
    if component == "weekday":
        return np.asarray((dt.dayofweek + dt.hour / 24.0) / 7.0)
    if component == "day":
        return np.asarray((dt.day - 1 + dt.hour / 24.0) / dt.days_in_month)
    return np.asarray((dt.month - 1 + (dt.day - 1) / dt.days_in_month) / 12.0)


def _varying(ts: pd.Series, component: str) -> bool:
    """Whether ``component`` takes more than one value across ``ts``."""
    dt = ts.dt
    values = {
        "hour": dt.hour,
        "weekday": dt.dayofweek,
        "day": dt.day,
        "month": dt.month,
    }[component]
    return bool(values.nunique() > 1)


class DatetimeEncoder(BaseEstimator, TransformerMixin):
    """Turn datetime columns into cyclical and trend features.

    For each column the encoder emits, in this order: a sine/cosine pair for
    each of hour, weekday, day-of-month and month that *varies in the training
    data* (a date-only column gets no hour pair), then one elapsed-time feature
    standardised with the training mean and standard deviation. Missing
    timestamps are replaced by the training median before encoding.
    Timezone-aware columns are converted to UTC.

    Attributes:
        specs_ (list[dict]): One entry per fitted column: its ``name``, the
            ``components`` encoded, and the ``mean``/``scale``/``fill`` seconds
            used for the trend feature and missing values.
        n_features_in_ (int): Number of input columns.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: Any, y: Any = None) -> "DatetimeEncoder":
        """Learn which calendar parts vary and the trend statistics."""
        frame = pd.DataFrame(X)
        specs: list[dict[str, Any]] = []
        for name in frame.columns:
            ts = _to_naive_utc(frame[name])
            valid = ts.dropna()
            if valid.empty:
                raise ValueError(f"Datetime column {name!r} has no valid timestamps")
            seconds = (valid - pd.Timestamp(0)) / _ONE_SECOND
            scale = float(seconds.std(ddof=0))
            specs.append(
                {
                    "name": name,
                    "components": [c for c in _COMPONENTS if _varying(valid, c)],
                    "mean": float(seconds.mean()),
                    "scale": scale if scale > 0 else 1.0,
                    "fill": float(seconds.median()),
                }
            )
        self.specs_ = specs
        self.n_features_in_ = len(specs)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Encode datetime columns; unseen timestamps are fine."""
        frame = pd.DataFrame(X)
        blocks: list[np.ndarray] = []
        for spec in self.specs_:
            ts = _to_naive_utc(frame[spec["name"]])
            fill = pd.Timestamp(0) + pd.Timedelta(seconds=spec["fill"])
            ts = ts.fillna(fill)
            for component in spec["components"]:
                angle = _TWO_PI * _fractions(ts, component)
                blocks.append(np.sin(angle))
                blocks.append(np.cos(angle))
            seconds = np.asarray((ts - pd.Timestamp(0)) / _ONE_SECOND, dtype=float)
            blocks.append((seconds - spec["mean"]) / spec["scale"])
        return np.column_stack(blocks) if blocks else np.empty((len(frame), 0))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Names of the output features, ``<column>_<part>_sin`` and so on."""
        names: list[str] = []
        for spec in self.specs_:
            for component in spec["components"]:
                names.extend([f"{spec['name']}_{component}_sin", f"{spec['name']}_{component}_cos"])
            names.append(f"{spec['name']}_elapsed")
        return np.asarray(names, dtype=object)
