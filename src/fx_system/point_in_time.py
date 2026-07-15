from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .factor_config import PointInTimeConfig
from .models import CurrencyPair

RATE_COLUMNS = ("policy_rate", "ois_1m", "ois_3m")
FORWARD_COLUMNS = ("forward_points_1m", "forward_points_3m", "spot_reference")


@dataclass(frozen=True)
class PointInTimeData:
    currency_rates: pd.DataFrame
    forward_points: pd.DataFrame
    maximum_staleness_days: int = 45

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, frame in (
            ("currency_rates", self.currency_rates),
            ("forward_points", self.forward_points),
        ):
            digest.update(name.encode())
            ordered = frame.sort_values(list(frame.columns[:3])).reset_index(drop=True)
            digest.update(pd.util.hash_pandas_object(ordered, index=True).values.tobytes())
        return digest.hexdigest()


def _normalize_timestamp_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"point-in-time data is missing {column!r}")
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if values.isna().any():
        raise ValueError(f"point-in-time data contains invalid {column!r} values")
    return values


def _validate_point_in_time_frame(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    value_columns: tuple[str, ...],
) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    result["observation_time"] = _normalize_timestamp_column(result, "observation_time")
    result["available_time"] = _normalize_timestamp_column(result, "available_time")
    if (result["observation_time"] > result["available_time"]).any():
        raise ValueError("observation_time cannot be later than available_time")
    if entity_column not in result:
        raise ValueError(f"point-in-time data is missing {entity_column!r}")
    result[entity_column] = result[entity_column].astype(str).str.upper()
    if entity_column == "currency":
        invalid = ~result[entity_column].str.fullmatch(r"[A-Z]{3}")
        if invalid.any():
            raise ValueError("currency must contain three-letter ISO codes")
    else:
        result[entity_column] = result[entity_column].map(
            lambda value: CurrencyPair.parse(value).symbol
        )
    missing = set(value_columns) - set(result)
    if missing:
        raise ValueError(f"point-in-time data is missing values {sorted(missing)}")
    for column in value_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(result[column].dropna()).all():
            raise ValueError(f"point-in-time data contains invalid {column!r} values")
    if result.duplicated([entity_column, "available_time"]).any():
        raise ValueError(f"duplicate {entity_column}/available_time point-in-time rows")
    return result.sort_values([entity_column, "available_time", "observation_time"]).reset_index(
        drop=True
    )


def validate_currency_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate revision-aware currency rates expressed in percentage points."""
    return _validate_point_in_time_frame(
        frame,
        entity_column="currency",
        value_columns=RATE_COLUMNS,
    )


def validate_forward_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate outright forward points expressed in the pair's price units."""
    return _validate_point_in_time_frame(
        frame,
        entity_column="symbol",
        value_columns=FORWARD_COLUMNS,
    )


def _synthetic_point_in_time_data(
    data: Mapping[str, pd.DataFrame], config: PointInTimeConfig
) -> PointInTimeData:
    """Deterministic PIT fixtures for software validation, never empirical evidence."""
    generator = np.random.default_rng(config.synthetic_seed)
    symbols = sorted(data)
    pairs = [CurrencyPair.parse(symbol) for symbol in symbols]
    currencies = sorted({currency for pair in pairs for currency in (pair.base, pair.quote)})
    common = sorted(set().union(*(frame.index for frame in data.values())))
    release_times = common[::20]
    base_rates = {
        "USD": 3.0,
        "EUR": 1.5,
        "GBP": 2.5,
        "JPY": 0.1,
        "CHF": 0.5,
        "CAD": 2.0,
        "AUD": 2.2,
        "NZD": 2.4,
    }
    rate_rows: list[dict[str, object]] = []
    for currency_number, currency in enumerate(currencies):
        innovations = generator.normal(0, 0.03, len(release_times)).cumsum()
        for location, (timestamp, innovation) in enumerate(
            zip(release_times, innovations, strict=True)
        ):
            timestamp = pd.Timestamp(timestamp)
            policy = base_rates.get(currency, 1.0) + innovation
            rate_rows.append(
                {
                    "observation_time": timestamp,
                    "available_time": timestamp + timedelta(microseconds=1),
                    "currency": currency,
                    "policy_rate": policy,
                    "ois_1m": policy + 0.02 * np.sin((location + currency_number) / 4),
                    "ois_3m": policy + 0.05 * np.cos((location + currency_number) / 6),
                }
            )
    rates = validate_currency_rates(pd.DataFrame(rate_rows))
    rate_lookup = pd.DataFrame(rate_rows).pivot(
        index="available_time", columns="currency", values="policy_rate"
    )
    forward_rows: list[dict[str, object]] = []
    for pair in pairs:
        frame = data[pair.symbol]
        for timestamp in release_times:
            timestamp = pd.Timestamp(timestamp)
            prior = frame.loc[frame.index <= timestamp, "close"]
            if prior.empty:
                continue
            available_time = timestamp + timedelta(microseconds=1)
            base_rate = float(rate_lookup.loc[available_time, pair.base]) / 100
            quote_rate = float(rate_lookup.loc[available_time, pair.quote]) / 100
            spot = float(prior.iloc[-1])
            one_month_points = -(base_rate - quote_rate) * spot * (30 / 365)
            forward_rows.append(
                {
                    "observation_time": timestamp,
                    "available_time": available_time,
                    "symbol": pair.symbol,
                    "forward_points_1m": one_month_points,
                    "forward_points_3m": one_month_points * 3,
                    "spot_reference": spot,
                }
            )
    forwards = validate_forward_points(pd.DataFrame(forward_rows))
    return PointInTimeData(rates, forwards, config.maximum_staleness_days)


def load_point_in_time_data(
    config: PointInTimeConfig,
    data: Mapping[str, pd.DataFrame] | None = None,
) -> PointInTimeData | None:
    if not config.enabled:
        return None
    if config.provider == "synthetic":
        if not data:
            raise ValueError("synthetic point-in-time data requires market data")
        return _synthetic_point_in_time_data(data, config)
    root = Path(config.directory)
    rates_path = root / config.currency_rates_file
    forwards_path = root / config.forward_points_file
    if not rates_path.exists():
        raise FileNotFoundError(f"No point-in-time currency rates found at {rates_path}")
    if not forwards_path.exists():
        raise FileNotFoundError(f"No point-in-time forward points found at {forwards_path}")
    rates = validate_currency_rates(pd.read_csv(rates_path))
    forwards = validate_forward_points(pd.read_csv(forwards_path))
    return PointInTimeData(rates, forwards, config.maximum_staleness_days)


def _asof_entity_values(
    timestamps: pd.DatetimeIndex,
    frame: pd.DataFrame,
    entity_column: str,
    entity: str,
    value_columns: tuple[str, ...],
    maximum_staleness_days: int,
) -> pd.DataFrame:
    left = pd.DataFrame({"_feature_time": timestamps}).sort_values("_feature_time")
    available = frame.loc[frame[entity_column] == entity, ["available_time", *value_columns]].copy()
    if available.empty:
        return pd.DataFrame(index=timestamps, columns=value_columns, dtype=float)
    available = available.sort_values("available_time")
    merged = pd.merge_asof(
        left,
        available,
        left_on="_feature_time",
        right_on="available_time",
        direction="backward",
        tolerance=timedelta(days=maximum_staleness_days),
        allow_exact_matches=True,
    ).set_index("_feature_time")
    return merged[list(value_columns)].reindex(timestamps)


def build_carry_factors(
    symbol: str,
    frame: pd.DataFrame,
    point_in_time: PointInTimeData,
    periods_per_year: int,
) -> pd.DataFrame:
    """Build only-as-known-then carry factors aligned to close-of-bar timestamps."""
    pair = CurrencyPair.parse(symbol)
    staleness = point_in_time.maximum_staleness_days
    base = _asof_entity_values(
        frame.index,
        point_in_time.currency_rates,
        "currency",
        pair.base,
        RATE_COLUMNS,
        staleness,
    )
    quote = _asof_entity_values(
        frame.index,
        point_in_time.currency_rates,
        "currency",
        pair.quote,
        RATE_COLUMNS,
        staleness,
    )
    forwards = _asof_entity_values(
        frame.index,
        point_in_time.forward_points,
        "symbol",
        symbol,
        FORWARD_COLUMNS,
        staleness,
    )
    output = pd.DataFrame(index=frame.index)
    output["rate_differential"] = (base["policy_rate"] - quote["policy_rate"]) / 100
    base_slope = base["ois_3m"] - base["ois_1m"]
    quote_slope = quote["ois_3m"] - quote["ois_1m"]
    output["curve_slope_differential"] = (base_slope - quote_slope) / 100
    close = frame["close"].astype(float)
    output["forward_discount_1m"] = (
        -forwards["forward_points_1m"] / forwards["spot_reference"] * (365 / 30)
    )
    annualized_volatility = (
        np.log(close).diff().rolling(20, min_periods=20).std(ddof=0) * np.sqrt(periods_per_year)
    )
    output["carry_to_vol_20"] = output["rate_differential"] / annualized_volatility.replace(
        0, np.nan
    )
    return output
