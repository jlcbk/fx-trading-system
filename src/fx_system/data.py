from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .models import CurrencyPair, utc_timestamp

REQUIRED_COLUMNS = ("open", "high", "low", "close")


def validate_bars(
    frame: pd.DataFrame,
    symbol: str = "unknown",
    invalid_ohlc: str = "raise",
) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).lower().replace(" ", "_") for column in result.columns]
    missing = set(REQUIRED_COLUMNS) - set(result.columns)
    if missing:
        raise ValueError(f"{symbol}: missing OHLC columns {sorted(missing)}")
    if not isinstance(result.index, pd.DatetimeIndex):
        timestamp_column = next(
            (name for name in ("timestamp", "datetime", "date") if name in result.columns), None
        )
        if timestamp_column is None:
            raise ValueError(f"{symbol}: expected DatetimeIndex or timestamp column")
        result.index = pd.to_datetime(result.pop(timestamp_column), utc=True)
    elif result.index.tz is None:
        result.index = result.index.tz_localize("UTC")
    else:
        result.index = result.index.tz_convert("UTC")
    result.index.name = "timestamp"
    result = result[~result.index.duplicated(keep="last")].sort_index()
    for column in REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if "volume" not in result:
        result["volume"] = 0.0
    result["volume"] = pd.to_numeric(result["volume"], errors="coerce").fillna(0.0)
    result = result.dropna(subset=list(REQUIRED_COLUMNS))
    invalid = (
        (result["low"] > result[["open", "close"]].min(axis=1))
        | (result["high"] < result[["open", "close"]].max(axis=1))
        | (result["high"] < result["low"])
        | (result[list(REQUIRED_COLUMNS)] <= 0).any(axis=1)
    )
    if invalid.any():
        invalid_count = int(invalid.sum())
        if invalid_ohlc == "drop":
            warnings.warn(
                f"{symbol}: dropping {invalid_count} invalid provider OHLC rows",
                RuntimeWarning,
                stacklevel=2,
            )
            result = result.loc[~invalid]
        else:
            raise ValueError(f"{symbol}: {invalid_count} invalid OHLC rows")
    else:
        invalid_count = 0
    if len(result) < 2:
        raise ValueError(f"{symbol}: not enough valid bars")
    result.attrs["dropped_invalid_ohlc"] = invalid_count
    return result


def load_csv(path: str | Path, symbol: str | None = None) -> pd.DataFrame:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    return validate_bars(frame, symbol or csv_path.stem)


def load_csv_directory(directory: str | Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    root = Path(directory)
    result: dict[str, pd.DataFrame] = {}
    for raw_symbol in symbols:
        symbol = CurrencyPair.parse(raw_symbol).symbol
        candidates = [root / f"{symbol}.csv", root / f"{symbol.lower()}.csv"]
        path = next((item for item in candidates if item.exists()), None)
        if path is None:
            raise FileNotFoundError(f"No CSV found for {symbol}; tried: {candidates}")
        result[symbol] = load_csv(path, symbol)
    return result


def save_csv_directory(data: Mapping[str, pd.DataFrame], directory: str | Path) -> None:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for symbol, frame in data.items():
        frame.to_csv(root / f"{symbol}.csv", index=True)
    manifest = {
        "symbols": {
            symbol: {
                "rows": len(frame),
                "start": frame.index[0].isoformat(),
                "end": frame.index[-1].isoformat(),
                "dropped_invalid_ohlc": int(frame.attrs.get("dropped_invalid_ohlc", 0)),
            }
            for symbol, frame in data.items()
        }
    }
    (root / "_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


class YahooFXProvider:
    """Public-data adapter for research; Yahoo data is not execution-quality bid/ask data."""

    @staticmethod
    def download(
        symbols: list[str],
        start: str,
        end: str | None,
        interval: str,
    ) -> dict[str, pd.DataFrame]:
        import yfinance as yf

        requested_interval = "1h" if interval == "4h" else interval
        result: dict[str, pd.DataFrame] = {}
        for raw_symbol in symbols:
            symbol = CurrencyPair.parse(raw_symbol).symbol
            ticker = yf.Ticker(f"{symbol}=X")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=DeprecationWarning, module=r"yfinance\..*"
                )
                frame = ticker.history(
                    start=start,
                    end=end,
                    interval=requested_interval,
                    auto_adjust=False,
                    actions=False,
                    repair=False,
                )
            if frame.empty:
                raise RuntimeError(f"Yahoo returned no data for {symbol}=X")
            frame = frame.rename(columns=str.lower)
            if interval == "1d":
                # Preserve the provider's trading-date label without DST-driven 23:00 shifts.
                frame.index = pd.DatetimeIndex(frame.index.date).tz_localize("UTC")
            elif frame.index.tz is None:
                frame.index = frame.index.tz_localize("UTC")
            else:
                frame.index = frame.index.tz_convert("UTC")
            if interval == "4h":
                frame["_samples"] = 1
                frame = (
                    frame.resample("4h", origin="start_day")
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                            "_samples": "sum",
                        }
                    )
                    .dropna(subset=["open", "high", "low", "close"])
                )
                frame = frame.loc[frame["_samples"] == 4].drop(columns="_samples")
            frame = drop_incomplete_bars(frame, interval)
            result[symbol] = validate_bars(frame, symbol, invalid_ohlc="drop")
        return result


def drop_incomplete_bars(
    frame: pd.DataFrame,
    interval: str,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Remove a live provider's still-forming final candle."""
    duration = {
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }[interval]
    current = now if now is not None else pd.Timestamp.now(tz="UTC")
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    return frame.loc[frame.index + duration <= current]


@dataclass(frozen=True)
class SyntheticFXProvider:
    """Creates correlated, cross-rate-consistent bars for deterministic system tests."""

    seed: int = 42

    def generate(
        self,
        symbols: list[str],
        bars: int = 3000,
        interval: str = "4h",
        start: str = "2020-01-01",
    ) -> dict[str, pd.DataFrame]:
        rng = np.random.default_rng(self.seed)
        pairs = [CurrencyPair.parse(value) for value in symbols]
        currencies = sorted({ccy for pair in pairs for ccy in (pair.base, pair.quote)})
        usd_values = {
            "USD": 1.0,
            "EUR": 1.10,
            "GBP": 1.28,
            "JPY": 0.0091,
            "CHF": 1.11,
            "CAD": 0.74,
            "AUD": 0.67,
            "NZD": 0.62,
        }
        annual_vol = {
            "USD": 0.045,
            "EUR": 0.060,
            "GBP": 0.070,
            "JPY": 0.065,
            "CHF": 0.055,
            "CAD": 0.060,
            "AUD": 0.075,
            "NZD": 0.080,
        }
        periods_per_year = {"1h": 24 * 260, "4h": 6 * 260, "1d": 260}[interval]
        frequency = {"1h": "1h", "4h": "4h", "1d": "1D"}[interval]
        index = pd.date_range(utc_timestamp(start), periods=bars, freq=frequency)
        index = index[index.dayofweek < 5][:bars]
        if len(index) < bars:
            # Build a longer calendar when weekend filtering removed rows.
            index = pd.date_range(utc_timestamp(start), periods=int(bars * 1.5), freq=frequency)
            index = index[index.dayofweek < 5][:bars]

        market = rng.normal(0, 1 / np.sqrt(periods_per_year), size=bars)
        regime = np.repeat(rng.choice([-1.0, 0.0, 1.0], size=(bars // 250) + 1), 250)[:bars]
        factors: dict[str, np.ndarray] = {}
        for i, currency in enumerate(currencies):
            vol = annual_vol.get(currency, 0.07)
            beta = 0.25 + (i % 4) * 0.12
            idiosyncratic = rng.normal(0, vol / np.sqrt(periods_per_year), size=bars)
            drift = regime * ((i - len(currencies) / 2) * 0.000002)
            returns = beta * market + idiosyncratic + drift
            factors[currency] = np.cumsum(returns)

        output: dict[str, pd.DataFrame] = {}
        for pair in pairs:
            initial = usd_values.get(pair.base, 1.0) / usd_values.get(pair.quote, 1.0)
            log_close = np.log(initial) + factors[pair.base] - factors[pair.quote]
            close = np.exp(log_close)
            open_ = np.r_[close[0], close[:-1]]
            bar_sigma = np.maximum(np.abs(close / open_ - 1), 0.00015)
            wick_up = rng.uniform(0.2, 1.0, bars) * bar_sigma
            wick_down = rng.uniform(0.2, 1.0, bars) * bar_sigma
            high = np.maximum(open_, close) * (1 + wick_up)
            low = np.minimum(open_, close) * (1 - wick_down)
            frame = pd.DataFrame(
                {
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": rng.integers(500, 5000, bars).astype(float),
                },
                index=index,
            )
            output[pair.symbol] = validate_bars(frame, pair.symbol)
        return output


def load_from_config(config: object) -> dict[str, pd.DataFrame]:
    provider = config.provider
    if provider == "csv":
        data = load_csv_directory(config.directory, config.symbols)
    elif provider == "yahoo":
        data = YahooFXProvider.download(config.symbols, config.start, config.end, config.interval)
    elif provider == "synthetic":
        data = SyntheticFXProvider(seed=config.seed).generate(
            config.symbols, config.synthetic_bars, config.interval, config.start
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    start = pd.Timestamp(config.start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = pd.Timestamp(config.end) if config.end is not None else None
    if end is not None:
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    sliced: dict[str, pd.DataFrame] = {}
    for symbol, frame in data.items():
        mask = frame.index >= start
        if end is not None:
            mask &= frame.index < end
        selected = frame.loc[mask].copy()
        if len(selected) < 2:
            raise ValueError(
                f"{symbol}: fewer than two bars remain in requested range "
                f"[{config.start}, {config.end or 'latest'})"
            )
        sliced[symbol] = selected
    return sliced
