from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from .models import CurrencyPair, utc_timestamp

REQUIRED_COLUMNS = ("open", "high", "low", "close")
QUOTE_COLUMNS = tuple(
    f"{side}_{field}" for side in ("bid", "ask") for field in REQUIRED_COLUMNS
)


def has_bid_ask(frame: pd.DataFrame) -> bool:
    return set(QUOTE_COLUMNS).issubset(frame.columns)


def _invalid_ohlc(frame: pd.DataFrame, prefix: str = "") -> pd.Series:
    columns = [f"{prefix}{column}" for column in REQUIRED_COLUMNS]
    open_, high, low, close = (frame[column] for column in columns)
    return (
        (low > pd.concat([open_, close], axis=1).min(axis=1))
        | (high < pd.concat([open_, close], axis=1).max(axis=1))
        | (high < low)
        | (frame[columns] <= 0).any(axis=1)
    )


def validate_bars(
    frame: pd.DataFrame,
    symbol: str = "unknown",
    invalid_ohlc: str = "raise",
) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).lower().replace(" ", "_") for column in result.columns]
    present_quote_columns = set(QUOTE_COLUMNS) & set(result)
    if present_quote_columns and present_quote_columns != set(QUOTE_COLUMNS):
        missing_quotes = sorted(set(QUOTE_COLUMNS) - set(result))
        raise ValueError(f"{symbol}: incomplete bid/ask OHLC columns {missing_quotes}")
    quote_mode = bool(present_quote_columns)
    if quote_mode:
        for column in QUOTE_COLUMNS:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        for field in REQUIRED_COLUMNS:
            result[field] = (result[f"bid_{field}"] + result[f"ask_{field}"]) / 2
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
    invalid = _invalid_ohlc(result)
    if quote_mode:
        crossed = pd.Series(False, index=result.index)
        for field in REQUIRED_COLUMNS:
            crossed |= result[f"ask_{field}"] < result[f"bid_{field}"]
        invalid |= _invalid_ohlc(result, "bid_") | _invalid_ohlc(result, "ask_") | crossed
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
    result.attrs["price_mode"] = "bid_ask" if quote_mode else "mid"
    if quote_mode:
        result["spread_open"] = result["ask_open"] - result["bid_open"]
        result["spread_close"] = result["ask_close"] - result["bid_close"]
    for column in ("swap_long_pips", "swap_short_pips"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
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
                "price_mode": "bid_ask" if has_bid_ask(frame) else "mid",
                "mean_spread_price": (
                    float(frame["spread_close"].mean()) if has_bid_ask(frame) else None
                ),
                "historical_swap": {
                    "long": "swap_long_pips" in frame,
                    "short": "swap_short_pips" in frame,
                },
            }
            for symbol, frame in data.items()
        }
    }
    (root / "_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def attach_historical_swaps(
    data: Mapping[str, pd.DataFrame],
    directory: str | Path,
    maximum_staleness_days: int = 14,
) -> dict[str, pd.DataFrame]:
    """As-of join historical financing known by each bar; future rows are never backfilled."""
    root = Path(directory)
    output: dict[str, pd.DataFrame] = {}
    for symbol, frame in data.items():
        candidates = [root / f"{symbol}.csv", root / f"{symbol.lower()}.csv"]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"No historical swap file found for {symbol} in {root}")
        swaps = pd.read_csv(path)
        swaps.columns = [str(column).strip().lower() for column in swaps.columns]
        required = {"available_time", "swap_long_pips", "swap_short_pips"}
        missing = required - set(swaps)
        if missing:
            raise ValueError(f"{symbol}: historical swap file is missing {sorted(missing)}")
        swaps["available_time"] = pd.to_datetime(swaps["available_time"], utc=True, errors="coerce")
        if swaps["available_time"].isna().any():
            raise ValueError(f"{symbol}: historical swap file contains invalid available_time")
        for column in ("swap_long_pips", "swap_short_pips"):
            swaps[column] = pd.to_numeric(swaps[column], errors="coerce")
            if swaps[column].isna().any() or not np.isfinite(swaps[column]).all():
                raise ValueError(f"{symbol}: historical swap file contains invalid {column}")
        if swaps.duplicated("available_time").any():
            raise ValueError(f"{symbol}: duplicate historical swap available_time")
        left = pd.DataFrame({"timestamp": frame.index}).sort_values("timestamp")
        merged = pd.merge_asof(
            left,
            swaps.sort_values("available_time"),
            left_on="timestamp",
            right_on="available_time",
            direction="backward",
            tolerance=timedelta(days=maximum_staleness_days),
            allow_exact_matches=True,
        ).set_index("timestamp")
        enriched = frame.copy()
        enriched["swap_long_pips"] = merged["swap_long_pips"].reindex(enriched.index)
        enriched["swap_short_pips"] = merged["swap_short_pips"].reindex(enriched.index)
        output[symbol] = enriched
    return output


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


class OandaCandleProvider:
    """Historical practice-domain bid/ask candles; live OANDA domains are forbidden."""

    PRACTICE_URL = "https://api-fxpractice.oanda.com"

    @classmethod
    def download(
        cls,
        symbols: list[str],
        start: str,
        end: str | None,
        interval: str,
        token: str,
        *,
        base_url: str = PRACTICE_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> dict[str, pd.DataFrame]:
        if base_url.rstrip("/") != cls.PRACTICE_URL:
            raise ValueError("OANDA historical data is restricted to the fxPractice domain")
        if not token:
            raise ValueError("OANDA_PRACTICE_TOKEN is required for OANDA historical data")
        granularity = {"1h": "H1", "4h": "H4", "1d": "D"}[interval]
        start_time = pd.Timestamp(start)
        start_time = (
            start_time.tz_localize("UTC")
            if start_time.tzinfo is None
            else start_time.tz_convert("UTC")
        )
        end_time = pd.Timestamp(end) if end is not None else pd.Timestamp.now(tz="UTC")
        end_time = (
            end_time.tz_localize("UTC")
            if end_time.tzinfo is None
            else end_time.tz_convert("UTC")
        )
        headers = {"Authorization": f"Bearer {token}"}
        result: dict[str, pd.DataFrame] = {}
        with httpx.Client(
            base_url=base_url,
            headers=headers,
            transport=transport,
            timeout=30,
        ) as client:
            for raw_symbol in symbols:
                pair = CurrencyPair.parse(raw_symbol)
                instrument = f"{pair.base}_{pair.quote}"
                cursor = start_time
                first_page = True
                records: list[dict[str, object]] = []
                while cursor < end_time:
                    response = client.get(
                        f"/v3/instruments/{instrument}/candles",
                        params={
                            "price": "BA",
                            "granularity": granularity,
                            "count": 5000,
                            "from": cursor.isoformat(),
                            "includeFirst": str(first_page).lower(),
                        },
                    )
                    response.raise_for_status()
                    candles = response.json().get("candles", [])
                    if not candles:
                        break
                    latest: pd.Timestamp | None = None
                    for candle in candles:
                        timestamp = pd.Timestamp(candle["time"])
                        timestamp = (
                            timestamp.tz_localize("UTC")
                            if timestamp.tzinfo is None
                            else timestamp.tz_convert("UTC")
                        )
                        latest = timestamp
                        if not candle.get("complete", False) or timestamp >= end_time:
                            continue
                        bid = candle.get("bid")
                        ask = candle.get("ask")
                        if bid is None or ask is None:
                            raise RuntimeError("OANDA BA candle response omitted bid or ask prices")
                        row: dict[str, object] = {
                            "timestamp": timestamp,
                            "volume": float(candle.get("volume", 0)),
                        }
                        for field, key in zip(REQUIRED_COLUMNS, ("o", "h", "l", "c"), strict=True):
                            row[f"bid_{field}"] = float(bid[key])
                            row[f"ask_{field}"] = float(ask[key])
                        records.append(row)
                    if latest is None or latest <= cursor:
                        break
                    cursor = latest
                    first_page = False
                    if len(candles) < 5000:
                        break
                if not records:
                    raise RuntimeError(
                        f"OANDA returned no complete bid/ask candles for {pair.symbol}"
                    )
                frame = pd.DataFrame(records).set_index("timestamp")
                result[pair.symbol] = validate_bars(frame, pair.symbol)
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
        price_mode: str = "mid",
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
            if price_mode == "bid_ask":
                spread_pips = np.maximum(0.25, rng.lognormal(mean=-0.1, sigma=0.35, size=bars))
                half_spread = spread_pips * pair.pip_size / 2
                for field in REQUIRED_COLUMNS:
                    frame[f"bid_{field}"] = frame[field] - half_spread
                    frame[f"ask_{field}"] = frame[field] + half_spread
                frame["swap_long_pips"] = -0.15 + 0.05 * np.sin(np.arange(bars) / 50)
                frame["swap_short_pips"] = -0.10 - 0.05 * np.sin(np.arange(bars) / 50)
            elif price_mode != "mid":
                raise ValueError(f"Unsupported synthetic price_mode: {price_mode}")
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
            config.symbols,
            config.synthetic_bars,
            config.interval,
            config.start,
            config.price_mode,
        )
    elif provider == "oanda":
        data = OandaCandleProvider.download(
            config.symbols,
            config.start,
            config.end,
            config.interval,
            os.environ.get("OANDA_PRACTICE_TOKEN", ""),
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if config.swap_directory is not None:
        data = attach_historical_swaps(
            data,
            config.swap_directory,
            config.maximum_swap_staleness_days,
        )

    start = pd.Timestamp(config.start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = pd.Timestamp(config.end) if config.end is not None else None
    if end is not None:
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    sliced: dict[str, pd.DataFrame] = {}
    for symbol, frame in data.items():
        if config.price_mode == "bid_ask" and not has_bid_ask(frame):
            raise ValueError(
                f"{symbol}: data.price_mode=bid_ask requires complete bid/ask OHLC columns"
            )
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
