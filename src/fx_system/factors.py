from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .indicators import atr, ema, rolling_zscore, rsi
from .models import CurrencyPair


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    family: str
    directional: bool
    description: str


def _definitions() -> list[FactorDefinition]:
    result: list[FactorDefinition] = []

    def add(name: str, family: str, directional: bool, description: str) -> None:
        result.append(FactorDefinition(name, family, directional, description))

    for window in (1, 3, 6, 12, 24):
        add(f"momentum_{window}", "momentum", True, f"{window}-bar log return")
    for window in (10, 20, 60):
        add(f"close_z_{window}", "reversal", True, f"Close z-score over {window} bars")
    add("rsi_14_centered", "reversal", True, "RSI(14) centered and scaled around 50")
    for fast, slow in ((5, 20), (10, 30), (20, 60)):
        add(
            f"ema_spread_{fast}_{slow}",
            "trend",
            True,
            f"EMA({fast}) minus EMA({slow}), normalized by ATR",
        )
    for window in (6, 12, 24, 60):
        add(f"realized_vol_{window}", "volatility", False, f"{window}-bar return volatility")
    add("vol_ratio_6_24", "volatility", False, "Short/medium realized volatility ratio")
    add("vol_ratio_12_60", "volatility", False, "Medium/slow realized volatility ratio")
    add("atr_percent", "volatility", False, "ATR as a fraction of close")
    for window in (10, 20, 60):
        add(
            f"channel_position_{window}",
            "market_structure",
            True,
            f"Close location within the trailing {window}-bar range",
        )
    add("candle_body_atr", "market_structure", True, "Candle body normalized by ATR")
    add("wick_balance_atr", "market_structure", True, "Lower minus upper wick normalized by ATR")
    add("range_expansion", "market_structure", False, "Current true range versus trailing median")
    for window in (12, 24):
        add(
            f"efficiency_{window}",
            "regime",
            False,
            f"Directional efficiency ratio over {window} bars",
        )
        add(
            f"autocorrelation_{window}",
            "regime",
            False,
            f"Lag-one return autocorrelation over {window} bars",
        )
    add("return_skew_24", "distribution", True, "Rolling 24-bar return skew")
    add("return_kurtosis_24", "distribution", False, "Rolling 24-bar excess kurtosis")
    add("hour_sin", "calendar", False, "Cyclical UTC hour encoding")
    add("hour_cos", "calendar", False, "Cyclical UTC hour encoding")
    add("weekday_sin", "calendar", False, "Cyclical weekday encoding")
    add("weekday_cos", "calendar", False, "Cyclical weekday encoding")
    for window in (1, 3, 6, 12):
        add(
            f"base_strength_{window}",
            "currency_graph",
            True,
            f"Least-squares base-currency strength over {window} bars",
        )
        add(
            f"quote_strength_{window}",
            "currency_graph",
            True,
            f"Least-squares quote-currency strength over {window} bars",
        )
        add(
            f"currency_relative_{window}",
            "currency_graph",
            True,
            f"Base minus quote graph strength over {window} bars",
        )
        add(
            f"pair_residual_{window}",
            "relative_value",
            True,
            f"Pair return unexplained by the currency graph over {window} bars",
        )
    return result


FACTOR_DEFINITIONS = {item.name: item for item in _definitions()}


def factor_catalog() -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in FACTOR_DEFINITIONS.values()])


def factor_columns() -> list[str]:
    return list(FACTOR_DEFINITIONS)


def directional_factor_columns() -> set[str]:
    return {name for name, definition in FACTOR_DEFINITIONS.items() if definition.directional}


def _efficiency(close: pd.Series, window: int) -> pd.Series:
    displacement = close.diff(window).abs()
    path = close.diff().abs().rolling(window, min_periods=window).sum().replace(0, np.nan)
    return displacement / path


def _autocorrelation(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=window).corr(returns.shift(1))


def _currency_graph(
    data: Mapping[str, pd.DataFrame], windows: tuple[int, ...] = (1, 3, 6, 12)
) -> dict[str, pd.DataFrame]:
    symbols = sorted(data)
    pairs = [CurrencyPair.parse(symbol) for symbol in symbols]
    currencies = sorted({currency for pair in pairs for currency in (pair.base, pair.quote)})
    currency_index = {currency: i for i, currency in enumerate(currencies)}
    incidence = np.zeros((len(symbols), len(currencies)))
    for row, pair in enumerate(pairs):
        incidence[row, currency_index[pair.base]] = 1.0
        incidence[row, currency_index[pair.quote]] = -1.0
    pseudo_inverse = np.linalg.pinv(incidence)
    has_graph_cycles = np.linalg.matrix_rank(incidence) < len(symbols)
    closes = pd.concat({symbol: data[symbol]["close"] for symbol in symbols}, axis=1).sort_index()
    closes = closes.ffill(limit=1)
    log_closes = np.log(closes)
    output = {symbol: pd.DataFrame(index=data[symbol].index) for symbol in symbols}
    for window in windows:
        pair_returns = log_closes.diff(window)
        strengths = pair_returns.to_numpy() @ pseudo_inverse.T
        fitted = strengths @ incidence.T
        residuals = (
            pair_returns.to_numpy() - fitted
            if has_graph_cycles
            else np.full_like(pair_returns.to_numpy(), np.nan)
        )
        strength_frame = pd.DataFrame(strengths, index=pair_returns.index, columns=currencies)
        fitted_frame = pd.DataFrame(
            fitted if has_graph_cycles else np.full_like(fitted, np.nan),
            index=pair_returns.index,
            columns=symbols,
        )
        residual_frame = pd.DataFrame(residuals, index=pair_returns.index, columns=symbols)
        for symbol, pair in zip(symbols, pairs, strict=True):
            target_index = data[symbol].index
            output[symbol][f"base_strength_{window}"] = strength_frame[pair.base].reindex(
                target_index
            )
            output[symbol][f"quote_strength_{window}"] = strength_frame[pair.quote].reindex(
                target_index
            )
            output[symbol][f"currency_relative_{window}"] = fitted_frame[symbol].reindex(
                target_index
            )
            output[symbol][f"pair_residual_{window}"] = residual_frame[symbol].reindex(target_index)
    return output


def build_factor_panel(data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build close-of-bar factors; every rolling calculation ends at the feature timestamp."""
    graph_features = _currency_graph(data)
    panels: list[pd.DataFrame] = []
    for symbol, frame in sorted(data.items()):
        close = frame["close"].astype(float)
        log_close = np.log(close)
        returns = log_close.diff()
        current_atr = atr(frame, 14)
        factors = pd.DataFrame(index=frame.index)
        for window in (1, 3, 6, 12, 24):
            factors[f"momentum_{window}"] = log_close.diff(window)
        for window in (10, 20, 60):
            factors[f"close_z_{window}"] = rolling_zscore(close, window)
        factors["rsi_14_centered"] = (rsi(close, 14) - 50) / 50
        for fast, slow in ((5, 20), (10, 30), (20, 60)):
            factors[f"ema_spread_{fast}_{slow}"] = (
                ema(close, fast) - ema(close, slow)
            ) / current_atr.replace(0, np.nan)
        for window in (6, 12, 24, 60):
            factors[f"realized_vol_{window}"] = returns.rolling(window, min_periods=window).std(
                ddof=0
            )
        factors["vol_ratio_6_24"] = factors["realized_vol_6"] / factors["realized_vol_24"].replace(
            0, np.nan
        )
        factors["vol_ratio_12_60"] = factors["realized_vol_12"] / factors[
            "realized_vol_60"
        ].replace(0, np.nan)
        factors["atr_percent"] = current_atr / close
        for window in (10, 20, 60):
            trailing_high = frame["high"].rolling(window, min_periods=window).max()
            trailing_low = frame["low"].rolling(window, min_periods=window).min()
            width = (trailing_high - trailing_low).replace(0, np.nan)
            factors[f"channel_position_{window}"] = 2 * (close - trailing_low) / width - 1
        body = close - frame["open"]
        upper_wick = frame["high"] - pd.concat([frame["open"], close], axis=1).max(axis=1)
        lower_wick = pd.concat([frame["open"], close], axis=1).min(axis=1) - frame["low"]
        factors["candle_body_atr"] = body / current_atr.replace(0, np.nan)
        factors["wick_balance_atr"] = (lower_wick - upper_wick) / current_atr.replace(0, np.nan)
        true_range = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - close.shift(1)).abs(),
                (frame["low"] - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        factors["range_expansion"] = true_range / true_range.rolling(20).median().replace(0, np.nan)
        for window in (12, 24):
            factors[f"efficiency_{window}"] = _efficiency(close, window)
            factors[f"autocorrelation_{window}"] = _autocorrelation(returns, window)
        factors["return_skew_24"] = returns.rolling(24, min_periods=24).skew()
        factors["return_kurtosis_24"] = returns.rolling(24, min_periods=24).kurt()
        factors["hour_sin"] = np.sin(2 * np.pi * frame.index.hour / 24)
        factors["hour_cos"] = np.cos(2 * np.pi * frame.index.hour / 24)
        factors["weekday_sin"] = np.sin(2 * np.pi * frame.index.dayofweek / 5)
        factors["weekday_cos"] = np.cos(2 * np.pi * frame.index.dayofweek / 5)
        factors = factors.join(graph_features[symbol])
        factors["_close"] = close
        factors["_atr"] = current_atr
        factors["_symbol"] = symbol
        factors.index.name = "_feature_time"
        panels.append(factors.reset_index())
    panel = pd.concat(panels, ignore_index=True)
    expected = set(factor_columns())
    missing = expected - set(panel)
    if missing:
        raise RuntimeError(f"Factor implementation is missing catalog entries: {sorted(missing)}")
    panel["_feature_time"] = pd.to_datetime(panel["_feature_time"], utc=True)
    return panel.sort_values(["_feature_time", "_symbol"]).reset_index(drop=True)
