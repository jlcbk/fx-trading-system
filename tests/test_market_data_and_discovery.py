from __future__ import annotations

import httpx
import pandas as pd
import pytest

from fx_system.config import CostConfig, DataConfig, RiskConfig
from fx_system.data import (
    OandaCandleProvider,
    SyntheticFXProvider,
    attach_historical_swaps,
    has_bid_ask,
    validate_bars,
)
from fx_system.engine import BacktestEngine
from fx_system.factor_config import (
    FactorDiscoverySettings,
    FactorMiningConfig,
    FactorSettings,
    PointInTimeConfig,
)
from fx_system.factor_dsl import generate_discovery_factors
from fx_system.factor_forward import (
    build_forward_predictions,
    fit_frozen_factor_model,
    validate_frozen_model,
)
from fx_system.factor_research import run_factor_mining, write_factor_artifacts
from fx_system.factors import FACTOR_DEFINITIONS, build_factor_panel
from fx_system.labels import _label_symbol
from fx_system.models import Side, Signal
from fx_system.point_in_time import (
    PointInTimeData,
    build_carry_factors,
    load_point_in_time_data,
    validate_currency_rates,
    validate_forward_points,
)


def _quote_frame(mid: list[float], spread: float = 0.0002) -> pd.DataFrame:
    index = pd.date_range("2025-01-06", periods=len(mid), freq="1D", tz="UTC")
    close = pd.Series(mid, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame(index=index)
    frame["open"] = open_
    frame["high"] = pd.concat([open_, close], axis=1).max(axis=1) + 0.0005
    frame["low"] = pd.concat([open_, close], axis=1).min(axis=1) - 0.0005
    frame["close"] = close
    for field in ("open", "high", "low", "close"):
        frame[f"bid_{field}"] = frame[field] - spread / 2
        frame[f"ask_{field}"] = frame[field] + spread / 2
    frame["volume"] = 1000.0
    return validate_bars(frame.drop(columns=["open", "high", "low", "close"]), "EURUSD")


def test_bid_ask_validation_derives_mid_and_rejects_crossed_quotes() -> None:
    frame = _quote_frame([1.0, 1.001, 1.002])
    assert has_bid_ask(frame)
    assert frame.iloc[0]["open"] == pytest.approx(1.0)
    assert frame.iloc[0]["spread_open"] == pytest.approx(0.0002)
    crossed = frame.copy()
    crossed.loc[crossed.index[1], "ask_close"] = crossed.loc[crossed.index[1], "bid_close"] - 1e-4
    with pytest.raises(ValueError, match="invalid OHLC"):
        validate_bars(crossed, "EURUSD")


def test_oanda_provider_requests_practice_bid_ask_candles() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api-fxpractice.oanda.com"
        assert request.url.params["price"] == "BA"
        assert request.headers["authorization"] == "Bearer token"
        candles = []
        for day, complete in (("2025-01-06", True), ("2025-01-07", True), ("2025-01-08", False)):
            candles.append(
                {
                    "time": f"{day}T00:00:00Z",
                    "complete": complete,
                    "volume": 10,
                    "bid": {"o": "1.0", "h": "1.1", "l": "0.9", "c": "1.0"},
                    "ask": {"o": "1.0002", "h": "1.1002", "l": "0.9002", "c": "1.0002"},
                }
            )
        return httpx.Response(200, json={"candles": candles})

    data = OandaCandleProvider.download(
        ["EURUSD"],
        "2025-01-06",
        "2025-01-09",
        "1d",
        "token",
        transport=httpx.MockTransport(handler),
    )
    assert len(data["EURUSD"]) == 2
    assert has_bid_ask(data["EURUSD"])
    with pytest.raises(ValueError, match="fxPractice"):
        OandaCandleProvider.download(
            ["EURUSD"],
            "2025-01-06",
            "2025-01-09",
            "1d",
            "token",
            base_url="https://api-fxtrade.oanda.com",
        )


def test_engine_uses_executable_quotes_and_historical_swap(permissive_risk: RiskConfig) -> None:
    frame = _quote_frame([1.0, 1.0, 1.001, 1.001])
    frame["swap_long_pips"] = -1.0
    frame["swap_short_pips"] = -1.0
    signal = Signal(
        frame.index[0], "EURUSD", Side.LONG, 0.9, "quote-test", 0.01, 1.0, 0.8, 24
    )
    result = BacktestEngine(
        permissive_risk,
        CostConfig(default_spread_pips=0, slippage_pips=0, commission_per_million=0),
    ).run({"EURUSD": frame}, [signal])
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(frame.iloc[1]["ask_open"])
    assert trade.exit_price == pytest.approx(frame.iloc[2]["bid_open"])
    assert trade.costs > 0


def test_historical_swap_join_never_backfills_future_rate(tmp_path) -> None:
    frame = _quote_frame([1.0, 1.0, 1.0, 1.0])
    swaps = pd.DataFrame(
        {
            "available_time": ["2025-01-07T00:00:00Z", "2025-01-09T00:00:00Z"],
            "swap_long_pips": [-0.5, -1.5],
            "swap_short_pips": [0.2, 0.7],
        }
    )
    swaps.to_csv(tmp_path / "EURUSD.csv", index=False)
    enriched = attach_historical_swaps({"EURUSD": frame}, tmp_path)["EURUSD"]
    assert pd.isna(enriched.loc[pd.Timestamp("2025-01-06", tz="UTC"), "swap_long_pips"])
    assert enriched.loc[pd.Timestamp("2025-01-08", tz="UTC"), "swap_long_pips"] == -0.5
    assert enriched.loc[pd.Timestamp("2025-01-09", tz="UTC"), "swap_long_pips"] == -1.5


def test_triple_barrier_uses_executable_side_of_quote() -> None:
    frame = _quote_frame([1.0, 1.0, 1.0, 1.0], spread=0.002)
    frame.loc[frame.index[1], "high"] = 1.008
    frame.loc[frame.index[1], "bid_high"] = 1.0069
    frame.loc[frame.index[1], "ask_high"] = 1.0089
    frame.loc[frame.index[1], "low"] = 0.999
    frame.loc[frame.index[1], "bid_low"] = 0.998
    frame.loc[frame.index[1], "ask_low"] = 1.0
    labels = _label_symbol(
        frame,
        pd.Series(0.01, index=frame.index),
        1,
        FactorSettings(target_atr=0.7, stop_atr=1.1, max_holding_hours=24),
    )
    assert labels.iloc[0]["_event"] != "target"


def _pit_fixture() -> PointInTimeData:
    rates = validate_currency_rates(
        pd.DataFrame(
            [
                ("2025-01-01", "2025-01-02", "EUR", 2.0, 2.1, 2.2),
                ("2025-01-01", "2025-01-02", "USD", 4.0, 4.1, 4.2),
                ("2025-01-01", "2025-01-10", "EUR", 3.0, 3.1, 3.2),
            ],
            columns=[
                "observation_time",
                "available_time",
                "currency",
                "policy_rate",
                "ois_1m",
                "ois_3m",
            ],
        )
    )
    forwards = validate_forward_points(
        pd.DataFrame(
            [
                ("2025-01-01", "2025-01-02", "EURUSD", 0.001, 0.003, 1.10),
                ("2025-01-01", "2025-01-10", "EURUSD", 0.002, 0.006, 1.11),
            ],
            columns=[
                "observation_time",
                "available_time",
                "symbol",
                "forward_points_1m",
                "forward_points_3m",
                "spot_reference",
            ],
        )
    )
    return PointInTimeData(rates, forwards, 30)


def test_carry_factors_use_available_time_not_future_revision() -> None:
    index = pd.DatetimeIndex(["2025-01-05", "2025-01-12"], tz="UTC")
    frame = pd.DataFrame({"close": [1.10, 1.11]}, index=index)
    factors = build_carry_factors("EURUSD", frame, _pit_fixture(), 260)
    assert factors.loc[index[0], "rate_differential"] == pytest.approx(-0.02)
    assert factors.loc[index[1], "rate_differential"] == pytest.approx(-0.01)
    assert factors.loc[index[0], "forward_discount_1m"] == pytest.approx(
        -0.001 / 1.10 * (365 / 30)
    )
    invalid = pd.DataFrame(
        [("2025-01-03", "2025-01-02", "EUR", 1, 1, 1)],
        columns=[
            "observation_time",
            "available_time",
            "currency",
            "policy_rate",
            "ois_1m",
            "ois_3m",
        ],
    )
    with pytest.raises(ValueError, match="observation_time"):
        validate_currency_rates(invalid)


def test_budgeted_factor_dsl_is_deterministic_and_has_no_future_dependency() -> None:
    data = SyntheticFXProvider(seed=301).generate(
        ["EURUSD", "GBPUSD", "USDJPY", "EURGBP"], 180, "4h"
    )
    settings = FactorDiscoverySettings(
        enabled=True,
        max_generated_factors=12,
        primitive_factors=["momentum_1", "momentum_12", "atr_percent"],
    )
    cutoff = data["EURUSD"].index[130]
    prefix_data = {symbol: frame.loc[:cutoff] for symbol, frame in data.items()}
    full, generated = generate_discovery_factors(
        build_factor_panel(data), dict(FACTOR_DEFINITIONS), settings
    )
    prefix, prefix_generated = generate_discovery_factors(
        build_factor_panel(prefix_data), dict(FACTOR_DEFINITIONS), settings
    )
    assert len(generated) == settings.max_generated_factors
    assert [item.name for item in generated] == [item.name for item in prefix_generated]
    assert {"cs_rank", "multiply", "delta"}.issubset({item.operator for item in generated})
    columns = [item.name for item in generated]
    full_prefix = full.loc[
        full["_feature_time"] <= cutoff, ["_feature_time", "_symbol", *columns]
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        full_prefix,
        prefix[["_feature_time", "_symbol", *columns]].reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_full_quote_carry_discovery_pipeline_records_stress_and_lineage(tmp_path) -> None:
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "EURGBP", "EURJPY"]
    data = SyntheticFXProvider(seed=302).generate(symbols, 500, "1d", price_mode="bid_ask")
    point_config = PointInTimeConfig(enabled=True, provider="synthetic", synthetic_seed=302)
    point_in_time = load_point_in_time_data(point_config, data)
    config = FactorMiningConfig(
        data=DataConfig(
            provider="synthetic",
            symbols=symbols,
            interval="1d",
            synthetic_bars=500,
            price_mode="bid_ask",
        ),
        costs=CostConfig(),
        risk=RiskConfig(close_before_weekend=False),
        factor=FactorSettings(
            train_bars=250,
            test_bars=75,
            step_bars=75,
            minimum_train_samples=1500,
            minimum_calibration_samples=300,
            bootstrap_samples=100,
            max_features=12,
            cost_stress_multipliers=[1.0, 1.5],
            promotion_required_stress_multiplier=1.5,
        ),
        point_in_time=point_config,
        discovery=FactorDiscoverySettings(
            enabled=True,
            max_generated_factors=8,
            primitive_factors=["momentum_1", "atr_percent", "rate_differential"],
        ),
    )
    mining = run_factor_mining(data, config, point_in_time)
    generated = mining.catalog.loc[mining.catalog["family"] == "discovered_expression"]
    assert len(generated) == 8
    assert mining.panel["rate_differential"].notna().mean() > 0.8
    assert mining.summary["point_in_time_fingerprint_sha256"] == point_in_time.fingerprint()
    assert set(mining.summary["cost_stress"]) == {"1x", "1.5x"}
    assert set(mining.folds[0].stress_metrics) == {"1x", "1.5x"}
    assert mining.summary["data_readiness"]["tier"] == "software_validation"
    assert not mining.summary["data_readiness"]["broker_ready"]
    assert mining.summary["verdict"] in {
        "rejected_for_trading",
        "research_candidate_requires_new_holdout",
    }
    output = write_factor_artifacts(mining, data, config, tmp_path)
    assert (output / "factor_catalog.csv").exists()
    assert (output / "cost_stress_by_fold.csv").exists()
    assert (output / "factor_manifest.json").exists()
    assert (output / "frozen_model_status.json").exists()
    assert not (output / "frozen_factor_model.json").exists()

    frozen = fit_frozen_factor_model(mining, config, allow_rejected_for_testing=True)
    validate_frozen_model(frozen, config)
    predictions, _ = build_forward_predictions(data, config, frozen, point_in_time)
    assert not predictions.empty
    assert predictions["_feature_time"].min() > pd.Timestamp(frozen["freeze_available_time"])
    assert predictions["probability"].between(0, 1).all()
    tampered = dict(frozen)
    tampered["classifier_intercept"] += 1
    with pytest.raises(ValueError, match="contract hash"):
        validate_frozen_model(tampered, config)


def test_forward_config_changes_only_the_allowed_data_horizon() -> None:
    development = FactorMiningConfig.from_yaml("configs/factors_broker_carry_dev.yaml")
    forward = FactorMiningConfig.from_yaml("configs/factors_broker_carry_forward.yaml")
    assert development.data.end == "2025-09-15"
    assert forward.data.end is None
    development_data = development.data.model_dump(mode="json", exclude={"end"})
    forward_data = forward.data.model_dump(mode="json", exclude={"end"})
    assert development_data == forward_data
    assert development.factor == forward.factor
    assert development.discovery == forward.discovery
    assert development.point_in_time == forward.point_in_time
    assert development.costs == forward.costs
    assert development.risk == forward.risk
