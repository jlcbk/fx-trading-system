from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .analytics import calculate_metrics, equity_frame, trades_frame
from .engine import BacktestEngine
from .factor_config import FactorMiningConfig, FactorSettings
from .factors import (
    build_factor_panel,
    directional_factor_columns,
    factor_catalog,
    factor_columns,
)
from .labels import build_directional_dataset
from .models import BacktestResult, CurrencyPair, Side, Signal
from .rates import FXRateGraph
from .reporting import data_fingerprint


@dataclass
class FactorFold:
    fold: int
    kind: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    evaluation_end: pd.Timestamp
    selected_features: list[str]
    factor_statistics: pd.DataFrame
    oos_factor_statistics: pd.DataFrame
    coefficients: pd.DataFrame
    predictions: pd.DataFrame
    signals: list[Signal]
    result: BacktestResult
    model_metrics: dict[str, float | int]
    trading_metrics: dict[str, Any]


@dataclass
class FactorMiningResult:
    panel: pd.DataFrame
    dataset: pd.DataFrame
    folds: list[FactorFold]
    summary: dict[str, Any]


def _bucket_spread(values: pd.Series, labels: pd.Series) -> float:
    valid = pd.DataFrame({"value": values, "label": labels}).dropna()
    if len(valid) < 50 or valid["value"].nunique() < 5:
        return 0.0
    try:
        buckets = pd.qcut(valid["value"], 5, labels=False, duplicates="drop")
    except ValueError:
        return 0.0
    rates = valid.groupby(buckets, observed=True)["label"].mean()
    return float(rates.iloc[-1] - rates.iloc[0]) if len(rates) >= 2 else 0.0


def _block_bootstrap_rank_test(
    values: pd.Series,
    labels: pd.Series,
    timestamps: pd.Series,
    settings: FactorSettings,
    seed_offset: int,
) -> tuple[float, float, int]:
    valid = pd.DataFrame(
        {"value": values, "label": labels, "timestamp": timestamps}
    ).dropna()
    if len(valid) < 50 or valid["value"].nunique() < 2 or valid["label"].nunique() < 2:
        return 0.0, 1.0, 0
    ranked = valid["value"].rank(method="average", pct=True)
    ranked_scale = float(ranked.std(ddof=0))
    label_scale = float(valid["label"].std(ddof=0))
    if ranked_scale <= 1e-12 or label_scale <= 1e-12:
        return 0.0, 1.0, 0
    contributions = ((ranked - ranked.mean()) / ranked_scale) * (
        (valid["label"] - valid["label"].mean()) / label_scale
    )
    time_scores = contributions.groupby(valid["timestamp"]).mean().sort_index().to_numpy()
    if len(time_scores) < settings.bootstrap_block_bars * 2:
        return float(time_scores.mean()), 1.0, 0

    observed = float(time_scores.mean())
    centered = time_scores - observed
    block_length = min(settings.bootstrap_block_bars, len(centered))
    circular = np.concatenate([centered, centered[: block_length - 1]])
    block_means = np.convolve(circular, np.ones(block_length), mode="valid") / block_length
    block_count = int(np.ceil(len(centered) / block_length))
    generator = np.random.default_rng(settings.random_state + seed_offset)
    starts = generator.integers(0, len(centered), size=(settings.bootstrap_samples, block_count))
    null_means = block_means[starts].mean(axis=1)
    p_value = float(
        (1 + np.count_nonzero(np.abs(null_means) >= abs(observed)))
        / (settings.bootstrap_samples + 1)
    )
    return observed, p_value, block_count


def _benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def _independent_factor_observations(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    required = {"_feature_time", "_symbol", "_direction"}
    outcome_column = "_realized_r" if "_realized_r" in frame else "_label"
    if not required.issubset(frame.columns):
        return pd.DataFrame(
            {
                "value": frame[feature],
                "outcome": frame[outcome_column],
                "timestamp": frame["_feature_time"],
            }
        ).dropna()

    columns = ["_feature_time", "_symbol", "_direction", feature, outcome_column]
    work = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    index = ["_feature_time", "_symbol"]
    feature_sides = work.pivot(index=index, columns="_direction", values=feature)
    outcome_sides = work.pivot(index=index, columns="_direction", values=outcome_column)
    if not {-1, 1}.issubset(feature_sides.columns) or not {-1, 1}.issubset(
        outcome_sides.columns
    ):
        return pd.DataFrame(columns=["value", "outcome", "timestamp"])
    if feature in directional_factor_columns():
        values = (feature_sides[1] - feature_sides[-1]) / 2
        outcomes = (outcome_sides[1] - outcome_sides[-1]) / 2
    else:
        values = (feature_sides[1] + feature_sides[-1]) / 2
        outcomes = (outcome_sides[1] + outcome_sides[-1]) / 2
    result = pd.DataFrame({"value": values, "outcome": outcomes}).dropna()
    result["timestamp"] = result.index.get_level_values("_feature_time")
    return result.reset_index(drop=True)


def factor_statistics(frame: pd.DataFrame, settings: FactorSettings) -> pd.DataFrame:
    features = factor_columns()
    if {"_feature_time", "_symbol"}.issubset(frame.columns):
        independent_population = len(frame[["_feature_time", "_symbol"]].drop_duplicates())
    else:
        independent_population = len(frame)
    rows: list[dict[str, Any]] = []
    for feature_number, feature in enumerate(features):
        valid = _independent_factor_observations(frame, feature)
        coverage = float(len(valid) / independent_population) if independent_population else 0.0
        information_coefficient = (
            float(valid["value"].corr(valid["outcome"], method="spearman"))
            if len(valid) >= 50
            and valid["value"].nunique() > 1
            and valid["outcome"].nunique() > 1
            else 0.0
        )
        if not np.isfinite(information_coefficient):
            information_coefficient = 0.0
        standard_deviation = float(valid["value"].std(ddof=0)) if len(valid) else 0.0
        robust_scale = (
            float(valid["value"].quantile(0.95) - valid["value"].quantile(0.05))
            if len(valid)
            else 0.0
        )
        clustered_score, bootstrap_p_value, time_blocks = _block_bootstrap_rank_test(
            valid["value"],
            valid["outcome"],
            valid["timestamp"],
            settings,
            feature_number,
        )
        rows.append(
            {
                "factor": feature,
                "coverage": coverage,
                "ic": information_coefficient,
                "absolute_ic": abs(information_coefficient),
                "standard_deviation": standard_deviation,
                "robust_scale": robust_scale,
                "quintile_target_spread": _bucket_spread(valid["value"], valid["outcome"]),
                "clustered_rank_score": clustered_score,
                "bootstrap_p_value": bootstrap_p_value,
                "effective_time_blocks": time_blocks,
                "independent_rows": len(valid),
                "test_role": (
                    "directional_payoff_difference"
                    if feature in directional_factor_columns()
                    else "average_payoff_regime"
                ),
            }
        )
    statistics = pd.DataFrame(rows)
    statistics["fdr_q_value"] = _benjamini_hochberg(statistics["bootstrap_p_value"])
    statistics["fdr_significant"] = statistics["fdr_q_value"] <= settings.factor_fdr_level
    return statistics.sort_values(
        ["fdr_q_value", "absolute_ic"], ascending=[True, False]
    ).reset_index(drop=True)


def screen_factors(train: pd.DataFrame, settings: FactorSettings) -> tuple[list[str], pd.DataFrame]:
    features = factor_columns()
    clean = train[features].replace([np.inf, -np.inf], np.nan)
    statistics = factor_statistics(train, settings)
    significance_eligible = (
        statistics["fdr_significant"]
        if settings.require_fdr_significance
        else pd.Series(True, index=statistics.index)
    )
    eligible = statistics.loc[
        (statistics["coverage"] >= settings.minimum_feature_coverage)
        & (statistics["absolute_ic"] > 1e-6)
        & (statistics["standard_deviation"] > 1e-10)
        & (statistics["robust_scale"] > 1e-10)
        & significance_eligible,
        "factor",
    ].tolist()
    if not eligible:
        if settings.require_fdr_significance:
            statistics["selected"] = False
            return [], statistics
        raise RuntimeError("No factor passed the configured coverage and variance checks")
    medians = clean[eligible].median()
    correlation = clean[eligible].fillna(medians).corr()
    selected: list[str] = []
    for candidate in eligible:
        if all(
            abs(float(correlation.loc[candidate, existing])) < settings.maximum_feature_correlation
            for existing in selected
        ):
            selected.append(candidate)
        if len(selected) >= settings.max_features:
            break
    statistics["selected"] = statistics["factor"].isin(selected)
    return selected, statistics


def _build_model(settings: FactorSettings) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="saga",
                    C=settings.model_c,
                    l1_ratio=settings.model_l1_ratio,
                    max_iter=4000,
                    random_state=settings.random_state,
                ),
            ),
        ]
    )


def _fit_probability_calibrator(
    model: Pipeline,
    calibration: pd.DataFrame,
    selected: list[str],
    random_state: int,
) -> LogisticRegression:
    scores = model.decision_function(calibration[selected]).reshape(-1, 1)
    calibrator = LogisticRegression(C=1_000_000, solver="lbfgs", random_state=random_state)
    calibrator.fit(scores, calibration["_label"])
    return calibrator


def _calibrated_probability(
    model: Pipeline,
    calibrator: LogisticRegression,
    frame: pd.DataFrame,
    selected: list[str],
) -> np.ndarray:
    scores = model.decision_function(frame[selected]).reshape(-1, 1)
    return calibrator.predict_proba(scores)[:, 1]


def _add_economic_scores(
    predictions: pd.DataFrame,
    data: Mapping[str, pd.DataFrame],
    config: FactorMiningConfig,
    non_target_mean_r: float,
) -> pd.DataFrame:
    rate_graph = FXRateGraph(data)
    estimated_costs: list[float] = []
    for _, row in predictions.iterrows():
        symbol = str(row["_symbol"])
        pair = CurrencyPair.parse(symbol)
        timestamp = pd.Timestamp(row["_feature_time"])
        prices = rate_graph.prices_at(timestamp)
        try:
            quote_conversion = FXRateGraph.convert_with_prices(
                1.0, pair.quote, rate_graph.account_currency, prices
            )
        except ValueError:
            estimated_costs.append(float("nan"))
            continue
        stop_risk = float(row["_atr"]) * config.factor.stop_atr * quote_conversion
        execution_price_cost = (
            config.costs.spread_for(symbol) + 2 * config.costs.slippage_pips
        ) * pair.pip_size
        execution_cost = execution_price_cost * quote_conversion
        commission_cost = 2 * config.costs.commission_per_million / 1_000_000
        estimated_costs.append(
            (execution_cost + commission_cost) / stop_risk + config.factor.cost_buffer_r
            if stop_risk > 0
            else float("nan")
        )
    result = predictions.copy()
    result["estimated_cost_r"] = estimated_costs
    result["expected_net_r"] = (
        result["probability"] * config.factor.reward_risk
        + (1 - result["probability"]) * non_target_mean_r
        - result["estimated_cost_r"]
    )
    return result


def _signals_from_predictions(
    predictions: pd.DataFrame, settings: FactorSettings, fold: int
) -> list[Signal]:
    candidates: list[Signal] = []
    for (_, symbol), group in predictions.groupby(["_feature_time", "_symbol"], sort=True):
        ranked = group.sort_values("expected_net_r", ascending=False)
        best = ranked.iloc[0]
        runner_up = float(ranked.iloc[1]["probability"]) if len(ranked) > 1 else 0.0
        probability = float(best["probability"])
        expected_net_r = float(best["expected_net_r"])
        if (
            expected_net_r < settings.minimum_expected_net_r
            or probability - runner_up < settings.minimum_direction_gap
        ):
            continue
        current_atr = float(best["_atr"])
        if not np.isfinite(current_atr) or current_atr <= 0:
            continue
        candidates.append(
            Signal(
                timestamp=pd.Timestamp(best["_feature_time"]),
                symbol=str(symbol),
                side=Side(int(best["_direction"])),
                confidence=probability,
                strategy=f"factor_logit_fold_{fold}",
                atr=current_atr,
                stop_atr=settings.stop_atr,
                target_atr=settings.target_atr,
                max_holding_hours=settings.max_holding_hours,
                reason=(
                    f"estimated target-first probability={probability:.3f}; "
                    f"expected_net_r={expected_net_r:.3f}"
                ),
            )
        )
    candidates.sort(key=lambda signal: (signal.timestamp, -signal.confidence, signal.symbol))
    output: list[Signal] = []
    for _, timestamp_signals in _group_signals_by_time(candidates).items():
        output.extend(timestamp_signals[: settings.max_signals_per_timestamp])
    return output


def _group_signals_by_time(signals: list[Signal]) -> dict[pd.Timestamp, list[Signal]]:
    output: dict[pd.Timestamp, list[Signal]] = {}
    for signal in signals:
        output.setdefault(signal.timestamp, []).append(signal)
    for items in output.values():
        items.sort(key=lambda signal: signal.confidence, reverse=True)
    return output


def _slice_data(
    data: Mapping[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    return {symbol: frame.loc[start:end] for symbol, frame in data.items()}


def _fold_boundaries(
    timestamps: list[pd.Timestamp], settings: FactorSettings
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    folds = []
    offset = 0
    required = settings.train_bars + settings.embargo_bars + settings.test_bars
    while offset + required <= len(timestamps):
        train_start = timestamps[offset]
        train_end = timestamps[offset + settings.train_bars - 1]
        test_start = timestamps[offset + settings.train_bars + settings.embargo_bars]
        test_end_location = offset + required - 1
        test_end = timestamps[test_end_location]
        if test_end_location + 1 >= len(timestamps):
            break
        last_entry_time = timestamps[test_end_location + 1]
        evaluation_end = last_entry_time + timedelta(hours=settings.max_holding_hours)
        if timestamps[-1] < evaluation_end:
            break
        folds.append((train_start, train_end, test_start, test_end, evaluation_end))
        offset += settings.step_bars
    return folds


def _holdout_boundary(
    timestamps: list[pd.Timestamp], settings: FactorSettings
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp] | None:
    if settings.holdout_bars == 0:
        return None
    last_feature_location = len(timestamps) - 2
    while last_feature_location > 0:
        entry_time = timestamps[last_feature_location + 1]
        if entry_time + timedelta(hours=settings.max_holding_hours) <= timestamps[-1]:
            break
        last_feature_location -= 1
    test_start_location = last_feature_location - settings.holdout_bars + 1
    train_end_location = test_start_location - settings.embargo_bars - 1
    train_start_location = train_end_location - settings.train_bars + 1
    if train_start_location < 0 or test_start_location <= train_end_location:
        raise RuntimeError("Not enough history for the configured untouched holdout")
    evaluation_end = timestamps[last_feature_location + 1] + timedelta(
        hours=settings.max_holding_hours
    )
    return (
        timestamps[train_start_location],
        timestamps[train_end_location],
        timestamps[test_start_location],
        timestamps[last_feature_location],
        evaluation_end,
    )


def run_factor_mining(
    data: Mapping[str, pd.DataFrame], config: FactorMiningConfig
) -> FactorMiningResult:
    panel = build_factor_panel(data)
    dataset = build_directional_dataset(panel, data, config.factor)
    common_times = sorted(set.intersection(*(set(frame.index) for frame in data.values())))
    boundaries = _fold_boundaries(common_times, config.factor)
    holdout_boundary = _holdout_boundary(common_times, config.factor)
    if holdout_boundary is not None:
        boundaries = [boundary for boundary in boundaries if boundary[3] < holdout_boundary[2]]
    if not boundaries:
        raise RuntimeError("Not enough common bars for one factor walk-forward fold")

    scheduled_folds = [("development", boundary) for boundary in boundaries]
    if holdout_boundary is not None:
        scheduled_folds.append(("holdout", holdout_boundary))

    folds: list[FactorFold] = []
    for fold_number, (kind, boundary) in enumerate(scheduled_folds, 1):
        train_start, train_end, test_start, test_end, evaluation_end = boundary
        train = dataset.loc[
            dataset["_feature_time"].between(train_start, train_end)
            & (dataset["_label_end_time"] < test_start)
        ].copy()
        test = dataset.loc[dataset["_feature_time"].between(test_start, test_end)].copy()
        if len(train) < config.factor.minimum_train_samples:
            raise RuntimeError(
                f"Fold {fold_number} has {len(train)} training rows; "
                f"minimum is {config.factor.minimum_train_samples}"
            )
        if train["_label"].nunique() < 2 or test["_label"].nunique() < 2:
            raise RuntimeError(f"Fold {fold_number} does not contain both label classes")
        train_times = sorted(train["_feature_time"].unique())
        calibration_location = int(len(train_times) * (1 - config.factor.calibration_fraction))
        if calibration_location <= 0 or calibration_location >= len(train_times):
            raise RuntimeError(
                f"Fold {fold_number} cannot create a chronological calibration split"
            )
        calibration_start = pd.Timestamp(train_times[calibration_location])
        model_train = train.loc[
            (train["_feature_time"] < calibration_start)
            & (train["_label_end_time"] < calibration_start)
        ].copy()
        calibration = train.loc[train["_feature_time"] >= calibration_start].copy()
        if len(calibration) < config.factor.minimum_calibration_samples:
            raise RuntimeError(
                f"Fold {fold_number} has {len(calibration)} calibration rows; minimum is "
                f"{config.factor.minimum_calibration_samples}"
            )
        if model_train["_label"].nunique() < 2 or calibration["_label"].nunique() < 2:
            raise RuntimeError(f"Fold {fold_number} fit/calibration split lacks both classes")
        selected, statistics = screen_factors(model_train, config.factor)
        if selected:
            model = _build_model(config.factor)
            model.fit(model_train[selected], model_train["_label"])
            calibrator = _fit_probability_calibrator(
                model, calibration, selected, config.factor.random_state
            )
            calibration_probabilities = _calibrated_probability(
                model, calibrator, calibration, selected
            )
            probabilities = _calibrated_probability(model, calibrator, test, selected)
            classifier = model.named_steps["classifier"]
            coefficients = pd.DataFrame(
                {
                    "factor": selected,
                    "standardized_coefficient": classifier.coef_[0],
                    "fold": fold_number,
                }
            )
        else:
            unconditional_probability = float(calibration["_label"].mean())
            calibration_probabilities = np.full(len(calibration), unconditional_probability)
            probabilities = np.full(len(test), unconditional_probability)
            coefficients = pd.DataFrame(
                columns=["factor", "standardized_coefficient", "fold"]
            )
        predictions = test[
            [
                "_feature_time",
                "_entry_time",
                "_label_end_time",
                "_symbol",
                "_direction",
                "_label",
                "_event",
                "_atr",
                "_realized_r",
            ]
        ].copy()
        predictions["probability"] = probabilities
        predictions["fold"] = fold_number
        non_target_mean_r = float(calibration.loc[calibration["_label"] == 0, "_realized_r"].mean())
        predictions = _add_economic_scores(predictions, data, config, non_target_mean_r)
        oos_statistics = factor_statistics(test, config.factor).assign(
            fold=fold_number, kind=kind
        )
        auc = float(roc_auc_score(test["_label"], probabilities))
        model_metrics: dict[str, float | int] = {
            "rows": len(test),
            "model_train_rows": len(model_train),
            "calibration_rows": len(calibration),
            "target_rate": float(test["_label"].mean()),
            "roc_auc": auc,
            "brier": float(brier_score_loss(test["_label"], probabilities)),
            "log_loss": float(log_loss(test["_label"], probabilities)),
            "calibration_brier": float(
                brier_score_loss(calibration["_label"], calibration_probabilities)
            ),
            "calibration_auc": float(
                roc_auc_score(calibration["_label"], calibration_probabilities)
            ),
            "non_target_mean_r": non_target_mean_r,
            "mean_estimated_cost_r": float(predictions["estimated_cost_r"].mean()),
            "minimum_expected_net_r": config.factor.minimum_expected_net_r,
        }
        signals = _signals_from_predictions(predictions, config.factor, fold_number)
        test_data = _slice_data(data, test_start, evaluation_end)
        result = BacktestEngine(config.risk, config.costs).run(
            test_data,
            signals,
            {
                "factor_fold": fold_number,
                "minimum_expected_net_r": config.factor.minimum_expected_net_r,
                "selected_features": selected,
            },
            risk_history_data=_slice_data(data, train_start, evaluation_end),
        )
        trading_metrics = calculate_metrics(result)
        folds.append(
            FactorFold(
                fold_number,
                kind,
                train_start,
                train_end,
                test_start,
                test_end,
                evaluation_end,
                selected,
                statistics.assign(fold=fold_number),
                oos_statistics,
                coefficients,
                predictions,
                signals,
                result,
                model_metrics,
                trading_metrics,
            )
        )

    development_folds = [fold for fold in folds if fold.kind == "development"]
    holdout = next((fold for fold in folds if fold.kind == "holdout"), None)
    returns = [float(fold.trading_metrics["total_return"]) for fold in development_folds]
    trades = [int(fold.trading_metrics["trades"]) for fold in development_folds]
    summary = {
        "folds": len(development_folds),
        "positive_folds": sum(value > 0 for value in returns),
        "no_eligible_factor_folds": sum(not fold.selected_features for fold in development_folds),
        "compounded_return": float(np.prod([1 + value for value in returns]) - 1),
        "mean_fold_return": float(np.mean(returns)),
        "worst_fold_return": float(np.min(returns)),
        "total_trades": sum(trades),
        "mean_roc_auc": float(
            np.mean([fold.model_metrics["roc_auc"] for fold in development_folds])
        ),
        "mean_brier": float(np.mean([fold.model_metrics["brier"] for fold in development_folds])),
        "minimum_expected_net_r": config.factor.minimum_expected_net_r,
        "reward_risk": config.factor.reward_risk,
        "maximum_holding_hours": config.factor.max_holding_hours,
        "holdout": (
            {
                "test_start": holdout.test_start,
                "test_end": holdout.test_end,
                "roc_auc": holdout.model_metrics["roc_auc"],
                "trades": holdout.trading_metrics["trades"],
                "total_return": holdout.trading_metrics["total_return"],
                "profit_factor": holdout.trading_metrics["profit_factor"],
                "max_drawdown": holdout.trading_metrics["max_drawdown"],
            }
            if holdout is not None
            else None
        ),
    }
    return FactorMiningResult(panel, dataset, folds, summary)


def _json_default(value: object) -> str | int | float | None:
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    return str(value)


def write_factor_artifacts(
    mining: FactorMiningResult,
    data: Mapping[str, pd.DataFrame],
    config: FactorMiningConfig,
    output_directory: str | Path | None = None,
) -> Path:
    output = Path(output_directory or config.factor.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    factor_catalog().to_csv(output / "factor_catalog.csv", index=False)
    all_statistics = pd.concat([fold.factor_statistics for fold in mining.folds], ignore_index=True)
    all_coefficients = pd.concat([fold.coefficients for fold in mining.folds], ignore_index=True)
    all_oos_statistics = pd.concat(
        [fold.oos_factor_statistics for fold in mining.folds], ignore_index=True
    )
    all_predictions = pd.concat([fold.predictions for fold in mining.folds], ignore_index=True)
    all_statistics.to_csv(output / "factor_statistics_by_fold.csv", index=False)
    all_oos_statistics.to_csv(output / "oos_factor_statistics_by_fold.csv", index=False)
    all_coefficients.to_csv(output / "model_coefficients_by_fold.csv", index=False)
    all_predictions.to_csv(output / "oos_predictions.csv", index=False)
    development_folds = {fold.fold for fold in mining.folds if fold.kind == "development"}
    stability = _factor_stability(
        all_statistics.loc[all_statistics["fold"].isin(development_folds)],
        all_coefficients.loc[all_coefficients["fold"].isin(development_folds)],
        all_oos_statistics.loc[all_oos_statistics["fold"].isin(development_folds)],
    )
    stability.to_csv(output / "factor_stability.csv", index=False)

    fold_payload = []
    for fold in mining.folds:
        fold_dir = output / f"fold_{fold.fold}"
        fold_dir.mkdir(exist_ok=True)
        trades_frame(fold.result).to_csv(fold_dir / "trades.csv", index=False)
        equity_frame(fold.result).to_csv(fold_dir / "equity.csv", index=True)
        pd.DataFrame([signal.to_dict() for signal in fold.signals]).to_csv(
            fold_dir / "signals.csv", index=False
        )
        fold_payload.append(
            {
                "fold": fold.fold,
                "kind": fold.kind,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "evaluation_end": fold.evaluation_end,
                "selected_features": fold.selected_features,
                "model_metrics": fold.model_metrics,
                "trading_metrics": fold.trading_metrics,
            }
        )
    (output / "fold_results.json").write_text(
        json.dumps(fold_payload, indent=2, default=_json_default, allow_nan=False), encoding="utf-8"
    )
    manifest = {
        "data_fingerprint_sha256": data_fingerprint(data),
        "symbols": sorted(data),
        "ranges": {
            symbol: [frame.index[0].isoformat(), frame.index[-1].isoformat()]
            for symbol, frame in data.items()
        },
        "rows": {symbol: len(frame) for symbol, frame in data.items()},
        "config": config.model_dump(mode="json"),
        "summary": mining.summary,
    }
    try:
        manifest["git_revision"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["git_revision"] = "uncommitted"
    source_manifest_path = Path(config.data.directory) / "_data_manifest.json"
    if source_manifest_path.exists():
        manifest["source_data_quality"] = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
    (output / "factor_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default, allow_nan=False), encoding="utf-8"
    )
    (output / "FACTOR_REPORT.md").write_text(
        _markdown_factor_report(mining, stability, manifest), encoding="utf-8"
    )
    return output


def _factor_stability(
    statistics: pd.DataFrame,
    coefficients: pd.DataFrame,
    oos_statistics: pd.DataFrame,
) -> pd.DataFrame:
    selected = statistics.loc[statistics["selected"]].copy()
    stats = selected.groupby("factor").agg(
        selected_folds=("fold", "nunique"),
        mean_ic=("ic", "mean"),
        std_ic=("ic", "std"),
        positive_ic_fraction=("ic", lambda values: float((values > 0).mean())),
        mean_quintile_spread=("quintile_target_spread", "mean"),
    )
    coefficient_stats = coefficients.groupby("factor").agg(
        mean_coefficient=("standardized_coefficient", "mean"),
        coefficient_sign_stability=(
            "standardized_coefficient",
            lambda values: float(max((values > 0).mean(), (values < 0).mean())),
        ),
    )
    oos_stats = oos_statistics.groupby("factor").agg(
        mean_oos_ic=("ic", "mean"),
        std_oos_ic=("ic", "std"),
        oos_ic_sign_stability=(
            "ic",
            lambda values: float(max((values > 0).mean(), (values < 0).mean())),
        ),
        significant_oos_folds=("fdr_significant", "sum"),
        mean_oos_q_value=("fdr_q_value", "mean"),
    )
    result = stats.join(coefficient_stats, how="outer").join(oos_stats, how="outer").reset_index()
    result["selected_folds"] = result["selected_folds"].fillna(0).astype(int)
    result["significant_oos_folds"] = result["significant_oos_folds"].fillna(0).astype(int)
    return result.sort_values(
        ["significant_oos_folds", "selected_folds", "mean_oos_ic"],
        ascending=[False, False, False],
    )


def _markdown_factor_report(
    mining: FactorMiningResult, stability: pd.DataFrame, manifest: dict[str, Any]
) -> str:
    def factor_row(row: Any) -> str:
        coefficient = (
            f"{row.mean_coefficient:.4f}" if pd.notna(row.mean_coefficient) else "n/a"
        )
        sign_stability = (
            f"{row.coefficient_sign_stability:.2%}"
            if pd.notna(row.coefficient_sign_stability)
            else "n/a"
        )
        return (
            f"| {row.factor} | {int(row.selected_folds)} | "
            f"{int(row.significant_oos_folds)} | {row.mean_oos_ic:.4f} | "
            f"{row.mean_oos_q_value:.3f} | {coefficient} | {sign_stability} |"
        )

    fold_rows = "\n".join(
        (
            "| {fold} | {start} | {end} | {auc:.3f} | {trades} | {ret:.2%} | {pf:.3f} | {dd:.2%} |"
        ).format(
            fold=f"{fold.fold} ({fold.kind})",
            start=fold.test_start.date(),
            end=fold.test_end.date(),
            auc=fold.model_metrics["roc_auc"],
            trades=fold.trading_metrics["trades"],
            ret=fold.trading_metrics["total_return"],
            pf=fold.trading_metrics["profit_factor"],
            dd=fold.trading_metrics["max_drawdown"],
        )
        for fold in mining.folds
    )
    top = stability.head(12)
    factor_rows = "\n".join(factor_row(row) for row in top.itertuples())
    summary = mining.summary
    holdout = summary["holdout"]
    holdout_section = (
        "## Untouched holdout verdict\n\n"
        f"The holdout from {pd.Timestamp(holdout['test_start']).date()} to "
        f"{pd.Timestamp(holdout['test_end']).date()} produced {holdout['trades']} trades, "
        f"{holdout['total_return']:.2%} return, PF {holdout['profit_factor']:.3f}, and "
        f"ROC AUC {holdout['roc_auc']:.3f}. This does not pass promotion gates.\n"
        if holdout is not None
        else "## Untouched holdout verdict\n\nNo holdout was configured.\n"
    )
    return f"""# FX multi-factor mining report

> This is a purged walk-forward research report, not evidence of guaranteed profitability.

## Research contract

- Symbols: {", ".join(manifest["symbols"])}
- Target/stop ratio: {summary["reward_risk"]:.3f}
- Maximum holding: {summary["maximum_holding_hours"]} hours
- Minimum expected net R: {summary["minimum_expected_net_r"]:.3f}
- Data SHA-256: `{manifest["data_fingerprint_sha256"]}`

Signals use factors known at a bar close and enter no earlier than the next bar open. Training rows
whose label horizon reaches a test window are purged, and an embargo separates train and test.
When factors pass screening, probabilities are Platt-calibrated on a later, purged training slice;
otherwise the fold uses an unconditional null model and cannot trade. Entry requires positive
expected R after pair-specific spread, slippage, commission, and an extra cost buffer. Directional
factors are tested on paired long-minus-short realized R, while regime factors use the paired mean.
Significance uses moving-block bootstrap samples clustered by feature timestamp, followed by
Benjamini-Hochberg false-discovery-rate correction. The stability table excludes the holdout.

## Out-of-sample folds

| Fold | Test start | Test end | ROC AUC | Trades | Return | PF | Max DD |
|---:|---|---|---:|---:|---:|---:|---:|
{fold_rows}

Compounded development-fold return: {summary["compounded_return"]:.2%}; positive folds:
{summary["positive_folds"]}/{summary["folds"]}; total trades: {summary["total_trades"]}; folds with
no FDR-eligible factor: {summary["no_eligible_factor_folds"]}/{summary["folds"]}.

{holdout_section}

## Development-only factor stability

| Factor | Selected | OOS sig. | OOS IC | OOS q | Coeff. | Sign stable |
|---|---:|---:|---:|---:|---:|---:|
{factor_rows}

A factor is not approved merely because it appears in this table. Promotion requires stable signs,
adequate trade count, positive net results across several untouched windows, broker bid/ask cost
stress, and a separate paper-trading period.
"""
