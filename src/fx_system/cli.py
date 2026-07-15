from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .analytics import calculate_metrics, equity_frame, trades_frame
from .brokers import OandaPracticeBroker
from .config import SystemConfig
from .data import YahooFXProvider, load_from_config, save_csv_directory
from .engine import BacktestEngine
from .factor_config import FactorMiningConfig
from .factor_forward import build_forward_predictions, load_frozen_model
from .factor_research import audit_factor_data, run_factor_mining, write_factor_artifacts
from .planner import create_paper_plan
from .point_in_time import load_point_in_time_data
from .reporting import write_backtest_artifacts
from .research import generate_ensemble_signals, screen_strategies, walk_forward

app = typer.Typer(no_args_is_help=True, help="Multi-currency FX research and paper-trading CLI")
console = Console()


def _load(path: Path) -> SystemConfig:
    return SystemConfig.from_yaml(path)


def _metrics_table(metrics: dict[str, object], title: str) -> Table:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for label, key, formatter in (
        ("Return", "total_return", lambda v: f"{v:.2%}"),
        ("Sharpe", "sharpe", lambda v: f"{v:.3f}"),
        ("Max drawdown", "max_drawdown", lambda v: f"{v:.2%}"),
        ("Trades", "trades", lambda v: str(v)),
        ("Win rate", "win_rate", lambda v: f"{v:.2%}"),
        ("Payoff ratio", "payoff_ratio", lambda v: f"{v:.3f}"),
        ("Profit factor", "profit_factor", lambda v: f"{v:.3f}"),
        ("Average hold", "average_holding_hours", lambda v: f"{v:.1f} h"),
        ("Maximum hold", "max_holding_hours", lambda v: f"{v:.1f} h"),
    ):
        table.add_row(label, formatter(metrics[key]))
    return table


@app.command("validate-config")
def validate_config(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
) -> None:
    parsed = _load(config)
    console.print(f"[green]Valid[/green]: {config} ({len(parsed.data.symbols)} symbols)")


@app.command()
def download(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    parsed = _load(config)
    console.print("Downloading public Yahoo midpoint bars; these are research-only data…")
    data = YahooFXProvider.download(
        parsed.data.symbols, parsed.data.start, parsed.data.end, parsed.data.interval
    )
    destination = output or parsed.data.directory
    save_csv_directory(data, destination)
    console.print(f"[green]Saved[/green] {len(data)} symbols to {destination}")


@app.command("factor-download")
def factor_download(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/factors_daily.yaml"),
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    parsed = FactorMiningConfig.from_yaml(config)
    if parsed.data.provider == "oanda":
        console.print("Downloading OANDA fxPractice bid/ask candles for factor research…")
        data = load_from_config(parsed.data)
    else:
        console.print("Downloading long-horizon public FX data for factor research…")
        data = YahooFXProvider.download(
            parsed.data.symbols, parsed.data.start, parsed.data.end, parsed.data.interval
        )
    destination = output or parsed.data.directory
    save_csv_directory(data, destination)
    console.print(f"[green]Saved[/green] {len(data)} symbols to {destination}")


@app.command("factor-mine")
def factor_mine(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/factors_daily.yaml"),
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    parsed = FactorMiningConfig.from_yaml(config)
    data = load_from_config(parsed.data)
    point_in_time = load_point_in_time_data(parsed.point_in_time, data)
    mining = run_factor_mining(data, parsed, point_in_time)
    destination = write_factor_artifacts(mining, data, parsed, output)
    table = Table(title="Purged walk-forward multi-factor results")
    for column in ("Fold", "Kind", "Test", "AUC", "Trades", "Return", "PF", "Max DD"):
        table.add_column(column)
    for fold in mining.folds:
        table.add_row(
            str(fold.fold),
            fold.kind,
            f"{fold.test_start.date()} → {fold.test_end.date()}",
            f"{fold.model_metrics['roc_auc']:.3f}",
            str(fold.trading_metrics["trades"]),
            f"{fold.trading_metrics['total_return']:.2%}",
            f"{fold.trading_metrics['profit_factor']:.3f}",
            f"{fold.trading_metrics['max_drawdown']:.2%}",
        )
    console.print(table)
    console.print(
        f"Compounded OOS return: {mining.summary['compounded_return']:.2%}; "
        f"positive folds: {mining.summary['positive_folds']}/{mining.summary['folds']}"
    )
    console.print(f"Artifacts: [cyan]{destination.resolve()}[/cyan]")


@app.command("factor-data-audit")
def factor_data_audit(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/factors_broker_carry_dev.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/factor_data_audit.json"
    ),
) -> None:
    parsed = FactorMiningConfig.from_yaml(config)
    data = load_from_config(parsed.data)
    point_in_time = load_point_in_time_data(parsed.point_in_time, data)
    audit = audit_factor_data(data, parsed, point_in_time)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    status = "green" if audit["broker_ready"] else "yellow"
    console.print(
        f"[{status}]broker_ready={audit['broker_ready']}[/{status}]; "
        f"tier={audit['tier']}; history={audit['minimum_history_years']:.2f}y; "
        f"swap={audit['minimum_swap_coverage']:.1%}; carry={audit['carry_coverage']:.1%}"
    )
    console.print(f"Audit: [cyan]{output.resolve()}[/cyan]")


@app.command("factor-forward-evaluate")
def factor_forward_evaluate(
    model: Annotated[Path, typer.Option("--model", "-m")],
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/factors_broker_carry_dev.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/factor_forward"
    ),
) -> None:
    """Evaluate strictly later data with a frozen factor model; never refit or reselect."""
    parsed = FactorMiningConfig.from_yaml(config)
    data = load_from_config(parsed.data)
    point_in_time = load_point_in_time_data(parsed.point_in_time, data)
    frozen = load_frozen_model(model, parsed)
    predictions, signals = build_forward_predictions(data, parsed, frozen, point_in_time)
    output.mkdir(parents=True, exist_ok=True)
    prediction_columns = [
        "_feature_time",
        "_entry_time",
        "_symbol",
        "_direction",
        "_atr",
        "probability",
        "estimated_swap_r",
        "estimated_cost_r",
        "expected_net_r",
    ]
    predictions[prediction_columns].to_csv(output / "forward_predictions.csv", index=False)
    pd.DataFrame([signal.to_dict() for signal in signals]).to_csv(
        output / "forward_signals.csv", index=False
    )
    freeze_time = pd.Timestamp(frozen["freeze_available_time"])
    latest_time = max(frame.index[-1] for frame in data.values())
    elapsed_days = max(0.0, (latest_time - freeze_time).total_seconds() / 86400)
    period_complete = elapsed_days >= int(frozen["minimum_forward_days"])
    if predictions.empty:
        metrics = {}
        status = "awaiting_strictly_later_data"
    else:
        forward_start = pd.Timestamp(predictions["_feature_time"].min())
        forward_data = {
            symbol: frame.loc[frame.index >= forward_start] for symbol, frame in data.items()
        }
        result = BacktestEngine(parsed.risk, parsed.costs).run(
            forward_data,
            signals,
            {
                "frozen_contract_sha256": frozen["contract_sha256"],
                "forward_only": True,
            },
            risk_history_data=data,
        )
        metrics = calculate_metrics(result)
        trades_frame(result).to_csv(output / "forward_trades.csv", index=False)
        equity_frame(result).to_csv(output / "forward_equity.csv", index=True)
        status = "duration_complete_requires_review" if period_complete else "collecting"
    manifest = {
        "status": status,
        "contract_sha256": frozen["contract_sha256"],
        "freeze_available_time": frozen["freeze_available_time"],
        "latest_data_time": latest_time.isoformat(),
        "elapsed_days": elapsed_days,
        "minimum_forward_days": frozen["minimum_forward_days"],
        "period_complete": period_complete,
        "signals": len(signals),
        "metrics": metrics,
        "note": "No fitting, selection, or automatic trade approval occurs in forward evaluation.",
    }
    (output / "forward_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    console.print(
        f"Forward status={status}; elapsed={elapsed_days:.1f}d; signals={len(signals)}; "
        f"artifacts=[cyan]{output.resolve()}[/cyan]"
    )


@app.command()
def backtest(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    parsed = _load(config)
    data = load_from_config(parsed.data)
    signals, raw_signals = generate_ensemble_signals(data, parsed)
    result = BacktestEngine(parsed.risk, parsed.costs).run(
        data, signals, {"raw_signal_count": len(raw_signals)}
    )
    destination = write_backtest_artifacts(result, signals, data, parsed, output)
    metrics = calculate_metrics(result)
    console.print(_metrics_table(metrics, "FX portfolio backtest"))
    console.print(f"Artifacts: [cyan]{destination.resolve()}[/cyan]")


@app.command()
def screen(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/screen"),
) -> None:
    parsed = _load(config)
    data = load_from_config(parsed.data)
    runs = screen_strategies(data, parsed)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    table = Table(title="Strategy screening (same data, costs and risk limits)")
    for column in ("Rank", "Strategy", "Score", "Trades", "Sharpe", "Max DD", "Win rate", "PF"):
        table.add_column(column, justify="right" if column != "Strategy" else "left")
    for rank, run in enumerate(runs, 1):
        metric = run.metrics
        rows.append({"rank": rank, "strategy": run.name, **metric})
        table.add_row(
            str(rank),
            run.name,
            f"{metric['screen_score']:.3f}",
            str(metric["trades"]),
            f"{metric['sharpe']:.2f}",
            f"{metric['max_drawdown']:.2%}",
            f"{metric['win_rate']:.2%}",
            f"{metric['profit_factor']:.2f}",
        )
        write_backtest_artifacts(run.result, [], data, parsed, output / run.name)
    pd.DataFrame(rows).to_csv(output / "ranking.csv", index=False)
    console.print(table)
    console.print(f"Artifacts: [cyan]{output.resolve()}[/cyan]")


@app.command("walk-forward")
def walk_forward_command(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    train_bars: Annotated[int, typer.Option(min=200)] = 1200,
    test_bars: Annotated[int, typer.Option(min=100)] = 400,
    top_k: Annotated[int, typer.Option(min=1, max=5)] = 2,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/walk_forward.json"),
) -> None:
    parsed = _load(config)
    data = load_from_config(parsed.data)
    folds = walk_forward(data, parsed, train_bars, test_bars, top_k)
    if not folds:
        raise typer.BadParameter("Not enough common bars for one walk-forward fold")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(folds, indent=2, allow_nan=False), encoding="utf-8")
    table = Table(title="Walk-forward out-of-sample folds")
    for column in ("Fold", "Selected", "Trades", "Return", "Sharpe", "Max DD"):
        table.add_column(column)
    for fold in folds:
        metrics = fold["test_metrics"]
        table.add_row(
            str(fold["fold"]),
            ", ".join(fold["selected"]),
            str(metrics["trades"]),
            f"{metrics['total_return']:.2%}",
            f"{metrics['sharpe']:.2f}",
            f"{metrics['max_drawdown']:.2%}",
        )
    console.print(table)
    console.print(f"Saved: [cyan]{output.resolve()}[/cyan]")


@app.command("paper-plan")
def paper_plan(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/default.yaml"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/paper_plan.json"),
    equity: Annotated[float | None, typer.Option(min=1)] = None,
    include_unapproved: Annotated[bool, typer.Option("--include-unapproved")] = False,
) -> None:
    parsed = _load(config)
    data = load_from_config(parsed.data)
    proposals = create_paper_plan(data, parsed, equity, include_unapproved)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(item) for item in proposals], indent=2), encoding="utf-8")
    console.print(
        f"Generated {len(proposals)} no-side-effect proposal(s): [cyan]{output.resolve()}[/cyan]"
    )
    if not proposals and not include_unapproved:
        console.print(
            "No strategy is marked paper_enabled; screening candidates were not promoted."
        )


@app.command("oanda-practice-submit")
def oanda_practice_submit(
    plan: Annotated[Path, typer.Option("--plan")] = Path("outputs/paper_plan.json"),
    confirm_practice: Annotated[bool, typer.Option("--confirm-practice")] = False,
) -> None:
    if not confirm_practice:
        raise typer.BadParameter("Pass --confirm-practice to submit orders to fxPractice")
    account_id = os.environ.get("OANDA_PRACTICE_ACCOUNT_ID", "")
    token = os.environ.get("OANDA_PRACTICE_TOKEN", "")
    proposals = json.loads(plan.read_text(encoding="utf-8"))
    with OandaPracticeBroker(account_id, token) as broker:
        for proposal in proposals:
            if datetime.now(UTC) > datetime.fromisoformat(proposal["expires_at"]):
                raise typer.BadParameter(
                    f"Proposal {proposal['proposal_id']} expired at {proposal['expires_at']}"
                )
            response = broker.submit_market_order(
                proposal["symbol"],
                int(proposal["units"]),
                float(proposal["stop_loss"]),
                float(proposal["take_profit"]),
                proposal["proposal_id"],
                confirm_practice=True,
            )
            transaction = response.get("orderCreateTransaction", {})
            console.print(f"Submitted {proposal['symbol']}: {transaction.get('id')}")


if __name__ == "__main__":
    app()
