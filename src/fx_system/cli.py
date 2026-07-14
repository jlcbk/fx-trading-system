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

from .analytics import calculate_metrics
from .brokers import OandaPracticeBroker
from .config import SystemConfig
from .data import YahooFXProvider, load_from_config, save_csv_directory
from .engine import BacktestEngine
from .planner import create_paper_plan
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
