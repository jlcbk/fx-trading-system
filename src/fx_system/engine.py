from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import timedelta
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import CostConfig, RiskConfig
from .models import BacktestResult, CurrencyPair, EquityPoint, Position, Side, Signal, Trade
from .rates import FXRateGraph
from .risk import PortfolioRiskManager


class ExecutionCostModel:
    def __init__(self, config: CostConfig) -> None:
        self.config = config

    def fill(self, mid: float, symbol: str, side: Side, is_entry: bool) -> float:
        pair = CurrencyPair.parse(symbol)
        half_spread = self.config.spread_for(symbol) * pair.pip_size / 2
        slippage = self.config.slippage_pips * pair.pip_size
        direction = int(side) if is_entry else -int(side)
        return mid + direction * (half_spread + slippage)

    def commission(self, units: float) -> float:
        return 2 * self.config.commission_per_million * units / 1_000_000

    def swap_pips(self, position: Position, exit_time: pd.Timestamp) -> float:
        rates = (
            self.config.daily_swap_pips_long
            if position.side == Side.LONG
            else self.config.daily_swap_pips_short
        )
        daily = rates.get(position.symbol, 0.0)
        if daily == 0:
            return 0.0
        total = 0.0
        day = position.entry_time.normalize() + pd.Timedelta(days=1)
        while day <= exit_time.normalize():
            if day.dayofweek < 5:
                total += daily * (3 if day.dayofweek == 2 else 1)
            day += pd.Timedelta(days=1)
        return total


class BacktestEngine:
    def __init__(self, risk: RiskConfig, costs: CostConfig, account_currency: str = "USD") -> None:
        self.risk_config = risk
        self.cost_model = ExecutionCostModel(costs)
        self.account_currency = account_currency

    @staticmethod
    def _returns(data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        closes = pd.concat({symbol: frame["close"] for symbol, frame in data.items()}, axis=1)
        return np.log(closes).diff()

    @staticmethod
    def _execution_schedule(
        signals: list[Signal], data: Mapping[str, pd.DataFrame]
    ) -> dict[pd.Timestamp, list[Signal]]:
        schedule: dict[pd.Timestamp, list[Signal]] = defaultdict(list)
        for signal in signals:
            index = data[signal.symbol].index
            location = index.searchsorted(signal.timestamp, side="right")
            if location < len(index):
                schedule[index[location]].append(signal)
        return schedule

    def run(
        self,
        data: Mapping[str, pd.DataFrame],
        signals: list[Signal],
        metadata: dict[str, object] | None = None,
    ) -> BacktestResult:
        if not data:
            raise ValueError("Backtest requires at least one symbol")
        rate_graph = FXRateGraph(data, self.account_currency)
        risk_manager = PortfolioRiskManager(self.risk_config, rate_graph, self._returns(data))
        schedule = self._execution_schedule(signals, data)
        timestamps = sorted(set().union(*(frame.index for frame in data.values())))
        positions: dict[str, Position] = {}
        trades: list[Trade] = []
        rejected: list[dict[str, object]] = []
        equity_curve: list[EquityPoint] = []
        cash = self.risk_config.initial_equity
        peak = cash
        day_start_equity = cash
        current_day = timestamps[0].date()
        halted = False
        is_daily = {
            symbol: len(frame) > 1 and (frame.index[1] - frame.index[0]).total_seconds() >= 86400
            for symbol, frame in data.items()
        }

        def bar_at(symbol: str, timestamp: pd.Timestamp) -> pd.Series | None:
            frame = data[symbol]
            if timestamp not in frame.index:
                return None
            return frame.loc[timestamp]

        def close_position(
            position: Position, timestamp: pd.Timestamp, mid: float, reason: str
        ) -> None:
            nonlocal cash
            exit_price = self.cost_model.fill(mid, position.symbol, position.side, is_entry=False)
            pair = CurrencyPair.parse(position.symbol)
            prices = rate_graph.prices_at(timestamp)
            prices[position.symbol] = mid
            quote_conversion = FXRateGraph.convert_with_prices(
                1.0, pair.quote, self.account_currency, prices
            )
            signed_units = int(position.side) * position.units
            gross = signed_units * (mid - position.entry_mid) * quote_conversion
            execution_pnl = signed_units * (exit_price - position.entry_price) * quote_conversion
            commission = self.cost_model.commission(position.units)
            swap_native = (
                self.cost_model.swap_pips(position, timestamp) * pair.pip_size * position.units
            )
            swap = swap_native * quote_conversion
            net = execution_pnl - commission + swap
            costs = gross - net
            trade = Trade(
                position_id=position.position_id,
                symbol=position.symbol,
                strategy=position.strategy,
                side=position.side,
                units=position.units,
                entry_time=position.entry_time,
                exit_time=timestamp,
                entry_mid=position.entry_mid,
                exit_mid=mid,
                entry_price=position.entry_price,
                exit_price=exit_price,
                gross_pnl=gross,
                costs=costs,
                net_pnl=net,
                initial_risk=position.initial_risk_account,
                exit_reason=reason,
                group_id=position.group_id,
            )
            trades.append(trade)
            cash += net
            positions.pop(position.symbol, None)

        def current_equity(timestamp: pd.Timestamp) -> tuple[float, float]:
            unrealized = 0.0
            prices = rate_graph.prices_at(timestamp)
            for position in positions.values():
                if position.symbol not in prices:
                    continue
                pair = CurrencyPair.parse(position.symbol)
                liquidation = self.cost_model.fill(
                    prices[position.symbol], position.symbol, position.side, is_entry=False
                )
                conversion = FXRateGraph.convert_with_prices(
                    1.0, pair.quote, self.account_currency, prices
                )
                unrealized += (
                    int(position.side)
                    * position.units
                    * (liquidation - position.entry_price)
                    * conversion
                )
            return cash + unrealized, unrealized

        def process_intrabar_exit(
            position: Position, timestamp: pd.Timestamp, bar: pd.Series
        ) -> bool:
            stop_hit = (
                float(bar["low"]) <= position.stop_price
                if position.side == Side.LONG
                else float(bar["high"]) >= position.stop_price
            )
            target_hit = (
                float(bar["high"]) >= position.target_price
                if position.side == Side.LONG
                else float(bar["low"]) <= position.target_price
            )
            if stop_hit:
                close_position(position, timestamp, position.stop_price, "STOP")
                return True
            if target_hit:
                close_position(position, timestamp, position.target_price, "TARGET")
                return True
            return False

        for timestamp in timestamps:
            equity_before, _ = current_equity(timestamp)
            if timestamp.date() != current_day:
                current_day = timestamp.date()
                day_start_equity = equity_before
            daily_return = equity_before / day_start_equity - 1 if day_start_equity else 0.0
            drawdown = max(0.0, 1 - equity_before / peak) if peak else 0.0

            # Time exits happen at bar open before any same-bar high/low can be observed.
            for position in list(positions.values()):
                bar = bar_at(position.symbol, timestamp)
                if bar is not None and timestamp >= position.max_exit_time:
                    close_position(position, timestamp, float(bar["open"]), "MAX_HOLD")

            # Existing stops/targets. If both were touched, stop wins; this is conservative.
            for position in list(positions.values()):
                bar = bar_at(position.symbol, timestamp)
                if bar is None:
                    continue
                process_intrabar_exit(position, timestamp, bar)

            candidates = sorted(
                schedule.get(timestamp, []), key=lambda item: item.confidence, reverse=True
            )
            by_group: dict[str, list[Signal]] = defaultdict(list)
            group_order: list[str] = []
            for sequence, signal in enumerate(candidates):
                key = signal.group_id or f"single-{sequence}-{signal.symbol}"
                if key not in by_group:
                    group_order.append(key)
                by_group[key].append(signal)

            for key in group_order:
                group = by_group[key]
                opened: list[str] = []
                group_failed = False
                for signal in group:
                    bar = bar_at(signal.symbol, timestamp)
                    if bar is None:
                        group_failed = True
                        continue
                    existing = positions.get(signal.symbol)
                    if existing is not None:
                        if existing.side != signal.side:
                            close_position(
                                existing, timestamp, float(bar["open"]), "OPPOSITE_SIGNAL"
                            )
                        else:
                            rejected.append(
                                {**signal.to_dict(), "reason": "same_direction_position_open"}
                            )
                            group_failed = True
                            continue
                    equity_now, _ = current_equity(timestamp)
                    peak_now = max(peak, equity_now)
                    drawdown_now = max(0.0, 1 - equity_now / peak_now) if peak_now else 0.0
                    decision = risk_manager.evaluate(
                        signal,
                        float(bar["open"]),
                        equity_now,
                        positions,
                        timestamp,
                        drawdown_now,
                        daily_return,
                    )
                    if not decision.approved or halted:
                        rejected.append(
                            {
                                **signal.to_dict(),
                                "reason": "system_halted" if halted else decision.reason,
                            }
                        )
                        group_failed = True
                        continue
                    stop_distance = signal.atr * signal.stop_atr
                    target_distance = min(
                        signal.atr * signal.target_atr,
                        stop_distance * self.risk_config.max_reward_risk,
                    )
                    entry_mid = float(bar["open"])
                    entry_fill = self.cost_model.fill(
                        entry_mid, signal.symbol, signal.side, is_entry=True
                    )
                    max_hours = min(signal.max_holding_hours, self.risk_config.max_holding_hours)
                    positions[signal.symbol] = Position(
                        position_id=uuid4().hex,
                        symbol=signal.symbol,
                        strategy=signal.strategy,
                        side=signal.side,
                        units=decision.units,
                        entry_time=timestamp,
                        entry_mid=entry_mid,
                        entry_price=entry_fill,
                        stop_price=entry_mid - int(signal.side) * stop_distance,
                        target_price=entry_mid + int(signal.side) * target_distance,
                        max_exit_time=timestamp + timedelta(hours=max_hours),
                        initial_risk_account=decision.initial_risk,
                        confidence=signal.confidence,
                        group_id=signal.group_id,
                    )
                    opened.append(signal.symbol)
                if group_failed and len(group) > 1:
                    for symbol in opened:
                        positions.pop(symbol, None)
                    rejected.append(
                        {
                            "timestamp": timestamp.isoformat(),
                            "group_id": key,
                            "reason": "atomic_group_rollback",
                        }
                    )
                else:
                    # An order filled at this bar's open is exposed to the same bar's range.
                    for symbol in opened:
                        position = positions.get(symbol)
                        bar = bar_at(symbol, timestamp)
                        if position is not None and bar is not None:
                            process_intrabar_exit(position, timestamp, bar)

            if self.risk_config.close_before_weekend and timestamp.dayofweek == 4:
                for position in list(positions.values()):
                    bar = bar_at(position.symbol, timestamp)
                    should_close = (
                        is_daily[position.symbol]
                        or timestamp.hour >= 20
                        or len(data[position.symbol].loc[timestamp:]) == 1
                    )
                    if bar is not None and should_close:
                        close_position(position, timestamp, float(bar["close"]), "WEEKEND")

            equity_now, unrealized = current_equity(timestamp)
            peak = max(peak, equity_now)
            drawdown = max(0.0, 1 - equity_now / peak) if peak else 0.0
            prices = rate_graph.prices_at(timestamp)
            gross = risk_manager.gross_notional(positions, prices) if positions else 0.0
            equity_curve.append(
                EquityPoint(
                    timestamp=timestamp,
                    cash=cash,
                    unrealized_pnl=unrealized,
                    equity=equity_now,
                    open_positions=len(positions),
                    gross_leverage=gross / equity_now if equity_now > 0 else float("inf"),
                    drawdown=drawdown,
                )
            )
            if drawdown >= self.risk_config.max_drawdown and positions:
                for position in list(positions.values()):
                    bar = bar_at(position.symbol, timestamp)
                    mid = float(bar["close"]) if bar is not None else prices[position.symbol]
                    close_position(position, timestamp, mid, "DRAWDOWN_KILL")
                halted = True

        last_timestamp = timestamps[-1]
        last_prices = rate_graph.prices_at(last_timestamp)
        for position in list(positions.values()):
            close_position(position, last_timestamp, last_prices[position.symbol], "END_OF_DATA")

        if equity_curve:
            final_point = equity_curve[-1]
            equity_curve[-1] = EquityPoint(
                timestamp=final_point.timestamp,
                cash=cash,
                unrealized_pnl=0.0,
                equity=cash,
                open_positions=0,
                gross_leverage=0.0,
                drawdown=max(0.0, 1 - cash / peak) if peak else 0.0,
            )

        return BacktestResult(
            trades=trades,
            equity=equity_curve,
            rejected_signals=rejected,
            metadata={
                **(metadata or {}),
                "initial_equity": self.risk_config.initial_equity,
                "final_cash": cash,
                "halted": halted,
                "signal_count": len(signals),
                "rejected_count": len(rejected),
            },
        )
