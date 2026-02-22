"""
portfolio.py -- portfolio state tracking and equity computation

equity = initial_capital + realized_pnl + unrealized_pnl - total_costs
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional

from .models import SpreadPosition, CompletedTrade
from .margin import MarginConfig, MarginManager


class Portfolio:
    """Manages open positions, records closed trades, computes equity."""

    def __init__(self, initial_capital: float = 100_000,
                 margin_config: Optional[MarginConfig] = None):
        self.initial_capital = initial_capital
        self.open_positions: dict[str, SpreadPosition] = {}
        self.completed_trades: list[CompletedTrade] = []
        self.equity_history: list[dict] = []

        self._realized_pnl = 0.0
        self._total_costs = 0.0

        self.margin = MarginManager(margin_config or MarginConfig())

    def equity_from_row(self, row: np.ndarray,
                        col_a: dict[str, int],
                        col_b: dict[str, int]) -> float:
        """Compute equity from a single price matrix row."""
        unrealized = 0.0
        for pid, pos in self.open_positions.items():
            ca = col_a.get(pid)
            cb = col_b.get(pid)
            if ca is None or cb is None:
                continue
            pa, pb = row[ca], row[cb]
            if np.isfinite(pa) and np.isfinite(pb):
                unrealized += pos.pnl(pa, pb)

        eq = self.initial_capital + self._realized_pnl + unrealized - self._total_costs
        if not np.isfinite(eq):
            return self.initial_capital + self._realized_pnl - self._total_costs
        return eq

    def open_spread(
        self,
        pair_id: str, sym_a: str, sym_b: str, sector: str,
        direction: int, beta: float,
        bar_idx: int, price_a: float, price_b: float, z_score: float,
        notional_a: float, notional_b: float, total_notional: float,
        cost_rate: float, half_life: float = 0.0,
    ) -> Optional[SpreadPosition]:
        """Open a spread position with pre-computed leg notionals."""
        if pair_id in self.open_positions:
            return None

        qty_a = direction * notional_a / price_a
        qty_b = -direction * np.sign(beta) * notional_b / price_b

        traded_notional = notional_a + notional_b
        entry_cost = traded_notional * cost_rate

        pos = SpreadPosition(
            pair_id=pair_id, sym_a=sym_a, sym_b=sym_b, sector=sector,
            direction=direction, beta=beta,
            entry_bar=bar_idx, entry_price_a=price_a, entry_price_b=price_b,
            entry_z=z_score, qty_a=qty_a, qty_b=qty_b,
            notional=total_notional, entry_cost=entry_cost,
            half_life=half_life, traded_notional=traded_notional,
        )
        self.open_positions[pair_id] = pos
        self._total_costs += entry_cost
        if self.margin.config.is_leveraged:
            self.margin.register_open(pair_id, total_notional)
        return pos

    def close_spread(
        self,
        pair_id: str, bar_idx: int,
        price_a: float, price_b: float, z_score: float,
        reason: str, cost_rate: float,
        exit_spread: float = 0.0,
        exit_beta: float = 0.0,
        exit_beta_P11: float = 0.0,
    ) -> Optional[CompletedTrade]:
        """Close a spread position and record the completed trade."""
        pos = self.open_positions.pop(pair_id, None)
        if pos is None:
            return None

        gross_pnl = pos.pnl(price_a, price_b)

        exit_traded = abs(pos.qty_a) * price_a + abs(pos.qty_b) * price_b
        exit_cost = exit_traded * cost_rate
        total_cost = pos.entry_cost + exit_cost
        net_pnl = gross_pnl - total_cost

        ct = CompletedTrade(
            pair_id=pos.pair_id, sym_a=pos.sym_a, sym_b=pos.sym_b,
            sector=pos.sector, direction=pos.direction, beta=pos.beta,
            entry_bar=pos.entry_bar, exit_bar=bar_idx,
            bars_held=bar_idx - pos.entry_bar,
            entry_price_a=pos.entry_price_a, entry_price_b=pos.entry_price_b,
            exit_price_a=price_a, exit_price_b=price_b,
            entry_z=pos.entry_z, exit_z=z_score, exit_reason=reason,
            qty_a=pos.qty_a,
            qty_b=pos.qty_b,
            notional=pos.notional,
            gross_pnl=gross_pnl, total_cost=total_cost, net_pnl=net_pnl,
            half_life=pos.half_life,
            entry_spread=pos.entry_spread,
            entry_beta_P11=pos.entry_beta_P11,
            entry_innov_S=pos.entry_innov_S,
            entry_confidence=pos.entry_confidence,
            entry_equity=pos.entry_equity,
            entry_implied_rho=pos.entry_implied_rho,
            entry_regime_conf=pos.entry_regime_conf,
            entry_adaptive_z=pos.entry_adaptive_z,
            entry_quality=pos.entry_quality,
            entry_z_peak=pos.entry_z_peak,
            exit_spread=exit_spread,
            exit_beta=exit_beta,
            exit_beta_P11=exit_beta_P11,
            exit_pnl_pct=net_pnl / pos.notional if pos.notional > 1e-10 else 0.0,
        )

        self.completed_trades.append(ct)
        self._realized_pnl += gross_pnl
        self._total_costs += exit_cost
        if self.margin.config.is_leveraged:
            self.margin.register_close(pair_id)
        return ct

    def rebalance_spread(
        self,
        pair_id: str, bar_idx: int,
        price_a: float, price_b: float,
        new_beta: float, vol_a: float, vol_b: float,
        cost_rate: float,
    ) -> float:
        """
        In-place hedge rebalance: adjust leg quantities without closing.
        Trades only the delta in each leg. Returns the rebalance cost.
        """
        pos = self.open_positions.get(pair_id)
        if pos is None:
            return 0.0

        accrued_pnl = pos.pnl(price_a, price_b)
        self._realized_pnl += accrued_pnl

        from .volatility import vol_adjusted_notionals
        notional_a, notional_b = vol_adjusted_notionals(
            pos.notional, new_beta, vol_a, vol_b)

        new_qty_a = pos.direction * notional_a / price_a
        new_qty_b = -pos.direction * np.sign(new_beta) * notional_b / price_b

        delta_a = abs(new_qty_a - pos.qty_a) * price_a
        delta_b = abs(new_qty_b - pos.qty_b) * price_b
        rebal_cost = (delta_a + delta_b) * cost_rate
        self._total_costs += rebal_cost

        pos.entry_price_a = price_a
        pos.entry_price_b = price_b
        pos.qty_a = new_qty_a
        pos.qty_b = new_qty_b
        pos.beta = new_beta
        pos.entry_cost += rebal_cost
        pos.traded_notional += delta_a + delta_b

        return rebal_cost

    def force_liquidation(
        self,
        bar_idx: int, timestamp,
        raw_row: np.ndarray,
        col_a: dict[str, int], col_b: dict[str, int],
        cost_rate: float,
    ) -> float:
        """Close ALL positions at current prices (margin liquidation)."""
        equity_before = self.equity_from_row(raw_row, col_a, col_b)
        n_closed = len(self.open_positions)

        total_exposure = sum(self.margin._position_notionals.values())

        for pid in list(self.open_positions.keys()):
            ca = col_a.get(pid)
            cb = col_b.get(pid)
            if ca is not None and cb is not None:
                pa, pb = raw_row[ca], raw_row[cb]
                if np.isfinite(pa) and np.isfinite(pb) and pa > 0 and pb > 0:
                    self.close_spread(pid, bar_idx, pa, pb, 0.0,
                                      "liquidation", cost_rate)
                    continue
            pos = self.open_positions.pop(pid, None)
            if pos is not None:
                self._realized_pnl += pos.pnl(pos.entry_price_a, pos.entry_price_b)

        liq_fee = total_exposure * self.margin.config.liquidation_fee_rate
        self._total_costs += liq_fee

        equity_after = self.equity_from_row(raw_row, col_a, col_b)

        self.margin.execute_liquidation(
            bar_idx, timestamp, equity_before, equity_after, n_closed)

        return equity_after

    def snapshot(self, bar_idx: int, timestamp,
                 row: np.ndarray,
                 col_a: dict[str, int],
                 col_b: dict[str, int]) -> dict:
        """Record equity and position state for the equity curve."""
        eq = self.equity_from_row(row, col_a, col_b)

        unrealized = 0.0
        for pid, pos in self.open_positions.items():
            ca, cb = col_a.get(pid), col_b.get(pid)
            if ca is not None and cb is not None:
                pa, pb = row[ca], row[cb]
                if np.isfinite(pa) and np.isfinite(pb):
                    unrealized += pos.pnl(pa, pb)

        snap = {
            "bar": bar_idx,
            "timestamp": timestamp,
            "total_equity": eq,
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": unrealized,
            "total_costs": self._total_costs,
            "n_open": len(self.open_positions),
        }
        if self.margin.config.is_leveraged:
            snap.update(self.margin.snapshot_fields(eq))
            self.margin.update_peak_metrics(eq)
        self.equity_history.append(snap)
        return snap

    def sector_exposure(self) -> dict[str, float]:
        """Total notional deployed per sector."""
        exposure = defaultdict(float)
        for pos in self.open_positions.values():
            exposure[pos.sector] += pos.notional
        return dict(exposure)

    def sector_count(self) -> dict[str, int]:
        """Number of open pairs per sector."""
        counts = defaultdict(int)
        for pos in self.open_positions.values():
            counts[pos.sector] += 1
        return dict(counts)

    @property
    def total_allocated(self) -> float:
        """Total notional currently deployed across all positions."""
        return sum(p.notional for p in self.open_positions.values())

    def trade_log_df(self) -> pd.DataFrame:
        if not self.completed_trades:
            return pd.DataFrame()
        return pd.DataFrame([ct.to_dict() for ct in self.completed_trades])

    def equity_df(self) -> pd.DataFrame:
        if not self.equity_history:
            return pd.DataFrame()
        return pd.DataFrame(self.equity_history)
