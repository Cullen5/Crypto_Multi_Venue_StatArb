"""
models.py -- data classes for spread positions and completed trades
"""

from dataclasses import dataclass


@dataclass
class SpreadPosition:
    """
    Open 2-leg spread. PnL from quantities and price deltas (no cash ledger).

    direction = +1: long spread (buy A, sell B when beta > 0)
    direction = -1: short spread (sell A, buy B when beta > 0)
    """
    pair_id: str
    sym_a: str
    sym_b: str
    sector: str
    direction: int
    beta: float

    entry_bar: int
    entry_price_a: float
    entry_price_b: float
    entry_z: float

    qty_a: float
    qty_b: float
    notional: float

    entry_cost: float
    half_life: float = 0.0
    traded_notional: float = 0.0

    entry_spread: float = 0.0
    entry_beta_P11: float = 0.0
    entry_innov_S: float = 0.0
    entry_confidence: float = 1.0
    entry_equity: float = 0.0
    entry_implied_rho: float = 0.0
    entry_regime_conf: float = 1.0
    entry_adaptive_z: float = 1.5
    entry_quality: float = 0.0
    entry_z_peak: float = 1.0

    def pnl(self, price_a: float, price_b: float) -> float:
        return (self.qty_a * (price_a - self.entry_price_a)
                + self.qty_b * (price_b - self.entry_price_b))

    def pnl_pct(self, price_a: float, price_b: float) -> float:
        if self.notional < 1e-10:
            return 0.0
        return self.pnl(price_a, price_b) / self.notional


@dataclass
class CompletedTrade:
    """
    Record of a completed spread trade with full entry/exit audit trail.

    exit_reason is one of:
        "signal"      -- z-score reverted through exit threshold
        "stop_loss"   -- pair pct stop hit
        "z_stop"      -- z-score blew past hard stop
        "max_hold"    -- held too long relative to half-life
        "beta_break"  -- beta drifted 10%+, closed with 7-day cooldown
        "stale"       -- held >= 1x half-life with insufficient z-progress
        "forced"      -- asset went null or pair dropped from active set
        "end"         -- backtest ended with position still open
    """
    pair_id: str
    sym_a: str
    sym_b: str
    sector: str
    direction: int
    beta: float

    entry_bar: int
    exit_bar: int
    bars_held: int

    entry_price_a: float
    entry_price_b: float
    exit_price_a: float
    exit_price_b: float

    entry_z: float
    exit_z: float
    exit_reason: str

    qty_a: float
    qty_b: float
    notional: float

    gross_pnl: float
    total_cost: float
    net_pnl: float

    half_life: float = 0.0

    entry_spread: float = 0.0
    entry_beta_P11: float = 0.0
    entry_innov_S: float = 0.0
    entry_confidence: float = 1.0
    entry_equity: float = 0.0
    entry_implied_rho: float = 0.0
    entry_regime_conf: float = 1.0
    entry_adaptive_z: float = 1.5
    entry_quality: float = 0.0
    entry_z_peak: float = 1.0

    exit_spread: float = 0.0
    exit_beta: float = 0.0
    exit_beta_P11: float = 0.0
    exit_pnl_pct: float = 0.0

    def to_dict(self) -> dict:
        """Flat dict for DataFrame export."""
        eq = self.entry_equity if self.entry_equity > 0 else 1.0
        return {
            "pair_id": self.pair_id,
            "sym_a": self.sym_a,
            "sym_b": self.sym_b,
            "sector": self.sector,
            "direction": self.direction,
            "beta_entry": round(self.beta, 6),
            "beta_exit": round(self.exit_beta, 6),
            "entry_bar": self.entry_bar,
            "exit_bar": self.exit_bar,
            "bars_held": self.bars_held,
            "half_life": round(self.half_life, 1),
            "entry_price_a": round(self.entry_price_a, 6),
            "entry_price_b": round(self.entry_price_b, 6),
            "exit_price_a": round(self.exit_price_a, 6),
            "exit_price_b": round(self.exit_price_b, 6),
            "entry_z": round(self.entry_z, 4),
            "exit_z": round(self.exit_z, 4),
            "entry_spread": round(self.entry_spread, 6),
            "exit_spread": round(self.exit_spread, 6),
            "exit_reason": self.exit_reason,
            "entry_beta_P11": f"{self.entry_beta_P11:.2e}",
            "exit_beta_P11": f"{self.exit_beta_P11:.2e}",
            "entry_innov_S": f"{self.entry_innov_S:.2e}",
            "entry_confidence": round(self.entry_confidence, 3),
            "entry_implied_rho": round(self.entry_implied_rho, 4),
            "entry_regime_conf": round(self.entry_regime_conf, 3),
            "entry_adaptive_z": round(self.entry_adaptive_z, 3),
            "entry_quality": round(self.entry_quality, 3),
            "entry_z_peak": round(self.entry_z_peak, 3),
            "notional": round(self.notional, 2),
            "entry_equity": round(self.entry_equity, 2),
            "position_pct": round(self.notional / eq, 4),
            "gross_pnl": round(self.gross_pnl, 2),
            "total_cost": round(self.total_cost, 4),
            "net_pnl": round(self.net_pnl, 2),
            "pnl_pct": round(self.exit_pnl_pct, 4),
        }
