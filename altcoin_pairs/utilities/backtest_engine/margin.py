"""
margin.py -- cross-margin account tracking for spot margin trading

Simulates Binance spot cross-margin:
  - Borrow against equity to take larger positions
  - Hourly interest on the borrowed amount (portion exceeding collateral)
  - Forced liquidation when equity falls below maintenance margin

At max_leverage=1.0 (default), all margin logic is bypassed and the
backtest behaves identically to cash-only mode. The is_leveraged gate
ensures zero overhead on the hot path when leverage is not used.
"""

from dataclasses import dataclass


@dataclass
class MarginConfig:
    """
    Configuration for cross-margin account.

    Defaults are backward-compatible (1x = no leverage).
    """
    max_leverage: float = 1.0              # 1.0 = cash-only (no borrowing)
    maintenance_margin_rate: float = 0.05  # 5% of notional (Binance spot cross)
    liquidation_fee_rate: float = 0.02     # 2% insurance fund fee on liquidation
    margin_interest_rate: float = 0.0003   # ~0.03% per hour (Binance typical)
    interest_interval_bars: int = 4        # 1 hour / 15-min bars = 4 bars

    @property
    def is_leveraged(self) -> bool:
        return self.max_leverage > 1.0 + 1e-9


class MarginManager:
    """
    Tracks margin state for a cross-margin account.

    All margin/interest/liquidation logic lives here. The portfolio and
    strategy modules call into this class but never compute margin math
    themselves.
    """

    def __init__(self, config: MarginConfig):
        self.config = config

        # per-position tracking
        self._position_notionals: dict[str, float] = {}

        # aggregate margin state
        self._total_initial_margin: float = 0.0
        self._total_maintenance_margin: float = 0.0

        # interest tracking
        self._total_interest_paid: float = 0.0
        self._last_interest_bar: int = -999999

        # liquidation tracking
        self._liquidation_count: int = 0
        self._liquidation_events: list[dict] = []

        # peak metrics
        self._peak_margin_usage: float = 0.0    # max (initial_margin / equity)
        self._peak_leverage_actual: float = 0.0  # max (total_exposure / equity)

    # position registration

    def register_open(self, pair_id: str, notional: float) -> None:
        """Register a new position's margin requirements."""
        self._position_notionals[pair_id] = notional
        self._total_initial_margin += notional / self.config.max_leverage
        self._total_maintenance_margin += notional * self.config.maintenance_margin_rate

    def register_close(self, pair_id: str) -> None:
        """Remove a closed position from margin tracking."""
        notional = self._position_notionals.pop(pair_id, 0.0)
        if notional > 0:
            self._total_initial_margin -= notional / self.config.max_leverage
            self._total_maintenance_margin -= notional * self.config.maintenance_margin_rate
            # guard against float drift
            self._total_initial_margin = max(self._total_initial_margin, 0.0)
            self._total_maintenance_margin = max(self._total_maintenance_margin, 0.0)

    # margin checks

    def can_open(self, notional: float, equity: float) -> tuple[bool, float]:
        """
        Check if there's enough free margin and total exposure room.

        Returns (can_open, max_notional). If can_open is False but
        max_notional > 0, the caller can scale down the position.
        """
        lev = self.config.max_leverage

        # Cap 1: free margin
        initial_margin_required = notional / lev
        free_margin = equity - self._total_initial_margin
        if free_margin <= 0:
            return False, 0.0

        if initial_margin_required > free_margin:
            notional = free_margin * lev

        # Cap 2: total exposure limit
        total_exposure = sum(self._position_notionals.values())
        max_exposure = equity * lev
        if total_exposure + notional > max_exposure:
            notional = max_exposure - total_exposure

        if notional <= 0:
            return False, 0.0

        return True, notional

    # interest

    def apply_interest(self, bar_idx: int, equity: float) -> float:
        """
        Charge interest on borrowed amount every interest_interval_bars.

        Borrowed = max(total_exposure - equity, 0).
        Returns interest charged this call (0.0 if not due yet).
        """
        if bar_idx - self._last_interest_bar < self.config.interest_interval_bars:
            return 0.0

        self._last_interest_bar = bar_idx

        total_exposure = sum(self._position_notionals.values())
        borrowed = max(total_exposure - equity, 0.0)
        if borrowed <= 0:
            return 0.0

        interest = borrowed * self.config.margin_interest_rate
        self._total_interest_paid += interest
        return interest

    # liquidation

    def check_liquidation(self, equity: float) -> bool:
        """Returns True if equity has fallen to or below maintenance margin."""
        if self._total_maintenance_margin <= 0:
            return False
        return equity <= self._total_maintenance_margin

    def execute_liquidation(self, bar: int, timestamp,
                            equity_before: float, equity_after: float,
                            n_closed: int) -> None:
        """Record a liquidation event and reset all margin state."""
        self._liquidation_count += 1
        self._liquidation_events.append({
            "bar": bar,
            "timestamp": timestamp,
            "equity_before": equity_before,
            "equity_after": equity_after,
            "n_closed": n_closed,
        })
        # clear all margin tracking
        self._position_notionals.clear()
        self._total_initial_margin = 0.0
        self._total_maintenance_margin = 0.0

    # peak metrics and snapshots

    def update_peak_metrics(self, equity: float) -> None:
        """Track peak margin usage and actual leverage for reporting."""
        if equity <= 0:
            return
        total_exposure = sum(self._position_notionals.values())

        margin_usage = self._total_initial_margin / equity
        if margin_usage > self._peak_margin_usage:
            self._peak_margin_usage = margin_usage

        actual_leverage = total_exposure / equity
        if actual_leverage > self._peak_leverage_actual:
            self._peak_leverage_actual = actual_leverage

    def snapshot_fields(self, equity: float) -> dict:
        """Return margin fields for the equity snapshot."""
        total_exposure = sum(self._position_notionals.values())
        return {
            "margin_used": self._total_initial_margin,
            "margin_free": max(equity - self._total_initial_margin, 0.0),
            "margin_ratio": (self._total_initial_margin / equity
                             if equity > 0 else 0.0),
            "actual_leverage": total_exposure / equity if equity > 0 else 0.0,
            "total_exposure": total_exposure,
            "interest_paid": self._total_interest_paid,
        }

    def summary_fields(self) -> dict:
        """Return final summary fields for metrics/reporting."""
        return {
            "max_leverage_setting": self.config.max_leverage,
            "liquidation_events": self._liquidation_count,
            "peak_margin_usage": self._peak_margin_usage,
            "peak_actual_leverage": self._peak_leverage_actual,
            "total_interest_paid": self._total_interest_paid,
        }
