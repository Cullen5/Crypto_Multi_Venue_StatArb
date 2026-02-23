"""
btc_futures_backtest.py — BTC Futures Calendar Spread Backtester
=================================================================
Strategy A: Traditional Calendar Spreads (CEX focus)
Per case study section 3.2.

Trade logic (carry / convergence):
  Contango entry:  annualized basis > 15%
    → Long spot, short far futures
    → Basis decays toward 0% at expiry → collect the spread
    → Exit: basis < 5%  OR  near expiry  OR  stop

  Backwardation entry: annualized basis < -10%
    → Short spot, long far futures
    → Basis normalizes toward 0% → collect the spread
    → Exit: basis > -3%  OR  near expiry  OR  stop

Enhancements on top of base case study spec:
  1. Full term structure: score all active contracts, pick best carry
  2. Z-score timing: prefer entry when basis is unusually rich vs history
  3. Regime-aware sizing: scale position by market regime
  4. Costs via costs.py venue model

P&L (delta-neutral spot vs futures):
  Short basis (contango): P&L = notional × (entry_basis − exit_basis) / 100
  Long basis (backwardation): P&L = notional × (exit_basis − entry_basis) / 100

Usage:
    from data_loader import DataLoader
    from btc_futures_backtest import run_calendar_spreads, CalendarConfig

    dl = DataLoader("./data/raw")
    result = run_calendar_spreads(dl)
    result.summary()
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

try:
    from utilities.options_calendar_spread.costs import get_venue_costs, VenueCosts
except ImportError:
    try:
        from costs import get_venue_costs, VenueCosts
    except ImportError:
        raise ImportError(
            "Cannot find costs module. Expected at either:\n"
            "  utilities/options_calendar_spread/costs.py  OR\n"
            "  costs.py (same directory)"
        )


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

@dataclass
class CalendarConfig:
    """All hyperparameters for Strategy A calendar spreads."""

    # ── Capital ──
    initial_capital: float = 100_000

    # ── Walk-forward ──
    train_end: str = "2023-06-30"
    test_start: str = "2023-07-01"

    # ── Data filters ──
    margin_type: str = "USDT-M"
    min_dte: int = 14             # ignore contracts < 14 DTE (noisy annualisation)
    max_dte: int = 200            # ignore contracts > 200 DTE
    roll_days: int = 5            # front-month rolls N days before expiry

    # ═════════════════════════════════════
    #  ENTRY thresholds (case study spec)
    # ═════════════════════════════════════
    contango_entry_ann: float = 15.0    # enter short-basis when ann > 15%
    backwardation_entry_ann: float = -10.0  # enter long-basis when ann < -10%

    # ═════════════════════════════════════
    #  EXIT thresholds (convergence)
    # ═════════════════════════════════════
    contango_exit_ann: float = 5.0      # exit short-basis when ann < 5%
    backwardation_exit_ann: float = -3.0  # exit long-basis when ann > -3%

    # ═════════════════════════════════════
    #  Z-SCORE enhancement (timing)
    # ═════════════════════════════════════
    # Z-score is computed on the front-month annualised basis.
    # Used to refine entry timing and scale position size,
    # NOT as the primary signal.
    z_lookback: int = 60          # rolling window (trading days)
    z_min_lookback: int = 20      # minimum obs before z valid
    z_entry_boost: float = 1.0    # prefer entry when z > this (basis rich vs history)
    z_full_size: float = 2.5      # at |z| >= this, position at regime-max
    z_stop: float = -2.0          # emergency exit: z flipped hard against us
                                  #   (for short-basis: z drops to -2 means basis collapsed
                                  #    below its historical mean, something changed)

    # ── Contract selection from term structure ──
    optimal_dte: int = 75         # prefer ~75 DTE (enough time to collect carry)
    dte_weight_spread: float = 40.0

    # ── Position management ──
    base_position_pct: float = 1.0   # base fraction of equity per trade
    max_positions: int = 3
    max_hold_days: int = 150          # forced exit (safety, should exit on convergence first)
    stop_loss_basis_pct: float = 5.0  # exit if basis moves 5% AGAINST us (raw %)
    min_trade_usd: float = 10_000

    # ── Regime sizing multipliers ──
    # Carry works BETTER in contango → larger size
    regime_size_steep_contango: float = 1.0    # >20% ann: full size, best environment
    regime_size_mild_contango: float = 0.85    # 5-20%: good, slightly less
    regime_size_flat: float = 0.40             # no clear carry → small
    regime_size_backwardation: float = 0.70    # backwardation trades possible but less reliable

    # ── Costs (via costs.py) ──
    spot_venue: str = "binance_spot"
    futures_venue: str = "binance"        # used for cost model (pick primary)
    futures_venues: str = "binance"       # data venues: "binance", "deribit", or "binance,deribit"

    # ── Re-entry cooldown ──
    cooldown_days: int = 3  # don't re-enter same contract within N days of exit

    @property
    def _spot_vc(self) -> VenueCosts:
        return get_venue_costs(self.spot_venue)

    @property
    def _futures_vc(self) -> VenueCosts:
        return get_venue_costs(self.futures_venue)

    @property
    def entry_cost_rate(self) -> float:
        svc, fvc = self._spot_vc, self._futures_vc
        return (svc.taker_fee + svc.avg_slippage) + (fvc.taker_fee + fvc.avg_slippage)

    @property
    def exit_cost_rate(self) -> float:
        svc, fvc = self._spot_vc, self._futures_vc
        return (svc.maker_fee + svc.avg_slippage) + (fvc.maker_fee + fvc.avg_slippage)

    @property
    def round_trip_rate(self) -> float:
        return self.entry_cost_rate + self.exit_cost_rate

    @property
    def gas_cost_usd(self) -> float:
        return (self._spot_vc.gas_cost_usd + self._futures_vc.gas_cost_usd) * 2

    def regime_multiplier(self, regime: str) -> float:
        return {
            "steep_contango": self.regime_size_steep_contango,
            "mild_contango": self.regime_size_mild_contango,
            "flat": self.regime_size_flat,
            "backwardation": self.regime_size_backwardation,
        }.get(regime, 0.4)

    def dte_weight(self, dte: float) -> float:
        """Gaussian preference peaking at optimal_dte."""
        return np.exp(-0.5 * ((dte - self.optimal_dte) / self.dte_weight_spread) ** 2)

    def to_dict(self) -> dict:
        d = {}
        for k in self.__dataclass_fields__:
            d[k] = getattr(self, k)
        d["round_trip_rate"] = self.round_trip_rate
        d["entry_cost_rate"] = self.entry_cost_rate
        d["exit_cost_rate"] = self.exit_cost_rate
        d["gas_cost_usd"] = self.gas_cost_usd
        d["spot_vc"] = f"{self._spot_vc.name} ({self._spot_vc.venue_type.value})"
        d["futures_vc"] = f"{self._futures_vc.name} ({self._futures_vc.venue_type.value})"
        return d


# ═══════════════════════════════════════════════════════════════
#  TRADE / POSITION MODELS
# ═══════════════════════════════════════════════════════════════

@dataclass
class Trade:
    trade_id: str
    contract: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    days_held: float
    direction: str              # "short_basis" (contango) or "long_basis" (backwardation)
    entry_basis_pct: float      # raw basis %
    exit_basis_pct: float
    entry_ann_basis_pct: float  # annualised at entry
    exit_ann_basis_pct: float
    entry_z: float
    exit_z: float
    entry_spot: float
    entry_futures: float
    exit_spot: float
    exit_futures: float
    entry_dte: float
    exit_dte: float
    exit_reason: str
    regime_at_entry: str
    regime_multiplier: float
    z_size_scalar: float
    notional: float
    gross_pnl: float
    cost: float
    net_pnl: float
    pnl_pct: float
    carry_collected_ann_pct: float  # annualised basis captured

    def to_dict(self) -> dict:
        d = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, pd.Timestamp):
                v = str(v.date())
            elif isinstance(v, float):
                v = round(v, 4)
            d[k] = v
        return d


@dataclass
class OpenPosition:
    trade_id: str
    contract: str
    entry_date: pd.Timestamp
    direction: str
    entry_basis_pct: float
    entry_ann_basis_pct: float
    entry_z: float
    entry_spot: float
    entry_futures: float
    entry_dte: float
    regime_at_entry: str
    regime_multiplier: float
    z_size_scalar: float
    notional: float

    def mark_to_market(self, current_basis_pct: float) -> float:
        """
        Short basis (contango): profit when basis FALLS
        Long basis (backwardation): profit when basis RISES
        """
        delta = current_basis_pct - self.entry_basis_pct
        sign = -1.0 if self.direction == "short_basis" else 1.0
        return self.notional * sign * delta / 100.0


# ═══════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════

def compute_metrics(equity_series: pd.Series, trades: list[Trade]) -> dict:
    if len(equity_series) < 2:
        return {"error": "insufficient data"}

    returns = equity_series.pct_change().dropna()
    total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
    n_days = (equity_series.index[-1] - equity_series.index[0]).days
    ann_factor = 365.25 / max(n_days, 1)

    # Annualisation factor from actual observation frequency
    periods_per_year = len(returns) / max(n_days / 365.25, 1e-6)
    rf_per_period = 0.04 / periods_per_year  # 4% annual risk-free rate

    excess = returns - rf_per_period
    sharpe = (excess.mean() / excess.std() * np.sqrt(periods_per_year)
              if excess.std() > 1e-10 else 0.0)

    # Sortino: proper downside deviation (target = rf, all observations)
    downside_sq = np.minimum(excess.values, 0.0) ** 2
    downside_dev = np.sqrt(np.mean(downside_sq))
    sortino = (excess.mean() / downside_dev * np.sqrt(periods_per_year)
               if downside_dev > 1e-10 else 0.0)
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = drawdown.min()
    ann_return = (1 + total_return) ** ann_factor - 1
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    pnls = np.array([t.net_pnl for t in trades]) if trades else np.array([0.0])
    n_trades = len(trades)
    if n_trades > 0:
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        win_rate = len(wins) / n_trades
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        pf = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else np.inf
        total_cost = sum(t.cost for t in trades)
        avg_carry = np.mean([t.carry_collected_ann_pct for t in trades])
    else:
        win_rate = avg_win = avg_loss = total_cost = avg_carry = 0.0
        pf = 0.0

    regime_counts, regime_pnl = {}, {}
    for t in trades:
        r = t.regime_at_entry
        regime_counts[r] = regime_counts.get(r, 0) + 1
        regime_pnl[r] = regime_pnl.get(r, 0.0) + t.net_pnl

    return {
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(ann_return * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar_ratio": round(calmar, 3),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 3),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(float(pf), 2) if np.isfinite(pf) else "inf",
        "total_pnl": round(float(pnls.sum()), 2),
        "total_cost": round(total_cost, 2),
        "cost_drag_pct": round(total_cost / max(equity_series.iloc[0], 1) * 100, 2),
        "avg_carry_collected_ann_pct": round(avg_carry, 2),
        "avg_days_held": round(np.mean([t.days_held for t in trades]), 1) if trades else 0,
        "n_days_traded": n_days,
        "regime_trades": regime_counts,
        "regime_pnl": {k: round(v, 2) for k, v in regime_pnl.items()},
    }


def print_summary(metrics: dict, title: str = "Strategy A"):
    _pct_keys = {"total_return_pct", "annualized_return_pct", "max_drawdown_pct",
                 "cost_drag_pct", "avg_carry_collected_ann_pct"}
    _rate_keys = {"win_rate"}  # stored as 0–1 decimal, display as %

    print(f"\n{'═' * 62}")
    print(f"  {title}")
    print(f"{'═' * 62}")
    for k, v in metrics.items():
        label = k.replace("_", " ").title()
        if isinstance(v, dict):
            print(f"  {label}:")
            for kk, vv in v.items():
                print(f"    {kk:24s} {str(vv):>10s}")
        elif k in _rate_keys and isinstance(v, (int, float)):
            print(f"  {label:32s} {v * 100:>11.1f}%")
        elif k in _pct_keys and isinstance(v, (int, float)):
            print(f"  {label:32s} {v:>11.2f}%")
        elif isinstance(v, float):
            print(f"  {label:32s} {v:>12.2f}")
        else:
            print(f"  {label:32s} {str(v):>12s}")
    print(f"{'═' * 62}\n")


# ═══════════════════════════════════════════════════════════════
#  RESULTS CONTAINER
# ═══════════════════════════════════════════════════════════════

@dataclass
class BacktestResults:
    strategy: str
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    metrics: dict
    config: dict
    daily_data: pd.DataFrame = None

    def summary(self):
        print_summary(self.metrics, title=self.strategy)

    def plot_equity(self, ax=None, benchmark_spot=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 5))
        eq = self.equity_curve.set_index("date")["equity"]
        ax.plot(eq.index, eq.values, label="Calendar Spread", linewidth=1.5, color="#1f77b4")
        if benchmark_spot is not None:
            bm = benchmark_spot / benchmark_spot.iloc[0] * eq.iloc[0]
            ax.plot(bm.index, bm.values, label="BTC Buy & Hold",
                    alpha=0.5, linewidth=1, color="#7f7f7f")
        ax.set_ylabel("Equity ($)")
        ax.set_title("Strategy A: Calendar Spreads — Equity")
        ax.legend(); ax.grid(alpha=0.3)
        return ax

    def plot_drawdown(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 3))
        eq = self.equity_curve.set_index("date")["equity"]
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax.fill_between(dd.index, dd.values, 0, alpha=0.5, color="#d62728")
        ax.set_ylabel("Drawdown (%)"); ax.set_title("Strategy A — Drawdown")
        ax.grid(alpha=0.3)
        return ax

    def plot_basis_and_trades(self, ax=None):
        """Plot annualised basis with trade entry/exit shading."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 5))
        if self.daily_data is not None and "annualized_basis_pct" in self.daily_data.columns:
            dd = self.daily_data.set_index("date")
            ax.plot(dd.index, dd["annualized_basis_pct"], linewidth=0.8,
                    color="#1f77b4", alpha=0.7, label="Ann. Basis (%)")
            ax.axhline(0, color="black", linewidth=0.5)
            # Draw thresholds from config
            _cfg = self.config
            ax.axhline(_cfg.get("contango_entry_ann", 15), color="#2ca02c",
                        linestyle="--", alpha=0.5, label="Contango entry")
            ax.axhline(_cfg.get("contango_exit_ann", 5), color="#ff7f0e",
                        linestyle=":", alpha=0.5, label="Contango exit")
            ax.axhline(_cfg.get("backwardation_entry_ann", -10), color="#d62728",
                        linestyle="--", alpha=0.5, label="Backwardation entry")
        if not self.trade_log.empty:
            for _, t in self.trade_log.iterrows():
                entry_d = pd.to_datetime(t["entry_date"])
                exit_d = pd.to_datetime(t["exit_date"])
                color = "#2ca02c" if t.get("net_pnl", 0) > 0 else "#d62728"
                ax.axvspan(entry_d, exit_d, alpha=0.12, color=color)
        ax.set_ylabel("Annualised Basis (%)")
        ax.set_title("Strategy A — Basis & Trades")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        return ax

    def plot_regime_pnl(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        regime_pnl = self.metrics.get("regime_pnl", {})
        if regime_pnl:
            regimes = list(regime_pnl.keys())
            pnls = [regime_pnl[r] for r in regimes]
            colors = ["#2ca02c" if p > 0 else "#d62728" for p in pnls]
            ax.bar(regimes, pnls, color=colors, alpha=0.7)
            ax.set_ylabel("Net P&L ($)"); ax.set_title("P&L by Regime")
            ax.grid(alpha=0.3, axis="y")
        return ax

    def plot_term_structure(self, dl, dates: list[str] = None):
        import matplotlib.pyplot as plt
        if dates is None:
            dates = ["2023-01-15", "2023-06-15", "2024-01-15", "2024-06-15"]
        fig, axes = plt.subplots(1, len(dates), figsize=(5 * len(dates), 4), sharey=True)
        if len(dates) == 1: axes = [axes]
        mt = self.config.get("margin_type", "USDT-M")
        for ax, d in zip(axes, dates):
            try:
                snap = dl.term_structure_snapshot(d, margin_type=mt)
                ax.bar(range(len(snap)), snap["annualized_basis_pct"].values, alpha=0.7)
                ax.set_xticks(range(len(snap)))
                ax.set_xticklabels([f"{int(dte)}d" for dte in snap["days_to_expiry"]],
                                   rotation=45, fontsize=8)
                ax.set_title(d); ax.grid(alpha=0.3, axis="y")
            except Exception:
                ax.set_title(f"{d}\n(no data)")
        axes[0].set_ylabel("Ann. Basis (%)")
        fig.suptitle("BTC Futures Term Structure Snapshots", fontsize=13)
        fig.tight_layout()
        return fig


# ═══════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════

def _classify_regime(ann_basis: float) -> str:
    if pd.isna(ann_basis):
        return "flat"
    if ann_basis > 20:
        return "steep_contango"
    elif ann_basis > 5:
        return "mild_contango"
    elif ann_basis > -5:
        return "flat"
    else:
        return "backwardation"


def _prepare_daily_data(dl, cfg: CalendarConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build daily trading dataset.

    Returns:
        front_daily: continuous front-month series with z-scores + regime
        all_daily:   all contracts daily (for contract selection)
    """
    # ── All contracts: hourly → daily ──
    basis = dl.basis_table(margin_type=cfg.margin_type, venue=cfg.futures_venues)
    basis = basis[
        (basis["days_to_expiry"] > cfg.min_dte) &
        (basis["days_to_expiry"] < cfg.max_dte)
    ].copy()
    basis["date"] = basis["datetime"].dt.normalize().dt.tz_localize(None)
    all_daily = (basis.sort_values("datetime")
                 .groupby(["symbol", "date"]).last()
                 .reset_index())

    # ── Front-month continuous (for signal generation) ──
    front = dl.continuous_front_month(
        margin_type=cfg.margin_type,
        roll_days_before_expiry=cfg.roll_days,
        venue=cfg.futures_venues)
    front = front[
        (front["days_to_expiry"] > cfg.min_dte) &
        (front["days_to_expiry"] < cfg.max_dte)
    ].copy()
    front["date"] = front["datetime"].dt.normalize().dt.tz_localize(None)
    front_daily = (front.sort_values("datetime")
                   .groupby("date").last()
                   .reset_index())

    # ── Z-score on annualised basis (enhancement, not primary signal) ──
    lb = cfg.z_lookback
    front_daily["basis_mean"] = (front_daily["annualized_basis_pct"]
                                  .rolling(lb, min_periods=cfg.z_min_lookback).mean())
    front_daily["basis_std"] = (front_daily["annualized_basis_pct"]
                                 .rolling(lb, min_periods=cfg.z_min_lookback).std())
    front_daily["basis_std"] = front_daily["basis_std"].clip(lower=0.5)
    front_daily["z_score"] = (
        (front_daily["annualized_basis_pct"] - front_daily["basis_mean"])
        / front_daily["basis_std"]
    )

    # ── Regime ──
    front_daily["regime"] = front_daily["annualized_basis_pct"].apply(_classify_regime)

    return front_daily, all_daily


def _select_best_contract(
    date: pd.Timestamp,
    direction: str,
    all_daily: pd.DataFrame,
    cfg: CalendarConfig,
    open_contracts: set,
    cooldown_map: dict,
) -> Optional[dict]:
    """
    From all active contracts on this date, pick the one with the best
    carry profile for the given direction.

    For contango (short_basis):
      Want high annualised basis at a good DTE.
      Score = ann_basis × dte_weight(dte)

    For backwardation (long_basis):
      Want deeply negative annualised basis at good DTE.
      Score = |ann_basis| × dte_weight(dte)
    """
    today = all_daily[all_daily["date"] == date]
    if today.empty:
        return None

    candidates = []
    for _, row in today.iterrows():
        sym = row["symbol"]
        if sym in open_contracts:
            continue
        # Cooldown check
        last_exit = cooldown_map.get(sym)
        if last_exit is not None and (date - last_exit).days < cfg.cooldown_days:
            continue

        ann_basis = row["annualized_basis_pct"]
        raw_basis = row["basis_pct"]
        dte = row["days_to_expiry"]

        if pd.isna(ann_basis) or pd.isna(raw_basis):
            continue

        # Direction filter: only contracts that meet the entry threshold
        if direction == "short_basis" and ann_basis < cfg.contango_entry_ann:
            continue
        if direction == "long_basis" and ann_basis > cfg.backwardation_entry_ann:
            continue

        dte_w = cfg.dte_weight(dte)
        score = abs(ann_basis) * dte_w

        candidates.append({
            "symbol": sym,
            "dte": dte,
            "basis_pct": raw_basis,
            "ann_basis_pct": ann_basis,
            "spot_close": row["spot_close"],
            "futures_close": row["futures_close"],
            "expiry_date": row.get("expiry_date", ""),
            "score": score,
        })

    if not candidates:
        return None
    return max(candidates, key=lambda c: c["score"])


# ═══════════════════════════════════════════════════════════════
#  MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════

def run_calendar_spreads(
    dl,
    cfg: CalendarConfig = None,
    test_only: bool = True,
) -> BacktestResults:
    """
    Run the calendar spread backtest per case study Strategy A.

    Parameters
    ----------
    dl : DataLoader
    cfg : CalendarConfig (defaults if None)
    test_only : if True, only trade in test period (train for z-score warmup)
    """
    if cfg is None:
        cfg = CalendarConfig()

    print(f"\n{'═' * 62}")
    print(f"  STRATEGY A: Calendar Spreads (Carry / Convergence)")
    print(f"{'═' * 62}")
    print(f"  Entry: contango >{cfg.contango_entry_ann}% ann, "
          f"backwardation <{cfg.backwardation_entry_ann}% ann")
    print(f"  Exit:  contango <{cfg.contango_exit_ann}% ann, "
          f"backwardation >{cfg.backwardation_exit_ann}% ann")
    print(f"  Z-score enhancement: lookback={cfg.z_lookback}d, "
          f"entry_boost={cfg.z_entry_boost}, z_stop={cfg.z_stop}")
    print(f"  DTE filter: {cfg.min_dte}–{cfg.max_dte}d, "
          f"optimal={cfg.optimal_dte}d")
    print(f"  Position: {cfg.base_position_pct:.0%} base × regime × z_scalar, "
          f"max {cfg.max_positions}")
    print(f"  Data venues: {cfg.futures_venues}")
    print(f"  Costs: {cfg.round_trip_rate:.4%} RT "
          f"({cfg.spot_venue} + {cfg.futures_venue})"
          f"{f', gas=${cfg.gas_cost_usd:.2f}' if cfg.gas_cost_usd > 0 else ''}")

    # ── Prepare data ──
    front_daily, all_daily = _prepare_daily_data(dl, cfg)

    print(f"\n  Full range: {front_daily['date'].min().date()} → "
          f"{front_daily['date'].max().date()}")
    venue_info = (f" ({all_daily['venue'].nunique()} venue(s))"
                  if "venue" in all_daily.columns else "")
    print(f"  Contracts: {all_daily['symbol'].nunique()} unique{venue_info}")

    # Filter to trading period
    if test_only:
        test_ts = pd.Timestamp(cfg.test_start)
        trade_mask = front_daily["date"] >= test_ts
    else:
        trade_mask = front_daily["z_score"].notna()

    trade_days = front_daily[trade_mask].copy().reset_index(drop=True)
    if trade_days.empty:
        raise ValueError("No trading days after filtering.")

    print(f"  Trading: {trade_days['date'].iloc[0].date()} → "
          f"{trade_days['date'].iloc[-1].date()} ({len(trade_days)} days)")

    # Regime distribution
    for regime, cnt in trade_days["regime"].value_counts().items():
        print(f"    {regime}: {cnt} days ({cnt / len(trade_days):.0%})")

    # How many days above entry thresholds?
    # Front-month only
    n_above_front = (trade_days["annualized_basis_pct"] > cfg.contango_entry_ann).sum()
    n_below_front = (trade_days["annualized_basis_pct"] < cfg.backwardation_entry_ann).sum()
    print(f"  Front-month days ann >{cfg.contango_entry_ann}%: {n_above_front} "
          f"({n_above_front/len(trade_days):.0%})")

    # FULL CURVE: any contract above threshold on a given day
    curve_max = (all_daily.groupby("date")["annualized_basis_pct"].max()
                 .reindex(trade_days["date"].values))
    curve_min = (all_daily.groupby("date")["annualized_basis_pct"].min()
                 .reindex(trade_days["date"].values))
    n_above_any = (curve_max > cfg.contango_entry_ann).sum()
    n_below_any = (curve_min < cfg.backwardation_entry_ann).sum()
    print(f"  Any-contract days ann >{cfg.contango_entry_ann}%: {n_above_any} "
          f"({n_above_any/len(trade_days):.0%})  ← full curve")
    print(f"  Any-contract days ann <{cfg.backwardation_entry_ann}%: {n_below_any} "
          f"({n_below_any/len(trade_days):.0%})")
    n_contracts_per_day = all_daily.groupby("date")["symbol"].nunique()
    print(f"  Avg active contracts/day: {n_contracts_per_day.mean():.1f}")

    # ── Backtest loop ──
    equity = cfg.initial_capital
    positions: list[OpenPosition] = []
    completed: list[Trade] = []
    equity_hist = []
    trade_counter = 0
    cooldown_map: dict[str, pd.Timestamp] = {}  # contract → last exit date

    for _, row in trade_days.iterrows():
        date = row["date"]
        ann_basis = row["annualized_basis_pct"]
        raw_basis = row.get("basis_pct", np.nan)
        spot = row["spot_close"]
        regime = row["regime"]
        z = row.get("z_score", np.nan)

        # ═══════════════════════════
        #  CHECK EXITS
        # ═══════════════════════════
        for pos in list(positions):
            days_held = (date - pos.entry_date).days

            # Current basis for THIS contract
            contract_row = all_daily[
                (all_daily["date"] == date) &
                (all_daily["symbol"] == pos.contract)
            ]

            if contract_row.empty:
                # Contract expired or no data → force close
                cur_basis = raw_basis if not pd.isna(raw_basis) else pos.entry_basis_pct
                cur_ann = ann_basis if not pd.isna(ann_basis) else pos.entry_ann_basis_pct
                cur_dte = max(pos.entry_dte - days_held, 0)
                cur_spot = spot if not pd.isna(spot) else pos.entry_spot
                cur_futures = cur_spot * (1 + cur_basis / 100)
                exit_reason = "near_expiry"
            else:
                cr = contract_row.iloc[0]
                cur_basis = cr["basis_pct"]
                cur_ann = cr["annualized_basis_pct"]
                cur_dte = cr["days_to_expiry"]
                cur_spot = cr["spot_close"]
                cur_futures = cr["futures_close"]
                exit_reason = None

            mtm = pos.mark_to_market(cur_basis)

            # ── Exit conditions (priority order) ──

            # 1. Near expiry — basis should be near 0 anyway
            if cur_dte <= cfg.min_dte and exit_reason is None:
                exit_reason = "near_expiry"

            # 2. CONVERGENCE EXIT (primary — case study spec)
            if exit_reason is None:
                if pos.direction == "short_basis" and cur_ann < cfg.contango_exit_ann:
                    exit_reason = "converged"    # basis decayed from >15% to <5%
                elif pos.direction == "long_basis" and cur_ann > cfg.backwardation_exit_ann:
                    exit_reason = "converged"

            # 3. Stop loss — basis moved AGAINST us
            if exit_reason is None:
                basis_move = cur_basis - pos.entry_basis_pct
                if pos.direction == "short_basis" and basis_move > cfg.stop_loss_basis_pct:
                    exit_reason = "stop_loss"    # contango got WORSE
                elif pos.direction == "long_basis" and basis_move < -cfg.stop_loss_basis_pct:
                    exit_reason = "stop_loss"    # backwardation deepened

            # 4. Z-score regime flip (enhancement)
            if exit_reason is None and not pd.isna(z):
                if pos.direction == "short_basis" and z < cfg.z_stop:
                    exit_reason = "z_regime_flip"  # basis collapsed below historical mean
                elif pos.direction == "long_basis" and z > -cfg.z_stop:
                    exit_reason = "z_regime_flip"

            # 5. Max hold
            if exit_reason is None and days_held >= cfg.max_hold_days:
                exit_reason = "max_hold"

            # ── Execute exit ──
            if exit_reason is not None:
                cost = pos.notional * cfg.round_trip_rate + cfg.gas_cost_usd
                net = mtm - cost
                equity += net
                cooldown_map[pos.contract] = date

                # Carry collected: difference in annualised basis
                if pos.direction == "short_basis":
                    carry = pos.entry_ann_basis_pct - cur_ann
                else:
                    carry = cur_ann - pos.entry_ann_basis_pct

                completed.append(Trade(
                    trade_id=pos.trade_id, contract=pos.contract,
                    entry_date=pos.entry_date, exit_date=date,
                    days_held=days_held, direction=pos.direction,
                    entry_basis_pct=pos.entry_basis_pct,
                    exit_basis_pct=cur_basis,
                    entry_ann_basis_pct=pos.entry_ann_basis_pct,
                    exit_ann_basis_pct=cur_ann,
                    entry_z=pos.entry_z, exit_z=z if not pd.isna(z) else 0,
                    entry_spot=pos.entry_spot, entry_futures=pos.entry_futures,
                    exit_spot=cur_spot, exit_futures=cur_futures,
                    entry_dte=pos.entry_dte, exit_dte=cur_dte,
                    exit_reason=exit_reason,
                    regime_at_entry=pos.regime_at_entry,
                    regime_multiplier=pos.regime_multiplier,
                    z_size_scalar=pos.z_size_scalar,
                    notional=pos.notional,
                    gross_pnl=mtm, cost=cost, net_pnl=net,
                    pnl_pct=net / pos.notional if pos.notional > 0 else 0,
                    carry_collected_ann_pct=carry,
                ))
                positions.remove(pos)

        # ═══════════════════════════
        #  CHECK ENTRIES (scan FULL curve, not just front-month)
        # ═══════════════════════════
        if len(positions) < cfg.max_positions:
            open_contracts = {p.contract for p in positions}
            today_contracts = all_daily[all_daily["date"] == date]

            # Scan every active contract for entry opportunities
            best_candidate = None
            best_score = -np.inf

            for _, cr in today_contracts.iterrows():
                sym = cr["symbol"]
                c_ann = cr["annualized_basis_pct"]
                c_raw = cr["basis_pct"]
                c_dte = cr["days_to_expiry"]

                if pd.isna(c_ann) or pd.isna(c_raw):
                    continue
                if sym in open_contracts:
                    continue
                # Cooldown
                last_exit = cooldown_map.get(sym)
                if last_exit is not None and (date - last_exit).days < cfg.cooldown_days:
                    continue

                # Does THIS contract meet entry threshold?
                direction = None
                if c_ann > cfg.contango_entry_ann:
                    direction = "short_basis"
                elif c_ann < cfg.backwardation_entry_ann:
                    direction = "long_basis"

                if direction is None:
                    continue

                # Score: |ann_basis| × DTE preference weight
                dte_w = cfg.dte_weight(c_dte)
                score = abs(c_ann) * dte_w

                if score > best_score:
                    best_score = score
                    best_candidate = {
                        "symbol": sym,
                        "dte": c_dte,
                        "basis_pct": c_raw,
                        "ann_basis_pct": c_ann,
                        "spot_close": cr["spot_close"],
                        "futures_close": cr["futures_close"],
                        "direction": direction,
                    }

            if best_candidate is not None:
                direction = best_candidate["direction"]

                # Z-SCORE ENHANCEMENT: optional timing filter
                # Uses front-month z as general market richness gauge
                z_ok = True
                if not pd.isna(z) and cfg.z_entry_boost > 0:
                    if direction == "short_basis" and z < cfg.z_entry_boost:
                        z_ok = False
                    elif direction == "long_basis" and z > -cfg.z_entry_boost:
                        z_ok = False

                if z_ok:
                    trade_counter += 1
                    contract_regime = _classify_regime(best_candidate["ann_basis_pct"])
                    regime_mult = cfg.regime_multiplier(contract_regime)

                    # Z-score confidence scalar
                    if not pd.isna(z):
                        z_scalar = min(abs(z) / cfg.z_full_size, 1.0)
                    else:
                        z_scalar = 0.6

                    effective_pct = cfg.base_position_pct * regime_mult * z_scalar
                    notional = equity * effective_pct

                    if notional >= cfg.min_trade_usd:
                        positions.append(OpenPosition(
                            trade_id=f"A_{trade_counter:04d}",
                            contract=best_candidate["symbol"],
                            entry_date=date,
                            direction=direction,
                            entry_basis_pct=best_candidate["basis_pct"],
                            entry_ann_basis_pct=best_candidate["ann_basis_pct"],
                            entry_z=z if not pd.isna(z) else 0,
                            entry_spot=best_candidate["spot_close"],
                            entry_futures=best_candidate["futures_close"],
                            entry_dte=best_candidate["dte"],
                            regime_at_entry=contract_regime,
                            regime_multiplier=regime_mult,
                            z_size_scalar=z_scalar,
                            notional=notional,
                        ))

        # ═══════════════════════════
        #  EQUITY SNAPSHOT
        # ═══════════════════════════
        unrealized = 0.0
        for pos in positions:
            cr = all_daily[
                (all_daily["date"] == date) &
                (all_daily["symbol"] == pos.contract)
            ]
            if not cr.empty:
                unrealized += pos.mark_to_market(cr.iloc[0]["basis_pct"])

        equity_hist.append({
            "date": date, "equity": equity + unrealized,
            "n_open": len(positions), "ann_basis_pct": ann_basis,
            "regime": regime, "z_score": z,
        })

    # ═══════════════════════════
    #  CLOSE REMAINING
    # ═══════════════════════════
    if positions:
        last = trade_days.iloc[-1]
        last_date = last["date"]
        for pos in list(positions):
            days_held = (last_date - pos.entry_date).days
            cr = all_daily[
                (all_daily["date"] == last_date) &
                (all_daily["symbol"] == pos.contract)
            ]
            if not cr.empty:
                cb = cr.iloc[0]["basis_pct"]
                ca = cr.iloc[0]["annualized_basis_pct"]
                cs = cr.iloc[0]["spot_close"]
                cf = cr.iloc[0]["futures_close"]
                cd = cr.iloc[0]["days_to_expiry"]
            else:
                cb = last.get("basis_pct", pos.entry_basis_pct)
                if pd.isna(cb): cb = pos.entry_basis_pct
                ca = last.get("annualized_basis_pct", pos.entry_ann_basis_pct)
                cs = last.get("spot_close", pos.entry_spot)
                cf = cs * (1 + cb / 100)
                cd = max(pos.entry_dte - days_held, 0)

            mtm = pos.mark_to_market(cb)
            cost = pos.notional * cfg.round_trip_rate + cfg.gas_cost_usd
            net = mtm - cost
            equity += net
            carry = (pos.entry_ann_basis_pct - ca) if pos.direction == "short_basis" \
                     else (ca - pos.entry_ann_basis_pct)
            completed.append(Trade(
                trade_id=pos.trade_id, contract=pos.contract,
                entry_date=pos.entry_date, exit_date=last_date,
                days_held=days_held, direction=pos.direction,
                entry_basis_pct=pos.entry_basis_pct, exit_basis_pct=cb,
                entry_ann_basis_pct=pos.entry_ann_basis_pct, exit_ann_basis_pct=ca,
                entry_z=pos.entry_z, exit_z=last.get("z_score", 0),
                entry_spot=pos.entry_spot, entry_futures=pos.entry_futures,
                exit_spot=cs, exit_futures=cf,
                entry_dte=pos.entry_dte, exit_dte=cd,
                exit_reason="end",
                regime_at_entry=pos.regime_at_entry,
                regime_multiplier=pos.regime_multiplier,
                z_size_scalar=pos.z_size_scalar,
                notional=pos.notional, gross_pnl=mtm,
                cost=cost, net_pnl=net,
                pnl_pct=net / pos.notional if pos.notional > 0 else 0,
                carry_collected_ann_pct=carry,
            ))

    # Correct final equity snapshot to reflect force-close costs
    if equity_hist:
        equity_hist[-1]["equity"] = equity

    # ═══════════════════════════
    #  BUILD RESULTS
    # ═══════════════════════════
    eq_df = pd.DataFrame(equity_hist)
    eq_series = eq_df.set_index("date")["equity"]
    metrics = compute_metrics(eq_series, completed)
    trade_df = pd.DataFrame([t.to_dict() for t in completed]) if completed else pd.DataFrame()

    if completed:
        reasons = {}
        for t in completed:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        metrics["exit_reasons"] = reasons

    print(f"\n  Trades: {len(completed)}")
    if not eq_df.empty:
        final_eq = eq_series.iloc[-1]
        print(f"  Final equity: ${final_eq:,.0f} "
              f"({(final_eq / cfg.initial_capital - 1) * 100:+.1f}%)")
    if completed:
        print(f"  Exit reasons: "
              f"{', '.join(f'{k}={v}' for k, v in sorted(reasons.items()))}")
        avg_carry = np.mean([t.carry_collected_ann_pct for t in completed])
        print(f"  Avg carry collected: {avg_carry:.1f}% ann")

    result = BacktestResults(
        strategy="A: Calendar Spreads (Carry / Convergence)",
        equity_curve=eq_df, trade_log=trade_df,
        metrics=metrics, config=cfg.to_dict(),
        daily_data=front_daily,
    )
    result.summary()

    # auto-plot
    try:
        import matplotlib.pyplot as plt

        spot_for_plot = None
        if front_daily is not None and "spot_close" in front_daily.columns:
            spot_s = front_daily.set_index("date")["spot_close"].dropna()
            if len(spot_s) > 0:
                spot_for_plot = spot_s

        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        fig.suptitle("Strategy A: Calendar Spreads (Carry / Convergence)",
                      fontsize=14, fontweight="bold")
        result.plot_equity(ax=axes[0], benchmark_spot=spot_for_plot)
        result.plot_drawdown(ax=axes[1])
        result.plot_basis_and_trades(ax=axes[2])
        result.plot_regime_pnl(ax=axes[3])
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass

    return result


# ═══════════════════════════════════════════════════════════════
#  ANALYSIS HELPERS
# ═══════════════════════════════════════════════════════════════

def regime_performance(result: BacktestResults) -> pd.DataFrame:
    eq = result.equity_curve.copy()
    eq["return"] = eq["equity"].pct_change()
    rows = []
    for regime, grp in eq.groupby("regime"):
        rets = grp["return"].dropna()
        if len(rets) < 3:
            continue
        rows.append({
            "regime": regime,
            "n_days": len(rets),
            "pct_of_time": round(len(rets) / len(eq) * 100, 1),
            "mean_daily_ret_bps": round(rets.mean() * 10000, 2),
            "sharpe": round(rets.mean() / rets.std() * np.sqrt(365), 2)
                     if rets.std() > 0 else 0,
        })
    return pd.DataFrame(rows).sort_values("n_days", ascending=False)


def contract_usage(result: BacktestResults) -> pd.DataFrame:
    if result.trade_log.empty:
        return pd.DataFrame()
    rows = []
    for contract, grp in result.trade_log.groupby("contract"):
        pnls = grp["net_pnl"].astype(float).values
        rows.append({
            "contract": contract,
            "n_trades": len(grp),
            "total_pnl": round(pnls.sum(), 2),
            "avg_pnl": round(pnls.mean(), 2),
            "win_rate": round((pnls > 0).mean(), 3),
            "avg_days_held": round(grp["days_held"].astype(float).mean(), 1),
            "avg_entry_dte": round(grp["entry_dte"].astype(float).mean(), 0),
            "avg_carry_ann": round(grp["carry_collected_ann_pct"].astype(float).mean(), 1),
        })
    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False)


def parameter_sensitivity(
    dl, param_name: str, values: list,
    base_cfg: CalendarConfig = None,
) -> pd.DataFrame:
    if base_cfg is None:
        base_cfg = CalendarConfig()
    import io, contextlib
    # Properties to strip before reconstructing config
    computed_keys = {"round_trip_rate", "entry_cost_rate", "exit_cost_rate",
                     "gas_cost_usd", "spot_vc", "futures_vc"}
    rows = []
    for v in values:
        override = {k: v_ for k, v_ in base_cfg.to_dict().items()
                    if k not in computed_keys}
        override[param_name] = v
        c = CalendarConfig(**override)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            res = run_calendar_spreads(dl, c, test_only=True)
        m = res.metrics
        rows.append({
            param_name: v,
            "sharpe": m.get("sharpe_ratio", 0),
            "return_pct": m.get("total_return_pct", 0),
            "max_dd_pct": m.get("max_drawdown_pct", 0),
            "n_trades": m.get("n_trades", 0),
            "win_rate": m.get("win_rate", 0),
            "pnl": m.get("total_pnl", 0),
            "avg_carry": m.get("avg_carry_collected_ann_pct", 0),
        })
    return pd.DataFrame(rows)