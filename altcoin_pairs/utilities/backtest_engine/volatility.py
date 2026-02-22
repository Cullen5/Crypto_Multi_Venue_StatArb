"""
volatility.py — vectorized EWMA volatility tracker, vol-adjusted sizing,
                 implied correlation regime filter

Provides:
  1. EWMAVolTracker         — per-asset 7-day EWMA vol, one vectorized call per bar
  2. vol_adjusted_notionals — variance-neutral leg sizing
  3. implied_correlation    — decompose Kalman beta into ρ × (σ_A / σ_B)
  4. regime_confidence      — smooth scalar from implied ρ for position sizing

The key insight for (3):
    β = ρ × (σ_A / σ_B)
    ∴ ρ_implied = β × (σ_B / σ_A)

This decomposes the Kalman's hedge ratio into *why* it is what it is.
When ρ is high (>0.7), the pair relationship is well-behaved and driven
by genuine co-movement. When ρ drops (<0.5), beta is being explained by
vol differences rather than correlation — the pair is degrading.
"""

import numpy as np


# =========================================================================
#  EWMA VOLATILITY TRACKER (fully vectorized)
# =========================================================================

class EWMAVolTracker:
    """
    Exponentially weighted moving average volatility for all assets
    simultaneously.  One .update() call per bar — no loops over assets.

    Parameters
    ----------
    n_assets : int
        Number of assets in the price matrix.
    halflife_bars : float
        EWMA half-life in bars.  For 15-min bars and 7-day vol:
            7 * 24 * 4 = 672 bars.
    min_history : int
        Minimum bars before vol estimates are valid.
        Assets with fewer observations return NaN.
    annualize_factor : float
        Multiply raw per-bar vol by this for annualized vol.
        Default 0 = return per-bar vol (best for sizing).
    """

    def __init__(
        self,
        n_assets: int,
        halflife_bars: float = 672.0,
        min_history: int = 96,
        annualize_factor: float = 0.0,
    ):
        self.n_assets = n_assets
        self.min_history = min_history
        self.annualize_factor = annualize_factor

        # EWMA decay:  alpha = 1 - exp(-ln2 / halflife)
        self.alpha = 1.0 - np.exp(-np.log(2.0) / halflife_bars)

        # state arrays — all vectorized operations
        self._prev_prices = np.full(n_assets, np.nan)
        self._ewma_var = np.zeros(n_assets)
        self._ewma_mean = np.zeros(n_assets)
        self._count = np.zeros(n_assets, dtype=np.int64)
        self._warmed_up = False

    def update(self, prices: np.ndarray) -> None:
        """
        Ingest one bar of prices for all assets.  Fully vectorized —
        no Python loops over assets.

        After warmup (all assets seen at least one bar), takes a fast
        path that skips first-time branching.
        """
        prev = self._prev_prices
        alpha = self.alpha

        valid = (
            np.isfinite(prices)
            & np.isfinite(prev)
            & (prices > 0)
            & (prev > 0)
        )

        # fast path: after warmup, no asset is "first" anymore
        if self._warmed_up:
            safe_prev = np.where(valid, prev, 1.0)
            safe_curr = np.where(valid, prices, 1.0)
            log_ret = np.where(valid, np.log(safe_curr / safe_prev), 0.0)

            self._ewma_mean[valid] = (
                alpha * log_ret[valid]
                + (1.0 - alpha) * self._ewma_mean[valid]
            )
            deviation = log_ret[valid] - self._ewma_mean[valid]
            self._ewma_var[valid] = (
                alpha * deviation ** 2
                + (1.0 - alpha) * self._ewma_var[valid]
            )
            self._count[valid] += 1
            has_price = np.isfinite(prices) & (prices > 0)
            self._prev_prices[has_price] = prices[has_price]
            return

        # slow path: handles first-time initialization
        safe_prev = np.where(valid, prev, 1.0)
        safe_curr = np.where(valid, prices, 1.0)
        log_ret = np.where(valid, np.log(safe_curr / safe_prev), 0.0)

        first = valid & (self._count == 0)
        if np.any(first):
            self._ewma_mean[first] = log_ret[first]
            self._ewma_var[first] = log_ret[first] ** 2

        cont = valid & (self._count > 0)
        if np.any(cont):
            self._ewma_mean[cont] = (
                alpha * log_ret[cont]
                + (1.0 - alpha) * self._ewma_mean[cont]
            )
            deviation = log_ret[cont] - self._ewma_mean[cont]
            self._ewma_var[cont] = (
                alpha * deviation ** 2
                + (1.0 - alpha) * self._ewma_var[cont]
            )

        self._count[valid] += 1
        has_price = np.isfinite(prices) & (prices > 0)
        self._prev_prices[has_price] = prices[has_price]

        # check if all assets have been seen
        if np.all(self._count > 0):
            self._warmed_up = True

    def get_vol(self, asset_idx: int) -> float:
        """Per-bar vol (std of log-returns). NaN if insufficient history."""
        if self._count[asset_idx] < self.min_history:
            return np.nan
        v = np.sqrt(max(self._ewma_var[asset_idx], 0.0))
        if self.annualize_factor > 0:
            v *= self.annualize_factor
        return v

    def get_vol_array(self) -> np.ndarray:
        """Per-bar vol for all assets. Shape (n_assets,). NaN if not ready."""
        vol = np.sqrt(np.maximum(self._ewma_var, 0.0))
        vol[self._count < self.min_history] = np.nan
        if self.annualize_factor > 0:
            vol *= self.annualize_factor
        return vol

    def is_ready(self, asset_idx: int) -> bool:
        return self._count[asset_idx] >= self.min_history

    def warmup_bulk(self, price_matrix: np.ndarray) -> None:
        """Warmup from (T, n_assets) matrix. Each row is vectorized."""
        for t in range(price_matrix.shape[0]):
            self.update(price_matrix[t])


# =========================================================================
#  VOL-ADJUSTED NOTIONAL ALLOCATION
# =========================================================================

def vol_adjusted_notionals(
    total_notional: float,
    beta: float,
    vol_a: float,
    vol_b: float,
    max_imbalance: float = 5.0,
) -> tuple[float, float]:
    """
    Split total_notional so each leg contributes equal variance.

    Falls back to simple beta-weighted split if vols are NaN.
    Returns (notional_a, notional_b) — both positive.
    """
    abs_beta = abs(beta)
    if abs_beta < 1e-10:
        return total_notional, 0.0

    if not (np.isfinite(vol_a) and np.isfinite(vol_b)
            and vol_a > 1e-12 and vol_b > 1e-12):
        n_a = total_notional / (1.0 + abs_beta)
        return n_a, abs_beta * n_a

    n_b = total_notional / (abs_beta * (1.0 + vol_b / vol_a))
    n_a = total_notional - abs_beta * n_b

    leg_b_dollar = abs_beta * n_b
    if n_a > 0 and leg_b_dollar > 0:
        ratio = max(n_a, leg_b_dollar) / min(n_a, leg_b_dollar)
        if ratio > max_imbalance:
            n_a = total_notional / (1.0 + abs_beta)
            n_b = abs_beta * n_a

    return max(n_a, 0.0), max(n_b, 0.0)


# =========================================================================
#  IMPLIED CORRELATION & REGIME FILTER
# =========================================================================

def implied_correlation(
    beta: float,
    vol_a: float,
    vol_b: float,
) -> float:
    """
    Decompose Kalman beta into implied correlation.

        β = ρ × (σ_A / σ_B)
        ∴ ρ = β × (σ_B / σ_A)

    Returns the RAW implied correlation (can exceed ±1.0).
    Values outside [-1, 1] indicate model stress: the Kalman beta
    and EWMA vol estimates are disagreeing. This is diagnostic
    information — regime_confidence() handles interpretation.

    Typical ranges for healthy pairs:
      ρ ∈ [0.3, 0.85]  — sweet spot for mean-reversion
      ρ ∈ [0.85, 1.0]  — very tight pair, small spread
      ρ > 1.0           — model stress (beta/vol mismatch)
      ρ < 0             — anti-correlated, spread trends

    Returns NaN if vols unavailable.
    """
    if not (np.isfinite(vol_a) and np.isfinite(vol_b)
            and vol_a > 1e-12 and vol_b > 1e-12):
        return np.nan

    rho = beta * (vol_b / vol_a)
    if not np.isfinite(rho):
        return np.nan
    return float(rho)


def regime_confidence(
    implied_rho: float,
    rho_high: float = 0.65,
    rho_low: float = 0.25,
    rho_stress: float = 1.2,
) -> float:
    """
    Smooth confidence scalar from implied correlation for position sizing.

    implied_rho is RAW (not clipped/transformed), so can exceed ±1.0.

    Shape:
        ρ ≤ 0                 → 0.0  (anti-correlated, hard reject)
        ρ ∈ (0, rho_low]      → 0.0  (too weak)
        ρ ∈ (rho_low, rho_high) → linear ramp 0→1
        ρ ∈ [rho_high, 1.0]   → 1.0  (sweet spot)
        ρ ∈ (1.0, rho_stress)  → linear ramp 1→0 (model stress)
        ρ ≥ rho_stress         → 0.0  (severe stress, reject)

    The penalty above 1.0 catches beta/vol disagreement: the Kalman
    beta implies a stronger relationship than EWMA vols can explain.
    """
    if not np.isfinite(implied_rho):
        return 0.5

    # negative rho = anti-correlated pair = not mean-reverting
    if implied_rho <= 0:
        return 0.0

    # stress penalty: rho > 1.0 means model components disagree
    if implied_rho >= rho_stress:
        return 0.0
    if implied_rho > 1.0:
        return 1.0 - (implied_rho - 1.0) / (rho_stress - 1.0)

    # normal ramp
    if implied_rho >= rho_high:
        return 1.0
    if implied_rho <= rho_low:
        return 0.0

    return (implied_rho - rho_low) / (rho_high - rho_low)
