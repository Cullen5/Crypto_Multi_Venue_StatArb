"""
kalman.py — batch Kalman filter for pairs spread trading

Features:
  - Numba-jitted batch update (all spreads in one call per bar)
  - Per-pair MLE delta fitting
  - Rolling z-score on spread levels via circular buffer
  - Python fallbacks when numba unavailable

State model:
  price_a[t] = alpha[t] + beta[t] * price_b[t] + noise
  alpha, beta follow random walks with process noise Q = delta * I
"""

import numpy as np
from scipy.optimize import minimize_scalar

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# =========================================================================
#  NUMBA KERNELS
# =========================================================================

if _HAS_NUMBA:
    @njit(cache=True)
    def _kalman_warmup_numba(prices_a, prices_b, delta, R,
                              x0, x1, P00, P01, P11):
        """Run Kalman on history, return final state + spread array."""
        n = len(prices_a)
        spreads = np.empty(n)
        for t in range(n):
            P00 += delta; P11 += delta
            h = prices_b[t]; y = prices_a[t]
            e = y - x0 - x1 * h
            S = P00 + 2.0 * h * P01 + h * h * P11 + R
            if S > 1e-12:
                inv_S = 1.0 / S
                K0 = (P00 + P01 * h) * inv_S
                K1 = (P01 + P11 * h) * inv_S
                x0 += K0 * e; x1 += K1 * e
                P01_old = P01
                P00 = (1.0 - K0) * P00 - K0 * h * P01_old
                P01 = (1.0 - K0) * P01_old - K0 * h * P11
                P11 = -K1 * P01_old + (1.0 - K1 * h) * P11
            spreads[t] = y - x0 - x1 * h
        return x0, x1, P00, P01, P11, spreads

    @njit(cache=True)
    def _batch_update_numba(n_active, state, idx_a, idx_b,
                             prices_row, deltas, R):
        """
        Update ALL active spreads in one jitted call.

        state: (n_active, 5) — [x0, x1, P00, P01, P11]
        Returns (spread_values, innovation_variances).
        """
        spreads = np.empty(n_active)
        s_values = np.empty(n_active)
        for i in range(n_active):
            delta = deltas[i]
            pa = prices_row[idx_a[i]]; pb = prices_row[idx_b[i]]
            x0 = state[i, 0]; x1 = state[i, 1]
            P00 = state[i, 2]; P01 = state[i, 3]; P11 = state[i, 4]
            P00 += delta; P11 += delta
            e = pa - x0 - x1 * pb
            S = P00 + 2.0 * pb * P01 + pb * pb * P11 + R
            if S > 1e-12:
                inv_S = 1.0 / S
                K0 = (P00 + P01 * pb) * inv_S
                K1 = (P01 + P11 * pb) * inv_S
                x0 += K0 * e; x1 += K1 * e
                P01_old = P01
                P00 = (1.0 - K0) * P00 - K0 * pb * P01_old
                P01 = (1.0 - K0) * P01_old - K0 * pb * P11
                P11 = -K1 * P01_old + (1.0 - K1 * pb) * P11
            state[i, 0] = x0; state[i, 1] = x1
            state[i, 2] = P00; state[i, 3] = P01; state[i, 4] = P11
            spreads[i] = pa - x0 - x1 * pb
            s_values[i] = S
        return spreads, s_values

    @njit(cache=True)
    def _nll_pairs_numba(log_delta, prices_a, prices_b, R):
        """Negative log-likelihood for MLE delta fitting."""
        delta = np.exp(log_delta)
        n = len(prices_a)
        x0, x1 = 0.0, 1.0
        P00, P01, P11 = 1.0, 0.0, 1.0
        ll = 0.0
        for t in range(n):
            P00 += delta; P11 += delta
            h = prices_b[t]; y = prices_a[t]
            e = y - x0 - x1 * h
            S = P00 + 2.0 * h * P01 + h * h * P11 + R
            if S > 1e-12:
                ll -= 0.5 * (np.log(S) + e * e / S)
                inv_S = 1.0 / S
                K0 = (P00 + P01 * h) * inv_S
                K1 = (P01 + P11 * h) * inv_S
                x0 += K0 * e; x1 += K1 * e
                P01_old = P01
                P00 = (1.0 - K0) * P00 - K0 * h * P01_old
                P01 = (1.0 - K0) * P01_old - K0 * h * P11
                P11 = -K1 * P01_old + (1.0 - K1 * h) * P11
        return -ll


# =========================================================================
#  PYTHON FALLBACKS
# =========================================================================

def _kalman_warmup_py(prices_a, prices_b, delta, R, x0, x1, P00, P01, P11):
    n = len(prices_a)
    spreads = np.empty(n)
    for t in range(n):
        P00 += delta; P11 += delta
        h = prices_b[t]; y = prices_a[t]
        e = y - x0 - x1 * h
        S = P00 + 2.0 * h * P01 + h * h * P11 + R
        if S > 1e-12:
            inv_S = 1.0 / S
            K0 = (P00 + P01 * h) * inv_S
            K1 = (P01 + P11 * h) * inv_S
            x0 += K0 * e; x1 += K1 * e
            P01_old = P01
            P00 = (1.0 - K0) * P00 - K0 * h * P01_old
            P01 = (1.0 - K0) * P01_old - K0 * h * P11
            P11 = -K1 * P01_old + (1.0 - K1 * h) * P11
        spreads[t] = y - x0 - x1 * h
    return x0, x1, P00, P01, P11, spreads


def _batch_update_py(n_active, state, idx_a, idx_b, prices_row, deltas, R):
    spreads = np.empty(n_active)
    s_values = np.empty(n_active)
    for i in range(n_active):
        delta = deltas[i]
        pa = prices_row[idx_a[i]]; pb = prices_row[idx_b[i]]
        x0 = state[i, 0]; x1 = state[i, 1]
        P00 = state[i, 2]; P01 = state[i, 3]; P11 = state[i, 4]
        P00 += delta; P11 += delta
        e = pa - x0 - x1 * pb
        S = P00 + 2.0 * pb * P01 + pb * pb * P11 + R
        if S > 1e-12:
            inv_S = 1.0 / S
            K0 = (P00 + P01 * pb) * inv_S
            K1 = (P01 + P11 * pb) * inv_S
            x0 += K0 * e; x1 += K1 * e
            P01_old = P01
            P00 = (1.0 - K0) * P00 - K0 * pb * P01_old
            P01 = (1.0 - K0) * P01_old - K0 * pb * P11
            P11 = -K1 * P01_old + (1.0 - K1 * pb) * P11
        state[i, 0] = x0; state[i, 1] = x1
        state[i, 2] = P00; state[i, 3] = P01; state[i, 4] = P11
        spreads[i] = pa - x0 - x1 * pb
        s_values[i] = S
    return spreads, s_values


def _nll_pairs_py(log_delta, prices_a, prices_b, R):
    delta = np.exp(log_delta)
    n = len(prices_a)
    x0, x1 = 0.0, 1.0
    P00, P01, P11 = 1.0, 0.0, 1.0
    ll = 0.0
    for t in range(n):
        P00 += delta; P11 += delta
        h = prices_b[t]; y = prices_a[t]
        e = y - x0 - x1 * h
        S = P00 + 2.0 * h * P01 + h * h * P11 + R
        if S > 1e-12:
            ll -= 0.5 * (np.log(S) + e * e / S)
            inv_S = 1.0 / S
            K0 = (P00 + P01 * h) * inv_S
            K1 = (P01 + P11 * h) * inv_S
            x0 += K0 * e; x1 += K1 * e
            P01_old = P01
            P00 = (1.0 - K0) * P00 - K0 * h * P01_old
            P01 = (1.0 - K0) * P01_old - K0 * h * P11
            P11 = -K1 * P01_old + (1.0 - K1 * h) * P11
    return -ll


# dispatch
_warmup = _kalman_warmup_numba if _HAS_NUMBA else _kalman_warmup_py
_batch = _batch_update_numba if _HAS_NUMBA else _batch_update_py
_nll = _nll_pairs_numba if _HAS_NUMBA else _nll_pairs_py


# =========================================================================
#  MLE DELTA FITTING
# =========================================================================

def fit_delta_mle(prices_a, prices_b, R=0.1,
                  delta_bounds=(1e-6, 1e-1), subsample=4):
    """
    Fit Kalman process noise (delta) via maximum likelihood.

    Returns (delta, at_boundary).
    """
    if len(prices_a) < 100:
        return 1e-4, True

    pa = np.ascontiguousarray(prices_a[::subsample], dtype=np.float64)
    pb = np.ascontiguousarray(prices_b[::subsample], dtype=np.float64)
    fit_bounds = (delta_bounds[0] * subsample, delta_bounds[1] * subsample)

    try:
        result = minimize_scalar(
            _nll,
            bounds=(np.log(fit_bounds[0]), np.log(fit_bounds[1])),
            method="bounded",
            args=(pa, pb, R),
        )
        fitted = np.exp(result.x) / subsample

        eps = 0.05
        at_boundary = (
            abs(fitted - delta_bounds[0]) / delta_bounds[0] < eps
            or abs(fitted - delta_bounds[1]) / delta_bounds[1] < eps
        )
        return fitted, at_boundary
    except Exception:
        return 1e-4, True


# =========================================================================
#  BATCH KALMAN MANAGER
# =========================================================================

class KalmanBatch:
    """
    Manages Kalman state for all active spreads. One batch call per bar.

    Per-pair state: [alpha, beta, P00, P01, P11] (5 floats)
    Rolling z-score via circular buffer on spread levels, using
    mean / std normalization.

    Usage:
        kb = KalmanBatch(R=0.1, z_window=960)
        kb.add("SOL__AVAX", col_a, col_b, prices_a, prices_b, init_beta)
        signals = kb.update(active_ids, price_row)
        # signals = {pair_id: (spread, z_score, spread_std, beta_var_P11)}
    """

    def __init__(self, default_delta=1e-4, R=0.1,
                 z_window=960, z_min_bars=50, use_mle=True,
                 z_hl_mult: float = 10.0,
                 z_peak_decay: float = 0.999):
        """
        Parameters
        ----------
        z_hl_mult : float
            EWMA z-score span = z_hl_mult × half_life per pair.
            Controls how adaptive the z-score tracks each pair's
            local mean. Short HL pairs get fast-tracking z-scores
            that don't drift; long HL pairs get smoother z-scores.
            Default 10: HL=20 → span=200, HL=100 → span=1000.
            Set to 0 to disable EWMA (uses fixed-window buffer).
        z_peak_decay : float
            Per-bar decay for the exponential peak tracker.
            0.999 → half-life ≈ 693 bars (~7 days at 15min).
            Recent peaks dominate, old peaks fade.
        """
        self.default_delta = default_delta
        self.R = R
        self.z_window = z_window
        self.z_min = z_min_bars
        self.use_mle = use_mle
        self.z_hl_mult = z_hl_mult
        self.z_peak_decay = z_peak_decay

        # per-pair state
        self._state: dict[str, np.ndarray] = {}      # [x0, x1, P00, P01, P11]
        self._cols: dict[str, tuple[int, int]] = {}   # (col_a, col_b)
        self._deltas: dict[str, float] = {}
        self._buf: dict[str, np.ndarray] = {}         # circular buffer (also for coint checks)
        self._buf_pos: dict[str, int] = {}
        self._buf_cnt: dict[str, int] = {}
        self._buf_sum: dict[str, float] = {}          # running sum for O(1) buffer mean
        self._buf_sq: dict[str, float] = {}            # running sum of squares

        # per-pair EWMA z-score state (adaptive window per pair)
        self._ewma_alpha: dict[str, float] = {}       # decay rate = 2/(span+1)
        self._ewma_mean: dict[str, float] = {}         # exponential moving mean
        self._ewma_var: dict[str, float] = {}           # exponential moving variance
        self._half_lives: dict[str, float] = {}         # for reporting

        # per-pair decaying peak |z| tracker
        self._z_peak: dict[str, float] = {}             # max(decay*peak, |z|)

        # batch arrays (rebuilt when dirty)
        self._dirty = True
        self._b_state = None
        self._b_ia = None
        self._b_ib = None
        self._b_sids: list[str] = []
        self._b_deltas = None

    def add(self, sid: str, col_a: int, col_b: int,
            prices_a: np.ndarray, prices_b: np.ndarray,
            init_beta: float, warmup_bars: int = 500,
            half_life: float = 100.0):
        """
        Add or re-register a spread.

        New spreads: fits delta, runs warmup, seeds z-score buffer + EWMA.
        Existing spreads: preserves Kalman state, z-buffer, AND EWMA state.
        Only re-fits delta (MLE) and updates half_life → EWMA alpha.
        """
        is_new = sid not in self._state

        # fit per-pair delta
        if self.use_mle:
            delta, _ = fit_delta_mle(prices_a, prices_b, self.R)
        else:
            delta = self.default_delta
        self._deltas[sid] = delta

        # always update half_life and EWMA alpha (half-life can change at refit)
        self._half_lives[sid] = half_life
        if self.z_hl_mult > 0:
            span = max(self.z_min, min(self.z_window,
                                        self.z_hl_mult * half_life))
            self._ewma_alpha[sid] = 2.0 / (span + 1.0)
        else:
            self._ewma_alpha[sid] = 2.0 / (self.z_window + 1.0)

        if is_new:
            n = min(warmup_bars, len(prices_a))
            # warmup on the MOST RECENT n bars so the Kalman state
            # reflects current conditions, not data from weeks ago
            pa = np.ascontiguousarray(prices_a[-n:], dtype=np.float64)
            pb = np.ascontiguousarray(prices_b[-n:], dtype=np.float64)
            x0, x1, P00, P01, P11, _ = _warmup(
                pa, pb, delta, self.R, 0.0, init_beta, 1.0, 0.0, 1.0)
            self._state[sid] = np.array([x0, x1, P00, P01, P11])

        # column mapping always updated (could change if feed rebuilt)
        self._cols[sid] = (col_a, col_b)

        # z-buffer + EWMA: only seed for NEW pairs
        # existing pairs keep their running state to avoid z-score
        # discontinuities at refit boundaries
        if is_new:
            st = self._state[sid]
            alpha_k, beta_k = st[0], st[1]
            formation_spread = prices_a - alpha_k - beta_k * prices_b

            # circular buffer (also used for coint-break ADF checks)
            buf = np.zeros(self.z_window)
            n_seed = min(len(formation_spread), self.z_window)
            buf[:n_seed] = formation_spread[-n_seed:]
            self._buf[sid] = buf
            self._buf_pos[sid] = n_seed % self.z_window
            self._buf_cnt[sid] = n_seed
            # seed running stats from buffer
            seeded = buf[:n_seed]
            self._buf_sum[sid] = float(np.sum(seeded))
            self._buf_sq[sid] = float(np.dot(seeded, seeded))

            # seed EWMA from formation data (process in order)
            ewma_a = self._ewma_alpha[sid]
            em = float(seeded[0]) if n_seed > 0 else 0.0
            ev = 0.0
            peak = 1.0  # floor: assume at least ±1σ is reachable
            for k in range(1, n_seed):
                diff = seeded[k] - em
                em += ewma_a * diff
                ev = (1.0 - ewma_a) * (ev + ewma_a * diff * diff)
                # track peak |z| from formation data
                s_k = np.sqrt(max(ev, 0.0))
                if s_k > 1e-10:
                    z_k = abs((seeded[k] - em) / s_k)
                    peak = max(self.z_peak_decay * peak, z_k)
            self._ewma_mean[sid] = em
            self._ewma_var[sid] = ev
            self._z_peak[sid] = max(peak, 1.0)

        self._dirty = True

    def remove(self, sid: str):
        for d in (self._state, self._cols, self._deltas,
                  self._buf, self._buf_pos, self._buf_cnt,
                  self._buf_sum, self._buf_sq,
                  self._ewma_alpha, self._ewma_mean, self._ewma_var,
                  self._half_lives, self._z_peak):
            d.pop(sid, None)
        self._dirty = True

    def _rebuild(self, active_sids):
        """Rebuild contiguous batch arrays for numba."""
        sids = [s for s in active_sids if s in self._state]
        n = len(sids)
        st = np.empty((max(n, 1), 5))
        ia = np.empty(max(n, 1), dtype=np.int64)
        ib = np.empty(max(n, 1), dtype=np.int64)
        deltas = np.empty(max(n, 1), dtype=np.float64)
        for i, s in enumerate(sids):
            st[i] = self._state[s]
            ia[i], ib[i] = self._cols[s]
            deltas[i] = self._deltas.get(s, self.default_delta)
        self._b_state = st
        self._b_ia = ia
        self._b_ib = ib
        self._b_sids = sids
        self._b_deltas = deltas
        self._dirty = False

    def update(self, active_sids, row: np.ndarray) -> dict:
        """
        Batch Kalman update for all active spreads.

        Returns {pair_id: (spread_value, z_score, spread_std, beta_var_P11)}.

        Z-scores use incremental running stats on the circular buffer:
        O(1) per pair per bar instead of O(z_window).
        """
        if self._dirty:
            self._rebuild(active_sids)
        n = len(self._b_sids)
        if n == 0:
            return {}

        # save state for NaN pairs before batch update
        nan_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            ca, cb = self._b_ia[i], self._b_ib[i]
            if np.isnan(row[ca]) or np.isnan(row[cb]):
                nan_mask[i] = True
        saved_state = self._b_state[nan_mask].copy() if nan_mask.any() else None

        spreads, s_vals = _batch(
            n, self._b_state, self._b_ia, self._b_ib,
            row, self._b_deltas, self.R)

        if saved_state is not None:
            self._b_state[nan_mask] = saved_state

        out = {}
        for i in range(n):
            sid = self._b_sids[i]
            if nan_mask[i]:
                continue

            sv = spreads[i]
            if not np.isfinite(sv):
                continue

            # write back state
            self._state[sid] = self._b_state[i].copy()
            P11 = float(self._b_state[i, 4])

            # --- circular buffer (for coint-break ADF checks) ---
            buf = self._buf[sid]
            pos = self._buf_pos[sid]
            cnt = self._buf_cnt[sid]

            if cnt >= self.z_window:
                old_val = buf[pos]
                self._buf_sum[sid] -= old_val
                self._buf_sq[sid] -= old_val * old_val

            buf[pos] = sv
            self._buf_pos[sid] = (pos + 1) % self.z_window
            self._buf_cnt[sid] = cnt + 1
            cnt += 1

            self._buf_sum[sid] += sv
            self._buf_sq[sid] += sv * sv

            if cnt < self.z_min:
                out[sid] = (sv, 0.0, 0.0, P11)
                continue

            # --- EWMA z-score (per-pair adaptive window) ---
            ea = self._ewma_alpha.get(sid, 2.0 / (self.z_window + 1.0))
            em = self._ewma_mean.get(sid, sv)
            ev = self._ewma_var.get(sid, 0.0)

            diff = sv - em
            em += ea * diff
            ev = (1.0 - ea) * (ev + ea * diff * diff)

            self._ewma_mean[sid] = em
            self._ewma_var[sid] = ev

            s = np.sqrt(max(ev, 0.0))
            z = (sv - em) / s if s > 1e-10 else 0.0
            z = z if np.isfinite(z) else 0.0

            # update decaying peak tracker
            old_peak = self._z_peak.get(sid, 1.0)
            self._z_peak[sid] = max(self.z_peak_decay * old_peak, abs(z))

            out[sid] = (sv, z, s, P11)
        return out

    def get_beta(self, sid: str) -> float:
        s = self._state.get(sid)
        return float(s[1]) if s is not None else 0.0

    def get_P11(self, sid: str) -> float:
        """Posterior variance of beta (hedge ratio uncertainty)."""
        s = self._state.get(sid)
        return float(s[4]) if s is not None else 1.0

    def get_z_peak(self, sid: str) -> float:
        """Decaying peak |z| for this pair. Used for adaptive entry threshold."""
        return self._z_peak.get(sid, 1.0)

    def get_cols(self, sid: str) -> tuple[int, int] | None:
        return self._cols.get(sid)
