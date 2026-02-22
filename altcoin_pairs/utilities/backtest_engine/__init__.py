"""
backtest_engine -- look-ahead-proof Kalman pairs trading backtester
"""

from .strategy import run_pairs, BacktestResults, PairsEngine
from .runner import DataFeed, RollingWindow
from .models import SpreadPosition, CompletedTrade
from .portfolio import Portfolio
from .margin import MarginConfig, MarginManager
from .kalman import KalmanBatch
from .screening import (
    screen_pairs, fast_eg_test, fast_half_life,
    test_residual_stationarity,
)
from .costs import get_venue_costs, VenueCosts, VENUE_COSTS
from .metrics import compute_all_metrics, print_summary
from .reporting import (
    generate_reports, print_trade_log,
    plot_rolling_sharpe, plot_equity_vs_btc, plot_pnl_by_reason,
)
from .volatility import (
    EWMAVolTracker, vol_adjusted_notionals,
    implied_correlation, regime_confidence,
)
