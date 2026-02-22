"""
costs.py — venue-aware transaction cost model

Supports CEX, hybrid (on-chain orderbooks), and DEX venues.
DEX costs vary significantly by chain (gas) and pool (swap fee tier).

Numbers from case study specs and current mainnet conditions.
Never hardcode fees in strategy code — always go through get_venue_costs().

Usage:
    vc = get_venue_costs("binance")          # CEX
    vc = get_venue_costs("uniswap_v3_arb")   # DEX on Arbitrum (cheap gas)
    vc = get_venue_costs("uniswap_v3_eth")   # DEX on Ethereum (expensive gas)
"""

from dataclasses import dataclass
from enum import Enum


class VenueType(Enum):
    CEX = "CEX"
    DEX = "DEX"
    HYBRID = "HYBRID"


@dataclass
class VenueCosts:
    name: str
    venue_type: VenueType
    maker_fee: float      # decimal, e.g. 0.0001 = 0.01%
    taker_fee: float
    avg_slippage: float   # decimal, estimated per trade
    gas_cost_usd: float   # per trade, 0 for CEX
    min_trade_usd: float  # minimum trade size to justify costs

    @property
    def round_trip_cost(self) -> float:
        """Total cost as a fraction of notional (excludes gas)."""
        return self.maker_fee + self.taker_fee + 2 * self.avg_slippage

    def all_in_cost(self, notional: float) -> float:
        """
        Total round-trip cost in USD including gas.

        For CEX: gas = 0, so this is just proportional costs.
        For DEX: gas is a fixed cost that matters at small sizes.
        A $1000 DEX trade with $2 gas adds 20bps; a $50k trade adds 0.4bps.
        """
        proportional = notional * self.round_trip_cost
        fixed = self.gas_cost_usd * 4  # 4 swaps per round trip (2 legs × entry + exit)
        return proportional + fixed

    def all_in_rate(self, notional: float) -> float:
        """All-in round-trip cost as a fraction of notional."""
        if notional <= 0:
            return float("inf")
        return self.all_in_cost(notional) / notional


# -----------------------------------------------------------------------
#  CEX venues
# -----------------------------------------------------------------------

_CEX_VENUES = {
    # Binance futures (used in original backtest)
    "binance": VenueCosts(
        "binance", VenueType.CEX,
        maker_fee=0.0004, taker_fee=0.0006,
        avg_slippage=0.0001, gas_cost_usd=0.0, min_trade_usd=100),

    # Binance spot (higher fees than futures)
    "binance_spot": VenueCosts(
        "binance_spot", VenueType.CEX,
        maker_fee=0.0004, taker_fee=0.0006,
        avg_slippage=0.0001, gas_cost_usd=0.0, min_trade_usd=100),

    "bybit": VenueCosts(
        "bybit", VenueType.CEX,
        maker_fee=0.0001, taker_fee=0.0006,
        avg_slippage=0.0003, gas_cost_usd=0.0, min_trade_usd=100),

    "okx": VenueCosts(
        "okx", VenueType.CEX,
        maker_fee=0.0002, taker_fee=0.0005,
        avg_slippage=0.0003, gas_cost_usd=0.0, min_trade_usd=100),

    "deribit": VenueCosts(
        "deribit", VenueType.CEX,
        maker_fee=0.0002, taker_fee=0.0005,
        avg_slippage=0.0003, gas_cost_usd=0.0, min_trade_usd=500),

    "coinbase": VenueCosts(
        "coinbase", VenueType.CEX,
        maker_fee=0.0004, taker_fee=0.0006,
        avg_slippage=0.0003, gas_cost_usd=0.0, min_trade_usd=100),
}

# -----------------------------------------------------------------------
#  Hybrid venues (on-chain settlement, orderbook matching)
# -----------------------------------------------------------------------

_HYBRID_VENUES = {
    "hyperliquid": VenueCosts(
        "hyperliquid", VenueType.HYBRID,
        maker_fee=0.0000, taker_fee=0.00025,
        avg_slippage=0.0003, gas_cost_usd=0.50, min_trade_usd=500),

    "dydx": VenueCosts(
        "dydx", VenueType.HYBRID,
        maker_fee=0.0000, taker_fee=0.0005,
        avg_slippage=0.0005, gas_cost_usd=0.10, min_trade_usd=500),

    "vertex": VenueCosts(
        "vertex", VenueType.HYBRID,
        maker_fee=0.0000, taker_fee=0.0002,
        avg_slippage=0.0004, gas_cost_usd=0.30, min_trade_usd=500),
}

# -----------------------------------------------------------------------
#  DEX venues — chain-specific gas costs
#
#  Gas costs are the critical differentiator for DEX pairs trading.
#  A round-trip pair trade = 4 swaps (buy A, sell B, close A, close B).
#  On Ethereum mainnet: 4 × $10-50 = $40-200 fixed cost.
#  On Arbitrum: 4 × $0.50-2 = $2-8 fixed cost.
#  On Solana: 4 × $0.01-0.05 = $0.04-0.20 fixed cost.
#
#  This means Ethereum DEX pairs need ~5-10× wider spreads than
#  Arbitrum pairs to pass the cost gate, and Solana pairs are
#  nearly as cheap as CEX.
# -----------------------------------------------------------------------

_DEX_VENUES = {
    # --- Uniswap V3 by chain ---
    # 30bps fee tier (most altcoin pools). 5bps and 1bps tiers exist
    # for stablecoin and major pairs but those aren't pairs-tradeable.

    "uniswap_v3_eth": VenueCosts(
        "uniswap_v3_eth", VenueType.DEX,
        maker_fee=0.003, taker_fee=0.003,  # AMM: no maker/taker distinction
        avg_slippage=0.003,                  # wider than CEX due to AMM curves
        gas_cost_usd=25.0,                   # Ethereum mainnet: ~$25 per swap avg
        min_trade_usd=10000),                # need $10k+ to justify gas

    "uniswap_v3_arb": VenueCosts(
        "uniswap_v3_arb", VenueType.DEX,
        maker_fee=0.003, taker_fee=0.003,
        avg_slippage=0.002,                  # slightly tighter on L2
        gas_cost_usd=1.00,                   # Arbitrum: ~$1 per swap
        min_trade_usd=2000),

    "uniswap_v3_base": VenueCosts(
        "uniswap_v3_base", VenueType.DEX,
        maker_fee=0.003, taker_fee=0.003,
        avg_slippage=0.002,
        gas_cost_usd=0.50,                   # Base: ~$0.50 per swap
        min_trade_usd=1000),

    "uniswap_v3_op": VenueCosts(
        "uniswap_v3_op", VenueType.DEX,
        maker_fee=0.003, taker_fee=0.003,
        avg_slippage=0.002,
        gas_cost_usd=0.80,                   # Optimism: ~$0.80 per swap
        min_trade_usd=1500),

    # --- Other DEXes ---

    "curve": VenueCosts(
        "curve", VenueType.DEX,
        maker_fee=0.0004, taker_fee=0.0004,  # Curve pools: 4bps typical
        avg_slippage=0.001,                    # tight for stableswap
        gas_cost_usd=20.0,                     # Ethereum mainnet
        min_trade_usd=10000),

    "sushiswap_arb": VenueCosts(
        "sushiswap_arb", VenueType.DEX,
        maker_fee=0.003, taker_fee=0.003,
        avg_slippage=0.003,
        gas_cost_usd=1.00,
        min_trade_usd=2000),

    "camelot": VenueCosts(
        "camelot", VenueType.DEX,
        maker_fee=0.003, taker_fee=0.003,
        avg_slippage=0.003,
        gas_cost_usd=1.00,                    # Arbitrum native DEX
        min_trade_usd=2000),

    "gmx": VenueCosts(
        "gmx", VenueType.DEX,
        maker_fee=0.001, taker_fee=0.001,     # GMX: 10bps swap
        avg_slippage=0.003,
        gas_cost_usd=1.50,                    # Arbitrum
        min_trade_usd=5000),

    # --- Solana DEXes (cheapest gas) ---

    "jupiter": VenueCosts(
        "jupiter", VenueType.DEX,
        maker_fee=0.003, taker_fee=0.003,     # aggregator, varies by route
        avg_slippage=0.002,
        gas_cost_usd=0.02,                    # Solana: ~$0.02 per tx
        min_trade_usd=500),

    "raydium": VenueCosts(
        "raydium", VenueType.DEX,
        maker_fee=0.0025, taker_fee=0.0025,
        avg_slippage=0.002,
        gas_cost_usd=0.02,
        min_trade_usd=500),

    "orca": VenueCosts(
        "orca", VenueType.DEX,
        maker_fee=0.003, taker_fee=0.003,
        avg_slippage=0.002,
        gas_cost_usd=0.02,
        min_trade_usd=500),

    "drift": VenueCosts(
        "drift", VenueType.DEX,
        maker_fee=0.0000, taker_fee=0.0003,
        avg_slippage=0.0005,
        gas_cost_usd=0.05,                    # Solana
        min_trade_usd=1000),
}


# -----------------------------------------------------------------------
#  Combined lookup
# -----------------------------------------------------------------------

VENUE_COSTS = {**_CEX_VENUES, **_HYBRID_VENUES, **_DEX_VENUES}

# convenience alias: "uniswap_v3" defaults to Arbitrum (cheapest viable chain)
VENUE_COSTS["uniswap_v3"] = VENUE_COSTS["uniswap_v3_arb"]


def get_venue_costs(venue: str) -> VenueCosts:
    """
    Look up venue costs. Falls back to conservative CEX defaults
    if venue is not recognized.
    """
    key = venue.lower().replace(" ", "_").replace("-", "_")
    if key not in VENUE_COSTS:
        return VenueCosts(venue, VenueType.CEX,
                          0.0005, 0.0005, 0.0005, 0.0, 100)
    return VENUE_COSTS[key]


def list_venues(venue_type: VenueType = None) -> list[str]:
    """List available venue names, optionally filtered by type."""
    if venue_type is None:
        return list(VENUE_COSTS.keys())
    return [k for k, v in VENUE_COSTS.items() if v.venue_type == venue_type]
