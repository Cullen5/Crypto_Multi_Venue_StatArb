"""
costs.py — venue-aware transaction cost model

Numbers from case study specs. Never hardcode fees in strategy code,
always go through get_venue_costs().
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
    maker_fee: float
    taker_fee: float
    avg_slippage: float
    gas_cost_usd: float
    min_trade_usd: float

    @property
    def round_trip_cost(self) -> float:
        return self.maker_fee + self.taker_fee + 2 * self.avg_slippage


VENUE_COSTS = {
    "binance": VenueCosts("binance", VenueType.CEX,
                          0.0001, 0.0004, 0.0002, 0.0, 100),
    "binance_spot": VenueCosts("binance_spot", VenueType.CEX,
                               0.0005, 0.0005, 0.0003, 0.0, 100),
    "bybit": VenueCosts("bybit", VenueType.CEX,
                         0.0001, 0.0006, 0.0003, 0.0, 100),
    "okx": VenueCosts("okx", VenueType.CEX,
                       0.0002, 0.0005, 0.0003, 0.0, 100),
    "deribit": VenueCosts("deribit", VenueType.CEX,
                           0.0002, 0.0005, 0.0003, 0.0, 500),
    "hyperliquid": VenueCosts("hyperliquid", VenueType.HYBRID,
                               0.0000, 0.00025, 0.0003, 0.50, 500),
    "dydx": VenueCosts("dydx", VenueType.HYBRID,
                        0.0000, 0.0005, 0.0005, 0.10, 500),
    "gmx": VenueCosts("gmx", VenueType.DEX,
                       0.001, 0.001, 0.003, 1.50, 5000),
    "uniswap_v3": VenueCosts("uniswap_v3", VenueType.DEX,
                              0.003, 0.003, 0.003, 1.00, 5000),
    "drift": VenueCosts("drift", VenueType.DEX,
                         0.0000, 0.0003, 0.0005, 0.05, 1000),
}


def get_venue_costs(venue: str) -> VenueCosts:
    key = venue.lower().replace(" ", "_").replace("-", "_")
    if key not in VENUE_COSTS:
        return VenueCosts(venue, VenueType.CEX,
                          0.0005, 0.0005, 0.0005, 0.0, 100)
    return VENUE_COSTS[key]
