from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    id: str
    name: str
    symbol: str
    calendar: str
    currency: str
    adjusted: bool = True


ASSETS = (
    Asset("TAIEX", "Taiwan weighted index", "^TWII", "XTAI", "TWD", False),
    Asset("SPX", "S&P 500 index", "^GSPC", "XNYS", "USD", False),
    Asset("SPY", "S&P 500 ETF", "SPY", "XNYS", "USD"),
    Asset("TLT", "20+ year US Treasury ETF", "TLT", "XNYS", "USD"),
    Asset("IEF", "7–10 year US Treasury ETF", "IEF", "XNYS", "USD"),
    Asset("IAU", "Gold trust", "IAU", "XNYS", "USD"),
    Asset("0050", "Taiwan 50 ETF", "0050.TW", "XTAI", "TWD"),
    Asset("VIX", "Cboe Volatility Index", "^VIX", "XNYS", "points", False),
)
HISTORY_START = "2016-01-01"
HV_WINDOW = 22
EWMA_DECAY = 0.94
FOURIER_WINDOW = 252
FOURIER_MODES = 8
GARCH_WINDOW = 504
GARCH_REFIT = 21
ANNUALIZATION = 252
