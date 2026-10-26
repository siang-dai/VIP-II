# VIP II

Daily volatility and cross-asset correlation for **TAIEX, SPX, SPY, TLT, IEF, IAU, 0050 and VIX**.

Interactive page: [VIP II on Siang-Dai's website](https://siang-dai.github.io/vip-ii.html).

## Acknowledgement

VIP II continues the idea of the original **Volatility Information Platform** by Erik Apostol and its contributors:
[ErikApostol/volatility_index_platform](https://github.com/ErikApostol/volatility_index_platform).
The predecessor is a reference only. This repository contains a fresh implementation and does not modify or bundle its Django app, databases, cached charts, or historical exports.

## Six estimators

All calculations use daily positive prices and log returns. Annualization is 252 trading sessions. Prices have their source's local trading dates. No prices are filled across holidays.

| Estimator | Definition |
| --- | --- |
| Historical volatility | Sample standard deviation of 22 log returns × sqrt(252) × 100; requires 23 prices. |
| EWMA volatility | Zero-mean return second moment, lambda 0.94; initialized with the first 22 return squares and updated with each latest return. Output is after-close annualized volatility. |
| GARCH(1,1) | Zero-mean Gaussian GARCH, 504 trailing returns; parameters refitted every 21 observations using only data through the previous observation. Filter with today's return and publish the next-session forecast at its origin date. Unit-scaled fitting retries failed optimization from up to three deterministic feasible starts; every accepted fit must converge with valid parameters and finite variance. Otherwise publication stops. |
| Fourier instantaneous volatility | Daily-data, full-band Fourier/Fejer spot second-moment estimate. Use 252 trailing returns, reflect the window, apply a triangular multiplier through 8 Fourier modes, evaluate at the newest observed endpoint, annualize and take the square root. |
| Historical correlation | Pearson correlation over 22 paired log returns. Intersect price dates *before* differencing so paired returns span the same dates. |
| Fourier instantaneous correlation | Apply the same Fourier/Fejer weights to paired return squares and cross-products, divide local covariance by local volatilities. The shared nonnegative kernel preserves a positive-semidefinite covariance matrix. Zero variance gives an undefined result, never a fabricated zero. |

### Fourier adaptation and timing

This is a trailing, reflected-boundary adaptation of the Fourier idea in VIP I, **not a numerical reproduction of the original Malliavin–Mancino implementation or a tick-data estimator**. Reflection mitigates the circular endpoint wrap; it uses no future observations. For a window q of squared returns or paired products, form q* = [q, reverse(q)]. With frequency index k, set H(k) = max(1 - |k|/9, 0); the spot moment is IFFT(FFT(q*) H) at index 251. The equivalent kernel is calculated once and applied to trailing windows. Tests check it against an independent FFT reconstruction. No clipping of substantive out-of-range correlations or replacement with previous values is used.

The 1M / 3M / 6M / 1Y / 5Y controls only change the displayed dates. They do not refit models. Calculations use history from 2016 for warm-up and stable estimator origins; only five years of output are published. New observations do not change past estimates unless the source revises its historical prices (including ETF adjustment factors).

## Sources and interpretation

- TAIEX `^TWII`, SPX `^GSPC`, SPY, TLT, IEF, IAU and `0050.TW`: Yahoo Finance through yfinance. ETFs use explicit **Adj Close**; indices use **Close**. This version does not claim to download TAIEX from TWSE.
- VIX: [Cboe daily historical CSV](https://www.cboe.com/tradable_products/vix/vix_historical_data/). VIX is an index level, not an investable asset. Calculating its log-change volatility is not calculating the VIX or official VVIX.
- [yfinance](https://github.com/ranaroussi/yfinance) is unofficial. Source availability and permitted use are governed by each provider; software availability is not a data redistribution license. Output here is a research presentation, not a guaranteed market data service.
- Cross-market pairs use local-date matching. Taiwan closes before the US: these are **asynchronous descriptive correlations**, not a same-information-time or predictive signal. Returns remain in each asset's native currency, with no FX conversion. Common-date returns may span more than one session around different holidays.
- Different assets may have different latest dates. Exchange calendars plus a two-hour post-close buffer exclude incomplete sessions. Missing/truncated/stale data or failed fits abort publication and preserve the last good release.

## Run locally

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m vip2.update
```

Use `--force` to recalculate unchanged dates. The companion website's `vip-ii.qmd` and `vip-ii/` assets are the only page-specific integration; other research pages are independent.
