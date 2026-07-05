# Changelog

## 2026-07-05

### Added

- Streamlit dashboard entrypoint for interactive forecasting.
- Exchange-aware ticker normalization for `US`, `NSE`, `BSE`, `JSE`, `LSE`, `ASX`, `HKEX`, `TSE`, `SGX`, `FRA`, and `EURONEXT`.
- Exchange-based ticker suggestions in the UI.
- Region presets for faster market selection.
- Saved CLI chart outputs in an `outputs/` folder.

### Fixed

- Prevented crashes when feature engineering produces too few rows.
- Hardened Yahoo Finance fetch handling for missing or malformed ticker data.
- Reduced scikit-learn feature-name warnings by preserving DataFrame inputs.
- Prevented CLI plot windows from blocking headless runs.

### Tested

- `AAPL / US`
- `RELIANCE / NSE`
- `RELIANCE / BSE`
- `JSE / JSE`
- `HSBA / LSE`
- `BHP / ASX`
- `0700 / HKEX`
- `7203 / TSE`
- `D05 / SGX`
- `SIE / FRA`
- `AIR / EURONEXT`

### Known limitations

- Some symbols need exact Yahoo Finance market tickers.
- Very short history windows can still produce too little data to train.
