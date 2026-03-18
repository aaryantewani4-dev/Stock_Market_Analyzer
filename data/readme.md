# 📂 Data Directory

This folder contains all raw and processed data files used in the project.

---

## 📄 Files

### AAPL (Apple Stock)

| File | Description |
|------|-------------|
| `aaple_stock.csv` | Raw AAPL stock data downloaded from Yahoo Finance |
| `aaple_stock_features_returns.csv` | Processed AAPL data with all engineered features |

### Bitcoin

| File | Description |
|------|-------------|
| `bitcoin.csv` | Raw Bitcoin (BTC-USD) data downloaded from Yahoo Finance |
| `bitcoin_features.csv` | Processed Bitcoin data with all engineered features |

---

## 📊 Raw Data Columns

Both `aaple_stock.csv` and `bitcoin.csv` contain:

| Column | Description |
|--------|-------------|
| `Price` | Trading date |
| `Open` | Opening price of the day ($) |
| `High` | Highest price during the day ($) |
| `Low` | Lowest price during the day ($) |
| `Close` | Closing price of the day ($) |
| `Volume` | Number of shares/coins traded |

---

## 🔧 Processed Data Features

Both `aaple_stock_features_returns.csv` and `bitcoin_features.csv` contain:

### Returns
| Feature | Description |
|---------|-------------|
| `Returns` | Daily percentage return |
| `Returns_Lag_1` | Yesterday's return |
| `Returns_Lag_2` | Return from 2 days ago |
| `Returns_Lag_3` | Return from 3 days ago |

### Price Position
| Feature | Description |
|---------|-------------|
| `Price_to_MA5` | Price / 5-day moving average |
| `Price_to_MA10` | Price / 10-day moving average |
| `Price_to_MA20` | Price / 20-day moving average |

### Momentum
| Feature | Description |
|---------|-------------|
| `Momentum_5` | 5-day percentage change |
| `Momentum_10` | 10-day percentage change |

### Volatility
| Feature | Description |
|---------|-------------|
| `Volatility_5` | 5-day standard deviation of returns |
| `Volatility_10` | 10-day standard deviation of returns |

### Volume
| Feature | Description |
|---------|-------------|
| `Volume_Change` | Daily volume change (%) |
| `Volume_Ratio` | Current volume / 5-day average volume |

### Technical Indicators
| Feature | Description |
|---------|-------------|
| `RSI` | Relative Strength Index (0-100) |
| `High_Low_Spread_Pct` | Intraday price range (%) |

### Target Variable
| Feature | Description |
|---------|-------------|
| `Target_Return` | Next day's percentage return (what we predict) |

---

## 📅 Date Ranges

| Asset | Start Date | End Date | Records |
|-------|-----------|----------|---------|
| AAPL | 2019-01-01 | 2025-12-31 | ~1,739 |
| Bitcoin | 2021-01-01 | 2026-02-12 | ~1,800+ |

---

## 🔄 How to Regenerate Data

If you need to download fresh data:

```bash
# From project root
python src/1_data_collection.py
```

Then recreate features:

```bash
python src/2_feature_engineering.py
```

---

## ⚠️ Notes

- Raw data downloaded using `yfinance` library
- All NaN values removed from processed files
- Data is ready to use directly in model training
- No additional preprocessing required
