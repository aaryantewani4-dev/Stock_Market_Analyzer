# 📈 Stock Market Analyzer

A machine learning project that predicts **Apple (AAPL)** stock prices and **Bitcoin (BTC-USD)** prices using Random Forest regression.

**Final Performance:** R² = 0.9698 | MAE = $2.69 | Error = 1.16% of avg price

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange)
![pandas](https://img.shields.io/badge/pandas-2.2.2-green)

---

## 🎯 Project Overview

This project builds a complete machine learning pipeline to predict next-day stock and cryptocurrency prices:

- Downloads historical price data using `yfinance`
- Engineers 15 technical indicators as model features
- Trains a **Random Forest Regressor** on percentage returns
- Converts predicted returns back to actual prices
- Evaluates performance against a baseline model

---

## 📊 Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Price R² | 0.9698 | 96.98% variance explained |
| Price MAE | $2.69 | Average prediction error |
| Price RMSE | $4.00 | Root mean squared error |
| Error % | 1.16% | Error as % of avg price |
| Return R² | -0.03 | Normal for stock returns |

### Baseline Comparison

| Model | MAE | R² |
|-------|-----|----|
| Our ML Model | $2.69 | 0.9698 |
| Baseline (Tomorrow=Today) | $2.62 | 0.9710 |

> Being within 0.001 R² of baseline proves the model is **realistic and not overfit**.

---

## 🛠️ Technologies

| Library | Purpose |
|---------|---------|
| `yfinance` | Download stock/crypto data |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Machine learning |
| `matplotlib` | Visualizations |
| `seaborn` | Statistical plots |

---

## 📁 Project Structure

```
stock_market_analyzer/
│
├── data/
│   ├── aaple_stock.csv                    # Raw AAPL data
│   ├── aaple_stock_features_returns.csv   # Processed AAPL features
│   ├── bitcoin.csv                        # Raw Bitcoin data
│   ├── bitcoin_features.csv               # Processed Bitcoin features
│   └── README.md                          # Data documentation
│
├── src/
│   ├── 1_data_collection.py               # Download stock/crypto data
│   ├── 2_feature_engineering.py           # Create technical indicators
│   ├── 3_feature_loading.py               # Load & split data
│   ├── 4_model_training.py                # Train Random Forest model
│   ├── 5_feature_importance.py            # Analyze feature importance
│   ├── 6_visualizations.py                # Generate charts
│   └── 7_diagnostics.py                   # Model validation
├── .gitignore
├── requirements.txt
├── README.md
└── PROJECT_REPORT.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stock_market_analyzer.git
cd stock_market_analyzer
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline

```bash
# Step 1: Download data
python src/1_data_collection.py

# Step 2: Create features
python src/2_feature_engineering.py

# Step 3: Load and prepare data
python src/3_feature_loading.py

# Step 4: Train model and predict
python src/4_model_training.py

# Step 5: Analyze features
python src/5_feature_importance.py

# Step 6: Generate visualizations
python src/6_visualizations.py

# Step 7: Run diagnostics
python src/7_diagnostics.py
```

---

## 🔄 Switch Between AAPL and Bitcoin

Each script has a configuration section at the top. To switch assets:

```python
# AAPL (Apple Stock) - DEFAULT:
INPUT_FILE = '../data/aaple_stock.csv'
ASSET_NAME = "AAPL"

# BITCOIN - UNCOMMENT THESE:
# INPUT_FILE = '../data/bitcoin.csv'
# ASSET_NAME = "Bitcoin"
```

---

## 📈 Features Used

15 scale-independent technical indicators:

| Category | Features |
|----------|---------|
| Returns | Daily %, Lag 1, Lag 2, Lag 3 |
| Price Position | Price/MA5, Price/MA10, Price/MA20 |
| Momentum | 5-day, 10-day |
| Volatility | 5-day std, 10-day std |
| Volume | Change %, Ratio |
| Technical | RSI, High-Low Spread % |

---

## 💡 Key Concepts

### Why Returns Instead of Prices?
```
$50 → $51 (2019) = +2.0% return
$250 → $255 (2024) = +2.0% return
Same pattern at any price level!
```

### Why Independent Predictions?
```
# Wrong (errors compound):
Day 2 price = predicted_day1 × (1 + return)

# Correct (no compounding):
Day 2 price = actual_day1 × (1 + predicted_return)
```

---

## 📝 Top Features by Importance

1. **Price_to_MA20** (9.6%) - Price vs 20-day average
2. **Price_to_MA10** (8.4%) - Price vs 10-day average
3. **Momentum_10** (8.2%) - 10-day momentum
4. **Price_to_MA5** (7.5%) - Price vs 5-day average
5. **Volatility_5** (7.5%) - Recent volatility

---

## 🔮 Future Improvements

- [ ] Add sentiment analysis from news/social media
- [ ] Multi-stock portfolio optimization
- [ ] Deep learning models (LSTM, GRU)
- [ ] Trading strategy backtesting
- [ ] Real-time prediction dashboard
- [ ] More cryptocurrencies

---

## 📄 Documentation

- [Data Documentation](data/README.md)
- [Technical Report](PROJECT_REPORT.md)

---

## 👤 Author
- GitHub: [@Github](https://github.com/aaryantewani4-dev)
- LinkedIn: [@LinkedIn](https://www.linkedin.com/in/aaryan-tewani-9200702b5/)

---

## ⭐ If you found this helpful, please give it a star!
