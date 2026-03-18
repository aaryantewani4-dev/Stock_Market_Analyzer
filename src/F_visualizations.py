"""
================================================================================
EXPLORATORY DATA ANALYSIS & VISUALIZATIONS
================================================================================
Purpose: Create comprehensive visualizations for stock/crypto data
Author: Your Name
Date: February 2026

This script generates various charts to understand:
- Price trends and patterns
- Moving averages and technical indicators
- Volume analysis
- Return distributions
- Cumulative performance

Note: This script is configured for AAPL. To analyze Bitcoin instead,
      uncomment the Bitcoin data loading line and update the title labels.
================================================================================
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Set seaborn style for better-looking plots
sns.set_style("dark")


# ================================================================================
# CONFIGURATION: Choose which asset to visualize
# ================================================================================
# AAPL (Apple Stock) - CURRENTLY ACTIVE:
DATA_FILE = '../data/aaple_stock.csv'
ASSET_NAME = "AAPL"
ASSET_FULL_NAME = "Apple Inc."

# BITCOIN - UNCOMMENT TO USE INSTEAD:
# DATA_FILE = '../data/bitcoin.csv'
# ASSET_NAME = "Bitcoin"
# ASSET_FULL_NAME = "Bitcoin (BTC-USD)"


# ================================================================================
# LOAD DATA
# ================================================================================
print("\n" + "="*80)
print(f"EXPLORATORY DATA ANALYSIS FOR {ASSET_NAME}")
print("="*80)

df = pd.read_csv(DATA_FILE)

print(f"\n✅ Data loaded: {len(df)} rows")
print(f"   Date range: {df['Price'].iloc[0]} to {df['Price'].iloc[-1]}")
print(f"   Columns: {list(df.columns)}")


# ================================================================================
# CHART 1: BASIC LINE PLOT (Price Over Time)
# ================================================================================
print("\n" + "-"*80)
print("Creating Chart 1: Basic Price Line Chart")
print("-"*80)

plt.figure(figsize=(10, 4))
sns.lineplot(data=df, y=df['Close'], x=df['Price'])
plt.title(f'{ASSET_NAME} Closing Price Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=11)
plt.ylabel('Closing Price ($)', fontsize=11)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("✅ Basic line chart displayed")


# ================================================================================
# CHART 2: AREA CHART (Price with Filled Area)
# ================================================================================
print("\n" + "-"*80)
print("Creating Chart 2: Area Chart")
print("-"*80)

# Area chart emphasizes the magnitude/volume of price
plt.figure(figsize=(12, 6))

# Fill area below the line
plt.fill_between(df['Price'], df['Close'], alpha=0.3, color='green')

# Draw line on top
plt.plot(df['Price'], df['Close'], linewidth=2, color='darkgreen')

plt.title(f'{ASSET_NAME} Stock Closing Price - Area Chart', 
          fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Closing Price ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("✅ Area chart displayed")


# ================================================================================
# CHART 3: PRICE WITH MOVING AVERAGES
# ================================================================================
print("\n" + "-"*80)
print("Creating Chart 3: Price with Moving Averages")
print("-"*80)

# Calculate moving averages
# These help identify trends and support/resistance levels
df['MA_20'] = df['Close'].rolling(window=20).mean()   # ~1 month
df['MA_50'] = df['Close'].rolling(window=50).mean()   # ~2.5 months
df['MA_200'] = df['Close'].rolling(window=200).mean() # ~10 months

plt.figure(figsize=(14, 6))

# Plot price and moving averages
plt.plot(df['Price'], df['Close'], label=f'{ASSET_NAME} Close Price',
         linewidth=2.5, color='black', zorder=5)
plt.plot(df['Price'], df['MA_20'], label='20-Day MA',
         linewidth=2, color='orange', linestyle='--', alpha=0.8)
plt.plot(df['Price'], df['MA_50'], label='50-Day MA',
         linewidth=2, color='red', linestyle='--', alpha=0.8)
plt.plot(df['Price'], df['MA_200'], label='200-Day MA',
         linewidth=2, color='green', linestyle='--', alpha=0.8)

plt.title(f'{ASSET_NAME} Stock Price with Moving Averages',
          fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=13)
plt.ylabel('Price ($)', fontsize=13)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("✅ Moving averages chart displayed")
print("\n💡 Interpretation:")
print("   • Price above MA: Bullish trend")
print("   • Price below MA: Bearish trend")
print("   • MA crossovers: Potential buy/sell signals")


# ================================================================================
# CHART 4: PRICE & VOLUME (Two Subplots)
# ================================================================================
print("\n" + "-"*80)
print("Creating Chart 4: Price & Volume Analysis")
print("-"*80)

# Create figure with two subplots sharing the same x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Subplot 1: Price
ax1.plot(df['Price'], df['Close'], linewidth=2, color='#0066cc')
ax1.set_title(f'{ASSET_NAME} Stock Price & Volume Analysis',
              fontsize=18, fontweight='bold')
ax1.set_ylabel('Closing Price ($)', fontsize=13)
ax1.grid(True, alpha=0.3)

# Subplot 2: Volume
ax2.bar(df['Price'], df['Volume'], color='black', alpha=0.6, width=1)
ax2.set_ylabel('Volume', fontsize=13)
ax2.set_xlabel('Date', fontsize=13)
ax2.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("✅ Price & Volume chart displayed")
print("\n💡 Interpretation:")
print("   • High volume + price increase: Strong buying")
print("   • High volume + price decrease: Strong selling")
print("   • Low volume: Weak conviction in price movement")


# ================================================================================
# CHART 5: CANDLESTICK CHART
# ================================================================================
print("\n" + "-"*80)
print("Creating Chart 5: Candlestick Chart")
print("-"*80)

# Candlestick charts show Open, High, Low, Close (OHLC) data
# Industry-standard for technical analysis

fig, ax = plt.subplots(figsize=(16, 8))

# Sample every 5th day for clarity (optional - adjust as needed)
df_sample = df[::5].reset_index(drop=True)

# Draw each candlestick
for idx, row in df_sample.iterrows():
    # Determine color based on price direction
    # Green: Close >= Open (price went up)
    # Red: Close < Open (price went down)
    color = 'green' if row['Close'] >= row['Open'] else 'red'
    
    # Draw high-low line (wick)
    ax.plot([idx, idx], [row['Low'], row['High']],
            color=color, linewidth=1.2)
    
    # Draw open-close rectangle (body)
    height = abs(row['Close'] - row['Open'])
    bottom = min(row['Open'], row['Close'])
    
    ax.add_patch(plt.Rectangle((idx - 0.3, bottom), 0.6, height,
                               facecolor=color, edgecolor=color))

ax.set_title(f'{ASSET_NAME} Candlestick Chart', fontsize=18, fontweight='bold')
ax.set_ylabel('Price ($)', fontsize=13)
ax.set_xlabel('Date (sampled every 5 days)', fontsize=13)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("✅ Candlestick chart displayed")
print("\n💡 Interpretation:")
print("   • Green candle: Bullish (price closed higher)")
print("   • Red candle: Bearish (price closed lower)")
print("   • Long wicks: High volatility/indecision")
print("   • Short body: Small price change")


# ================================================================================
# CHART 6: RETURN DISTRIBUTION
# ================================================================================
print("\n" + "-"*80)
print("Creating Chart 6: Return Distribution Analysis")
print("-"*80)

# Calculate daily returns (percentage change)
df['Returns'] = df['Close'].pct_change() * 100

# Create two subplots: histogram and box plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Histogram
ax1.hist(df['Returns'].dropna(), bins=50, color='steelblue',
         edgecolor='black', alpha=0.7)
ax1.set_title(f'{ASSET_NAME} Daily Returns Distribution',
              fontsize=16, fontweight='bold')
ax1.set_xlabel('Daily Returns (%)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero return')
ax1.legend()

# Add statistics
mean_return = df['Returns'].mean()
std_return = df['Returns'].std()
ax1.text(0.02, 0.98, f'Mean: {mean_return:.4f}%\nStd: {std_return:.4f}%',
         transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Subplot 2: Box Plot
ax2.boxplot(df['Returns'].dropna(), vert=True)
ax2.set_title(f'{ASSET_NAME} Returns Box Plot', fontsize=16, fontweight='bold')
ax2.set_ylabel('Daily Returns (%)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.show()

print("✅ Return distribution charts displayed")
print(f"\n📊 Return Statistics:")
print(f"   Mean: {mean_return:.4f}%")
print(f"   Std Dev: {std_return:.4f}%")
print(f"   Min: {df['Returns'].min():.2f}%")
print(f"   Max: {df['Returns'].max():.2f}%")


# ================================================================================
# CHART 7: CUMULATIVE RETURNS
# ================================================================================
print("\n" + "-"*80)
print("Creating Chart 7: Cumulative Returns")
print("-"*80)

# Calculate cumulative returns
# Shows total growth/loss if you invested at the start
df['Returns'] = df['Close'].pct_change() * 100
df['Cumulative_Return'] = (1 + df['Returns']/100).cumprod() - 1

# Plot cumulative returns
plt.figure(figsize=(15, 7))
plt.plot(df['Price'], df['Cumulative_Return'] * 100,
         linewidth=2.5, color='green')
plt.fill_between(df['Price'], df['Cumulative_Return'] * 100,
                 alpha=0.3, color='green')
plt.title(f'{ASSET_NAME} Cumulative Returns Over Time',
          fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=13)
plt.ylabel('Cumulative Return (%)', fontsize=13)
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate total return
total_return = df['Cumulative_Return'].iloc[-1] * 100
print(f"✅ Cumulative returns chart displayed")
print(f"\n💰 Investment Performance:")
print(f"   Total Return: {total_return:.2f}%")
print(f"   Initial Investment: $100")
print(f"   Final Value: ${100 * (1 + total_return/100):.2f}")

# Annualized return (approximate)
n_years = len(df) / 252  # 252 trading days per year
annualized_return = ((1 + total_return/100) ** (1/n_years) - 1) * 100
print(f"   Annualized Return: ~{annualized_return:.2f}% per year")


# ================================================================================
# SUMMARY
# ================================================================================
print("\n" + "="*80)
print("VISUALIZATION ANALYSIS COMPLETE!")
print("="*80)

print(f"\n📊 Charts Created:")
print(f"   1. ✅ Basic price line chart")
print(f"   2. ✅ Area chart")
print(f"   3. ✅ Price with moving averages")
print(f"   4. ✅ Price & volume analysis")
print(f"   5. ✅ Candlestick chart")
print(f"   6. ✅ Return distribution (histogram & box plot)")
print(f"   7. ✅ Cumulative returns")

print(f"\n🎯 Key Insights:")
print(f"   • Average daily return: {mean_return:.4f}%")
print(f"   • Daily volatility: {std_return:.4f}%")
print(f"   • Total return: {total_return:.2f}%")
print(f"   • Annualized return: ~{annualized_return:.2f}%")

print(f"\n💡 Note: To analyze Bitcoin instead:")
print(f"   • Uncomment Bitcoin data loading line at top")
print(f"   • Update ASSET_NAME and ASSET_FULL_NAME variables")
print(f"   • Re-run this script")
