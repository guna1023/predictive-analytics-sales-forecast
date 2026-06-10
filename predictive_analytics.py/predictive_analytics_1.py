"""
============================================================
  PREDICTIVE ANALYTICS - SALES FORECASTING PROJECT
  By: Guna | B.Sc Data Science | Portfolio Project
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# STEP 1: CREATE / LOAD DATASET
# ============================================================

print("=" * 55)
print("   PREDICTIVE ANALYTICS - SALES FORECASTING")
print("=" * 55)

# Sample 24-month historical sales data (Jan 2023 - Dec 2024)
# Real project la: df = pd.read_csv("your_data.csv")

data = {
    "Month": pd.date_range(start="2023-01", periods=24, freq="MS"),
    "Sales": [
        42000, 38500, 45000, 48000, 51000, 55000,
        52000, 58000, 61000, 65000, 70000, 75000,
        69000, 64000, 72000, 76000, 80000, 84000,
        79000, 88000, 92000, 96000, 101000, 108000
    ]
}

df = pd.DataFrame(data)
df["Month_Num"] = range(len(df))  # Numeric index for regression

print("\n✅ STEP 1: Dataset Loaded")
print(f"   📊 Total Records : {len(df)}")
print(f"   📅 Date Range    : {df['Month'].min().strftime('%b %Y')} → {df['Month'].max().strftime('%b %Y')}")
print(f"   💰 Sales Range   : ₹{df['Sales'].min():,} → ₹{df['Sales'].max():,}")

# ============================================================
# STEP 2: DATA PREPROCESSING
# ============================================================

print("\n✅ STEP 2: Data Preprocessing")

# Check for missing values
missing = df.isnull().sum().sum()
print(f"   🔍 Missing Values : {missing} (Clean!)" if missing == 0 else f"   ⚠️  Missing Values : {missing}")

# Basic statistics
print(f"   📈 Mean Sales     : ₹{df['Sales'].mean():,.0f}")
print(f"   📉 Std Deviation  : ₹{df['Sales'].std():,.0f}")

# Feature: Month number (for regression)
X = df[["Month_Num"]].values
y = df["Sales"].values

# ============================================================
# STEP 3: TRAIN MODEL - LINEAR REGRESSION
# ============================================================

print("\n✅ STEP 3: Training Linear Regression Model")

model = LinearRegression()
model.fit(X, y)

# Predictions on training data (backtest)
y_pred_train = model.predict(X)

# Accuracy metrics
mae = mean_absolute_error(y, y_pred_train)
r2  = r2_score(y, y_pred_train)

print(f"   📐 Slope (Trend)        : ₹{model.coef_[0]:,.0f} per month")
print(f"   📌 Intercept            : ₹{model.intercept_:,.0f}")
print(f"   📉 MAE (Error)          : ₹{mae:,.0f}")
print(f"   🎯 R² Score (Accuracy)  : {r2:.4f} ({r2*100:.1f}%)")
print(f"   ⭐ Rating               : {'Excellent' if r2 > 0.95 else 'Good' if r2 > 0.85 else 'Fair'}")

# ============================================================
# STEP 4: MOVING AVERAGE MODEL
# ============================================================

print("\n✅ STEP 4: Moving Average Model (Window = 3)")

window = 3
df["MA_3"] = df["Sales"].rolling(window=window).mean()

# Last 3 values average for projection
last_avg = df["Sales"].iloc[-window:].mean()
growth   = (df["Sales"].iloc[-1] - df["Sales"].iloc[-4]) / (df["Sales"].iloc[-4] * 3)
print(f"   📊 Last 3-Month Avg     : ₹{last_avg:,.0f}")
print(f"   📈 Growth Rate/Month    : {growth*100:.2f}%")

# ============================================================
# STEP 5: FORECAST FUTURE 6 MONTHS
# ============================================================

forecast_months = 6
print(f"\n✅ STEP 5: Forecasting Next {forecast_months} Months")

# Future month numbers
future_nums = np.array([[len(df) + i] for i in range(forecast_months)])
future_dates = pd.date_range(start="2025-01", periods=forecast_months, freq="MS")

# Linear Regression forecast
lr_forecast = model.predict(future_nums)

# Moving Average forecast
ma_forecast = []
for i in range(forecast_months):
    projected = last_avg * (1 + growth * (i + 1))
    ma_forecast.append(projected)
ma_forecast = np.array(ma_forecast)

forecast_df = pd.DataFrame({
    "Month"           : future_dates,
    "LR_Forecast"     : lr_forecast.astype(int),
    "MA_Forecast"     : ma_forecast.astype(int)
})

print("\n   📅 Month         | LR Forecast  | MA Forecast")
print("   " + "-" * 46)
for _, row in forecast_df.iterrows():
    print(f"   {row['Month'].strftime('%b %Y'):<16} | ₹{row['LR_Forecast']:>10,} | ₹{row['MA_Forecast']:>10,}")

total_lr = forecast_df["LR_Forecast"].sum()
total_ma = forecast_df["MA_Forecast"].sum()
print(f"\n   💰 Total LR Forecast (6 months) : ₹{total_lr:,}")
print(f"   💰 Total MA Forecast (6 months) : ₹{total_ma:,}")

# ============================================================
# STEP 6: VISUALIZATIONS
# ============================================================

print("\n✅ STEP 6: Generating Charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Predictive Analytics — Sales Forecasting Dashboard",
             fontsize=16, fontweight="bold", y=1.01)

colors = {
    "actual"   : "#378ADD",
    "lr"       : "#D85A30",
    "ma"       : "#1D9E75",
    "trend"    : "#888780",
    "forecast_lr": "#D85A30",
    "forecast_ma": "#534AB7"
}

# ─── Chart 1: Historical Sales + Trend ─────────────────────
ax1 = axes[0, 0]
ax1.plot(df["Month"], df["Sales"], color=colors["actual"], linewidth=2,
         marker="o", markersize=4, label="Actual Sales", zorder=3)
ax1.plot(df["Month"], y_pred_train, color=colors["lr"], linestyle="--",
         linewidth=1.5, label=f"Trend Line (R²={r2:.3f})", zorder=2)
ax1.fill_between(df["Month"], df["Sales"], alpha=0.1, color=colors["actual"])
ax1.set_title("Historical Sales + Linear Trend", fontweight="bold")
ax1.set_ylabel("Sales (₹)")
ax1.legend(fontsize=9)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax1.tick_params(axis="x", rotation=45)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
ax1.grid(alpha=0.3)

# ─── Chart 2: Future Forecast ───────────────────────────────
ax2 = axes[0, 1]
all_dates   = list(df["Month"]) + list(future_dates)
all_actual  = list(df["Sales"]) + [None] * forecast_months
all_lr      = [None] * len(df) + list(lr_forecast)
all_ma      = [None] * len(df) + list(ma_forecast)

ax2.plot(df["Month"], df["Sales"], color=colors["actual"], linewidth=2,
         marker="o", markersize=3, label="Actual Sales", zorder=3)
ax2.plot(future_dates, lr_forecast, color=colors["forecast_lr"], linewidth=2.5,
         marker="^", markersize=7, linestyle="--", label="LR Forecast", zorder=4)
ax2.plot(future_dates, ma_forecast, color=colors["forecast_ma"], linewidth=2.5,
         marker="s", markersize=7, linestyle="-.", label="MA Forecast", zorder=4)
ax2.axvline(x=df["Month"].iloc[-1], color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
ax2.text(df["Month"].iloc[-1], ax2.get_ylim()[0] if ax2.get_ylim()[0] else 30000,
         "  Forecast →", fontsize=9, color="gray", va="bottom")
ax2.set_title("6-Month Sales Forecast", fontweight="bold")
ax2.set_ylabel("Sales (₹)")
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax2.tick_params(axis="x", rotation=45)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
ax2.grid(alpha=0.3)

# ─── Chart 3: Moving Average Smoothing ──────────────────────
ax3 = axes[1, 0]
ax3.plot(df["Month"], df["Sales"], color=colors["actual"], linewidth=1.5,
         alpha=0.5, marker="o", markersize=3, label="Actual Sales")
ax3.plot(df["Month"], df["MA_3"], color=colors["ma"], linewidth=2.5,
         label="3-Month Moving Average")
ax3.set_title("Moving Average Smoothing (Window=3)", fontweight="bold")
ax3.set_ylabel("Sales (₹)")
ax3.legend(fontsize=9)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax3.tick_params(axis="x", rotation=45)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
ax3.grid(alpha=0.3)

# ─── Chart 4: Model Accuracy Bar ────────────────────────────
ax4 = axes[1, 1]
residuals = y - y_pred_train
bar_colors = ["#D85A30" if r < 0 else "#1D9E75" for r in residuals]
ax4.bar(df["Month"], residuals, color=bar_colors, width=20, alpha=0.8)
ax4.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax4.set_title(f"Prediction Residuals (Error)\nMAE = ₹{mae:,.0f} | R² = {r2:.4f}",
              fontweight="bold")
ax4.set_ylabel("Error (₹)")
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax4.tick_params(axis="x", rotation=45)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
ax4.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("sales_forecast_dashboard.png", dpi=150, bbox_inches="tight")
print("   💾 Chart saved: sales_forecast_dashboard.png")
plt.show()

# ============================================================
# STEP 7: SUMMARY REPORT
# ============================================================

print("\n" + "=" * 55)
print("   📋 FINAL PROJECT SUMMARY")
print("=" * 55)
print(f"""
   Dataset       : 24 months sales data (2023–2024)
   Models Used   : Linear Regression + Moving Average
   MAE           : ₹{mae:,.0f}
   R² Score      : {r2:.4f} ({r2*100:.1f}% accuracy)

   6-Month Forecast (LR):
   • Jan 2025    : ₹{int(lr_forecast[0]):,}
   • Jun 2025    : ₹{int(lr_forecast[-1]):,}
   • Growth      : +{((lr_forecast[-1]-df['Sales'].iloc[-1])/df['Sales'].iloc[-1]*100):.1f}%

   Files Created:
   ✅ predictive_analytics.py     (this script)
   ✅ sales_forecast_dashboard.png (chart image)
""")
print("=" * 55)
print("  Project Complete! LinkedIn post ready. 🚀")
print("=" * 55)
