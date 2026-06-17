# ============================================
# MODULE 9: SALES FORECASTING
# E-Commerce Sales Analytics Project
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================
# STEP 1: LOAD DATASET
# ============================================

df = pd.read_csv("Excel/monthly_revenue_analysis_orders.csv")

print("\nMonthly Revenue Dataset")
print(df.head())

# ============================================
# STEP 2: CREATE TIME INDEX
# ============================================

df['Month_Number'] = range(1, len(df) + 1)

print("\nDataset with Time Index")
print(df)

# ============================================
# STEP 3: PREPARE DATA
# ============================================

X = df[['Month_Number']]
y = df['Revenue']

# ============================================
# STEP 4: TRAIN MODEL
# ============================================

model = LinearRegression()

model.fit(X, y)

print("\nModel Training Completed")

# ============================================
# STEP 5: PREDICT EXISTING DATA
# ============================================

df['Predicted_Revenue'] = model.predict(X)

# ============================================
# STEP 6: MODEL EVALUATION
# ============================================

mae = mean_absolute_error(y, df['Predicted_Revenue'])
r2 = r2_score(y, df['Predicted_Revenue'])

print("\nModel Performance")
print("MAE :", round(mae, 2))
print("R2 Score :", round(r2, 4))

# ============================================
# STEP 7: FORECAST NEXT 6 MONTHS
# ============================================

future_months = pd.DataFrame({
    'Month_Number': range(
        len(df) + 1,
        len(df) + 7
    )
})

future_revenue = model.predict(future_months)

forecast_df = pd.DataFrame({
    'Future_Month_Number': future_months['Month_Number'],
    'Predicted_Revenue': np.round(future_revenue, 2)
})

print("\nFuture Revenue Forecast")
print(forecast_df)

# ============================================
# STEP 8: SAVE FORECAST
# ============================================

forecast_df.to_csv(
    "future_revenue_forecast.csv",
    index=False
)

print("\nForecast saved as future_revenue_forecast.csv")

# ============================================
# STEP 9: FORECAST VISUALIZATION
# ============================================

plt.figure(figsize=(12,6))

plt.plot(
    df['Month_Number'],
    df['Revenue'],
    marker='o',
    label='Actual Revenue'
)

plt.plot(
    df['Month_Number'],
    df['Predicted_Revenue'],
    marker='o',
    label='Model Fit'
)

plt.plot(
    future_months['Month_Number'],
    future_revenue,
    marker='o',
    label='Forecast Revenue'
)

plt.title("Sales Revenue Forecast")
plt.xlabel("Month Number")
plt.ylabel("Revenue")
plt.grid(True)
plt.legend()

plt.savefig("forecast_chart.png")

plt.show()

print("\nForecast chart saved as forecast_chart.png")

# ============================================
# STEP 10: BUSINESS SUMMARY
# ============================================

print("\n========== FORECAST SUMMARY ==========")

print("Total Historical Months :", len(df))
print("Forecast Months :", 6)

print("\nExpected Revenue For Next 6 Months:")

for i, revenue in enumerate(future_revenue, start=1):
    print(
        f"Month +{i}: ₹{revenue:,.2f}"
    )

print("\nSales Forecasting Completed Successfully")