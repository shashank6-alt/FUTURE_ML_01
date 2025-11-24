# -----------------------------------------------------------
# Superstore Sales Forecasting using Prophet
# -----------------------------------------------------------

# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Step 2: Load data
df = pd.read_csv("Sample - Superstore.csv", encoding='latin1')

# Step 3: Inspect
print("First five rows:")
print(df.head(), "\n")
print("Data info:")
print(df.info(), "\n")

# Step 4: Clean and prepare data
df['Order Date'] = pd.to_datetime(df['Order Date'])
df = df[['Order Date', 'Sales']]
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

# Prophet needs columns 'ds' and 'y'
daily_sales.rename(columns={'Order Date': 'ds', 'Sales': 'y'}, inplace=True)

# Step 5: Plot sales trend
plt.figure(figsize=(10,5))
plt.plot(daily_sales['ds'], daily_sales['y'])
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# Step 6: Train Prophet model
print("Training Prophet model...")
model = Prophet()
model.fit(daily_sales)

# Step 7: Create future dates (next 90 days)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Step 8: Plot forecast
fig1 = model.plot(forecast)
plt.title("Sales Forecast (Next 90 Days)")
plt.show()

# Step 9: Plot components (trend, weekly, yearly)
fig2 = model.plot_components(forecast)
plt.show()

# Step 10: Export forecast results
forecast_export = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_export.to_csv("superstore_forecast.csv", index=False)
print("✅ Forecast exported to 'superstore_forecast.csv'")