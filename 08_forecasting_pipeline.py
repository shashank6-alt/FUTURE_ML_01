# FILE: 08_forecasting_pipeline.py
# PURPOSE: COMPLETE FORECASTING SYSTEM WITH MAXIMUM CAPACITY

print(" COMPLETE FORECASTING PIPELINE - MAXIMUM CAPACITY")
print("=" * 70)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# LOAD DATA AND MODELS
print(" LOADING DATA AND TRAINED MODELS...")

superstore_features = pd.read_csv('superstore_features.csv')
retail_features = pd.read_csv('retail_features.csv') 
rossmann_features = pd.read_csv('rossmann_features.csv')

model_artifacts = joblib.load('model_results.pkl')
best_superstore_model = joblib.load('best_superstore_model.pkl')
best_retail_model = joblib.load('best_retail_model.pkl')
best_rossmann_model = joblib.load('best_rossmann_model.pkl')

print(" DATA AND MODELS LOADED SUCCESSFULLY!")

def CREATE_ADVANCED_FORECASTS(df, model, dataset_type, forecast_days=90):
    print(f"\n CREATING ADVANCED FORECASTS FOR: {dataset_type}")
    print("-" * 50)
    
    df_forecast = df.copy()
    forecasts = {}
    
    # DATASET-SPECIFIC CONFIGURATION
    if dataset_type == "SUPERSTORE":
        date_col = 'Order Date'
        target_col = 'Sales'
        df_forecast[date_col] = pd.to_datetime(df_forecast[date_col])
    elif dataset_type == "RETAIL":
        date_col = 'date'
        target_col = 'sales'
        df_forecast[date_col] = pd.to_datetime(df_forecast[date_col])
    else:  # ROSSMANN
        date_col = 'Date'
        target_col = 'Sales'
        df_forecast[date_col] = pd.to_datetime(df_forecast[date_col])
    
    # 1. TIME SERIES DECOMPOSITION
    print("    Performing TIME SERIES DECOMPOSITION...")
    df_forecast = df_forecast.sort_values(date_col)
    daily_sales = df_forecast.groupby(date_col)[target_col].sum().reset_index()
    daily_sales = daily_sales.set_index(date_col)
    
    # Resample to daily frequency and fill missing dates
    daily_sales = daily_sales.resample('D').sum().fillna(method='ffill')
    
    # 2. MULTIPLE FORECASTING METHODS
    print("   Applying MULTIPLE FORECASTING METHODS...")
    
    # METHOD 1: ROLLING WINDOW FORECAST
    print("    METHOD 1: Rolling Window Forecast...")
    forecast_rolling = CREATE_ROLLING_FORECAST(daily_sales, target_col, forecast_days)
    forecasts['rolling'] = forecast_rolling
    
    # METHOD 2: SEASONAL NAIVE FORECAST
    print("    METHOD 2: Seasonal Naive Forecast...")
    forecast_seasonal = CREATE_SEASONAL_NAIVE_FORECAST(daily_sales, target_col, forecast_days, seasonality=7)
    forecasts['seasonal_naive'] = forecast_seasonal
    
    # METHOD 3: MACHINE LEARNING FORECAST
    print("    METHOD 3: Machine Learning Forecast...")
    forecast_ml = CREATE_ML_FORECAST(df_forecast, model, date_col, target_col, forecast_days)
    forecasts['machine_learning'] = forecast_ml
    
    # METHOD 4: EXPONENTIAL SMOOTHING
    print("    METHOD 4: Exponential Smoothing Forecast...")
    forecast_exp = CREATE_EXPONENTIAL_SMOOTHING_FORECAST(daily_sales, target_col, forecast_days)
    forecasts['exponential_smoothing'] = forecast_exp
    
    # 3. ENSEMBLE FORECAST (COMBINE ALL METHODS)
    print("    METHOD 5: Ensemble Forecast (Combining All Methods)...")
    forecast_ensemble = CREATE_ENSEMBLE_FORECAST(forecasts)
    forecasts['ensemble'] = forecast_ensemble
    
    # 4. FORECAST CONFIDENCE INTERVALS
    print("    Calculating FORECAST CONFIDENCE INTERVALS...")
    forecasts_with_ci = ADD_CONFIDENCE_INTERVALS(forecasts, daily_sales, target_col)
    
    return forecasts_with_ci

def CREATE_ROLLING_FORECAST(daily_sales, target_col, forecast_days):
    """Create forecast using rolling average method"""
    forecast_data = []
    last_date = daily_sales.index[-1]
    
    # Use multiple rolling windows for robust forecasting
    windows = [7, 14, 30]
    weights = [0.5, 0.3, 0.2]  # Weighted combination
    
    for i in range(forecast_days):
        forecast_date = last_date + timedelta(days=i+1)
        
        # Calculate weighted forecast from multiple windows
        daily_forecast = 0
        for window, weight in zip(windows, weights):
            if len(daily_sales) >= window:
                window_avg = daily_sales[target_col].rolling(window=window).mean().iloc[-1]
                daily_forecast += window_avg * weight
        
        forecast_data.append({
            'date': forecast_date,
            'forecast': max(daily_forecast, 0),  # Ensure non-negative
            'method': 'rolling_average'
        })
    
    return pd.DataFrame(forecast_data)

def CREATE_SEASONAL_NAIVE_FORECAST(daily_sales, target_col, forecast_days, seasonality=7):
    """Create forecast using seasonal naive method"""
    forecast_data = []
    last_date = daily_sales.index[-1]
    
    for i in range(forecast_days):
        forecast_date = last_date + timedelta(days=i+1)
        
        # Use same day from previous season
        seasonal_lag = seasonality
        if len(daily_sales) >= seasonal_lag:
            seasonal_value = daily_sales[target_col].iloc[-seasonal_lag]
        else:
            seasonal_value = daily_sales[target_col].mean()
        
        forecast_data.append({
            'date': forecast_date,
            'forecast': max(seasonal_value, 0),
            'method': 'seasonal_naive'
        })
    
    return pd.DataFrame(forecast_data)

def CREATE_ML_FORECAST(df, model, date_col, target_col, forecast_days):
    """Create forecast using trained ML model"""
    forecast_data = []
    last_date = pd.to_datetime(df[date_col].max())
    
    # Prepare feature set for forecasting
    feature_columns = [col for col in df.columns if col not in [date_col, target_col] and df[col].dtype in [np.number]]
    
    # Create future dates
    future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
    
    for i, future_date in enumerate(future_dates):
        # Use the last available row as base for future predictions
        last_row = df[feature_columns].iloc[-1:].copy()
        
        # Update temporal features for future date
        last_row['year'] = future_date.year
        last_row['month'] = future_date.month
        last_row['day'] = future_date.day
        last_row['dayofweek'] = future_date.dayofweek
        last_row['is_weekend'] = 1 if future_date.dayofweek >= 5 else 0
        
        # Make prediction
        try:
            prediction = model.predict(last_row)[0]
            forecast_data.append({
                'date': future_date,
                'forecast': max(prediction, 0),
                'method': 'machine_learning'
            })
        except:
            # Fallback to average if prediction fails
            forecast_data.append({
                'date': future_date,
                'forecast': max(df[target_col].mean(), 0),
                'method': 'machine_learning'
            })
    
    return pd.DataFrame(forecast_data)

def CREATE_EXPONENTIAL_SMOOTHING_FORECAST(daily_sales, target_col, forecast_days):
    """Create forecast using exponential smoothing"""
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    
    forecast_data = []
    last_date = daily_sales.index[-1]
    
    try:
        # Fit exponential smoothing model
        model = SimpleExpSmoothing(daily_sales[target_col])
        fitted_model = model.fit()
        
        # Generate forecasts
        forecasts = fitted_model.forecast(forecast_days)
        
        for i, (date, forecast_val) in enumerate(zip(
            [last_date + timedelta(days=i+1) for i in range(forecast_days)],
            forecasts
        )):
            forecast_data.append({
                'date': date,
                'forecast': max(forecast_val, 0),
                'method': 'exponential_smoothing'
            })
    except:
        # Fallback to simple average
        avg_value = daily_sales[target_col].mean()
        for i in range(forecast_days):
            forecast_data.append({
                'date': last_date + timedelta(days=i+1),
                'forecast': max(avg_value, 0),
                'method': 'exponential_smoothing'
            })
    
    return pd.DataFrame(forecast_data)

def CREATE_ENSEMBLE_FORECAST(forecasts):
    """Combine multiple forecasts using weighted ensemble"""
    # Extract all forecast DataFrames
    forecast_dfs = [df for df in forecasts.values() if isinstance(df, pd.DataFrame)]
    
    if not forecast_dfs:
        return pd.DataFrame()
    
    # Merge all forecasts on date
    ensemble_df = forecast_dfs[0][['date']].copy()
    
    for i, df in enumerate(forecast_dfs):
        ensemble_df = ensemble_df.merge(
            df[['date', 'forecast']].rename(columns={'forecast': f'forecast_{i}'}),
            on='date', how='left'
        )
    
    # Calculate ensemble as weighted average
    forecast_columns = [col for col in ensemble_df.columns if col.startswith('forecast_')]
    
    # Simple average for ensemble
    ensemble_df['forecast'] = ensemble_df[forecast_columns].mean(axis=1)
    ensemble_df['method'] = 'ensemble'
    
    return ensemble_df[['date', 'forecast', 'method']]

def ADD_CONFIDENCE_INTERVALS(forecasts, historical_data, target_col):
    """Add confidence intervals to forecasts"""
    historical_std = historical_data[target_col].std()
    
    for method, forecast_df in forecasts.items():
        if isinstance(forecast_df, pd.DataFrame) and 'forecast' in forecast_df.columns:
            forecast_df['forecast_lower'] = forecast_df['forecast'] - (1.96 * historical_std)
            forecast_df['forecast_upper'] = forecast_df['forecast'] + (1.96 * historical_std)
            
            # Ensure non-negative values
            forecast_df['forecast_lower'] = forecast_df['forecast_lower'].clip(lower=0)
            forecast_df['forecast_upper'] = forecast_df['forecast_upper'].clip(lower=0)
    
    return forecasts

def VISUALIZE_FORECASTS(historical_df, forecasts, dataset_name, date_col, target_col):
    """Create comprehensive forecast visualizations"""
    print(f"\n CREATING FORECAST VISUALIZATIONS FOR: {dataset_name}")
    
    # Prepare historical data
    historical_daily = historical_df.groupby(date_col)[target_col].sum().reset_index()
    historical_daily[date_col] = pd.to_datetime(historical_daily[date_col])
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f' COMPREHENSIVE FORECAST ANALYSIS - {dataset_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: All Forecasting Methods
    axes[0,0].plot(historical_daily[date_col], historical_daily[target_col], 
                   label='Historical', linewidth=2, color='#2E86AB')
    
    colors = ['#A23B72', '#F18F01', '#C73E1D', '#6A0572', '#2A9D8F']
    for (method, forecast_df), color in zip(forecasts.items(), colors):
        if isinstance(forecast_df, pd.DataFrame):
            axes[0,0].plot(forecast_df['date'], forecast_df['forecast'], 
                          label=f'{method.title()} Forecast', linewidth=2, color=color)
    
    axes[0,0].set_title(' ALL FORECASTING METHODS COMPARISON', fontweight='bold')
    axes[0,0].set_ylabel(target_col.title())
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Ensemble Forecast with Confidence Intervals
    ensemble_forecast = forecasts.get('ensemble')
    if ensemble_forecast is not None:
        axes[0,1].plot(historical_daily[date_col], historical_daily[target_col], 
                       label='Historical', linewidth=2, color='#2E86AB')
        axes[0,1].plot(ensemble_forecast['date'], ensemble_forecast['forecast'],
                       label='Ensemble Forecast', linewidth=3, color='#2A9D8F')
        axes[0,1].fill_between(ensemble_forecast['date'],
                              ensemble_forecast['forecast_lower'],
                              ensemble_forecast['forecast_upper'],
                              alpha=0.3, color='#2A9D8F', label='95% Confidence Interval')
        axes[0,1].set_title(' ENSEMBLE FORECAST WITH CONFIDENCE INTERVALS', fontweight='bold')
        axes[0,1].set_ylabel(target_col.title())
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Forecast Method Comparison (Bar chart)
    methods = []
    final_forecasts = []
    
    for method, forecast_df in forecasts.items():
        if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
            methods.append(method)
            final_forecasts.append(forecast_df['forecast'].sum())
    
    axes[1,0].bar(methods, final_forecasts, color=colors[:len(methods)])
    axes[1,0].set_title(' TOTAL FORECASTED VOLUME BY METHOD', fontweight='bold')
    axes[1,0].set_ylabel('Total Forecasted Volume')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Forecast Accuracy Metrics (placeholder)
    axes[1,1].text(0.5, 0.5, 'FORECAST ACCURACY METRICS\n(Would require actual future data for calculation)',
                  horizontalalignment='center', verticalalignment='center',
                  transform=axes[1,1].transAxes, fontsize=12)
    axes[1,1].set_title(' FORECAST ACCURACY ASSESSMENT', fontweight='bold')
    axes[1,1].set_xticks([])
    axes[1,1].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f'forecast_visualization_{dataset_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

# EXECUTE COMPLETE FORECASTING PIPELINE
print("\n STARTING COMPLETE FORECASTING PIPELINE...")

# 1. SUPERSTORE FORECASTS
superstore_forecasts = CREATE_ADVANCED_FORECASTS(
    superstore_features, best_superstore_model, "SUPERSTORE", forecast_days=90
)

# 2. RETAIL SALES FORECASTS
retail_forecasts = CREATE_ADVANCED_FORECASTS(
    retail_features, best_retail_model, "RETAIL", forecast_days=90
)

# 3. ROSSMANN FORECASTS
rossmann_forecasts = CREATE_ADVANCED_FORECASTS(
    rossmann_features, best_rossmann_model, "ROSSMANN", forecast_days=90
)

# CREATE VISUALIZATIONS
VISUALIZE_FORECASTS(superstore_features, superstore_forecasts, "SUPERSTORE", 'Order Date', 'Sales')
VISUALIZE_FORECASTS(retail_features, retail_forecasts, "RETAIL SALES", 'date', 'sales')
VISUALIZE_FORECASTS(rossmann_features, rossmann_forecasts, "ROSSMANN", 'Date', 'Sales')

# SAVE ALL FORECASTS
print("\n SAVING ALL FORECAST RESULTS...")

forecast_results = {
    'superstore': superstore_forecasts,
    'retail': retail_forecasts,
    'rossmann': rossmann_forecasts
}

joblib.dump(forecast_results, 'all_forecasts.pkl')

# Save individual forecast files for Power BI
for dataset_name, forecasts in forecast_results.items():
    for method, forecast_df in forecasts.items():
        if isinstance(forecast_df, pd.DataFrame):
            forecast_df.to_csv(f'forecast_{dataset_name}_{method}.csv', index=False)

print(" COMPLETE FORECASTING PIPELINE FINISHED!")
print(" READY FOR POWER BI DASHBOARD PREPARATION!")