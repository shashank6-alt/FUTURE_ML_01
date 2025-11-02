# FILE: 03_data_analysis.py
# PURPOSE: COMPREHENSIVE DATA ANALYSIS OF ALL DATASETS
import pandas as pd
import numpy as np
print("🔍 COMPREHENSIVE DATA ANALYSIS - MAXIMUM CAPACITY")
print("=" * 65)

# LOAD ALL DATASETS
print("\n LOADING ALL DATASETS...")

# 1. SUPERSTORE DATASET
superstore_df = pd.read_csv('superstore-dataset-final/Sample - Superstore.csv', encoding='latin-1')
print(f" SUPERSTORE LOADED: {superstore_df.shape}")

# 2. RETAIL SALES DATASET  
retail_df = pd.read_csv('retail-sales-forecasting/train.csv')
print(f" RETAIL SALES LOADED: {retail_df.shape}")

# 3. ROSSMANN DATASETS
rossmann_df = pd.read_csv('rossmann-store-sales/train.csv')
store_df = pd.read_csv('rossmann-store-sales/store.csv')
print(f" ROSSMANN TRAIN LOADED: {rossmann_df.shape}")
print(f" ROSSMANN STORE LOADED: {store_df.shape}")

# COMPREHENSIVE DATA ANALYSIS FUNCTION
def MAXIMUM_DATA_ANALYSIS(df, dataset_name):
    print(f"\n{'='*70}")
    print(f" MAXIMUM ANALYSIS: {dataset_name}")
    print(f"{'='*70}")
    
    # BASIC INFORMATION
    print(f" SHAPE: {df.shape}")
    print(f"  COLUMNS: {list(df.columns)}")
    
    # DATA TYPES
    print(f"\n DATA TYPES:")
    print(df.dtypes)
    
    # MISSING VALUES - COMPREHENSIVE
    print(f"\n MISSING VALUES ANALYSIS:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_info = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percent': missing_percent.values
    })
    missing_info = missing_info[missing_info['Missing_Count'] > 0]
    
    if len(missing_info) > 0:
        print(missing_info)
    else:
        print("    NO MISSING VALUES!")
    
    # DUPLICATE ANALYSIS
    duplicates = df.duplicated().sum()
    print(f"\n DUPLICATE ROWS: {duplicates}")
    
    # BASIC STATISTICS FOR NUMERICAL COLUMNS
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print(f"\n NUMERICAL COLUMNS STATISTICS:")
        print(df[numerical_cols].describe())
    
    return missing_info

# RUN ANALYSIS ON ALL DATASETS
print("\n PERFORMING MAXIMUM CAPACITY DATA ANALYSIS...")

superstore_analysis = MAXIMUM_DATA_ANALYSIS(superstore_df, "SUPERSTORE DATASET")
retail_analysis = MAXIMUM_DATA_ANALYSIS(retail_df, "RETAIL SALES DATASET") 
rossmann_analysis = MAXIMUM_DATA_ANALYSIS(rossmann_df, "ROSSMANN TRAIN DATASET")
store_analysis = MAXIMUM_DATA_ANALYSIS(store_df, "ROSSMANN STORE DATASET")

# DATASET-SPECIFIC DETAILED ANALYSIS
print(f"\n{'='*70}")
print(" DATASET-SPECIFIC KEY INSIGHTS")
print(f"{'='*70}")

# SUPERSTORE SPECIFIC ANALYSIS
print(f"\n SUPERSTORE KEY COLUMNS:")
superstore_cols = ['Order Date', 'Ship Date', 'Sales', 'Profit', 'Quantity', 'Category', 'Sub-Category', 'Region']
print(superstore_df[superstore_cols].info())

# RETAIL SALES SPECIFIC ANALYSIS  
print(f"\n RETAIL SALES KEY COLUMNS:")
print(retail_df.info())

# ROSSMANN SPECIFIC ANALYSIS
print(f"\n ROSSMANN KEY COLUMNS:")
print(rossmann_df.info())
print(f"\n ROSSMANN STORE KEY COLUMNS:")
print(store_df.info())

print("\n COMPREHENSIVE DATA ANALYSIS COMPLETED!")
print(" READY FOR ADVANCED VISUALIZATIONS!")