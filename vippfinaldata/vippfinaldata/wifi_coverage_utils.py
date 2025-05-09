import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_wifi_data(wifi_path, nrows=None):
    """
    Load WiFi data with proper data types and handle hidden SSIDs
    """
    print(f"Loading WiFi data from: {wifi_path}")
    
    if nrows:
        wifi_df = pd.read_csv(wifi_path, nrows=nrows)
    else:
        wifi_df = pd.read_csv(wifi_path)
    
    # Convert timestamp to datetime
    wifi_df['timestamp'] = pd.to_datetime(wifi_df['timestamp'], unit='ms')
    
    # Add timestamp_ms for merging
    wifi_df['timestamp_ms'] = wifi_df['timestamp'].astype(int) // 10**6
    
    # Handle hidden SSIDs - replace empty with "__hidden__" 
    if 'ssid' in wifi_df.columns:
        wifi_df['ssid'] = wifi_df['ssid'].replace('', '__hidden__')
    
    # Convert RSSI to numeric
    if 'rssi' in wifi_df.columns:
        wifi_df['rssi'] = pd.to_numeric(wifi_df['rssi'], errors='coerce')
    
    print(f"WiFi data loaded: {len(wifi_df)} records")
    return wifi_df

def load_location_data(location_path, nrows=None):
    """
    Load location data with proper data types
    """
    print(f"Loading location data from: {location_path}")
    
    if nrows:
        location_df = pd.read_csv(location_path, nrows=nrows)
    else:
        location_df = pd.read_csv(location_path)
    
    # Convert timestamp to datetime
    location_df['timestamp'] = pd.to_datetime(location_df['timestamp_ms'], unit='ms')
    
    # Ensure timestamp_ms exists for merging
    if 'timestamp_ms' not in location_df.columns:
        location_df['timestamp_ms'] = location_df['timestamp'].astype(int) // 10**6
    
    print(f"Location data loaded: {len(location_df)} records")
    return location_df

def merge_wifi_and_location(wifi_df, location_df, tolerance_seconds=300):
    """
    Merge WiFi and location data with proper timestamp handling and validation
    
    Parameters:
    -----------
    wifi_df : DataFrame
        WiFi data with timestamp column
    location_df : DataFrame
        Location data with timestamp column
    tolerance_seconds : int
        Time tolerance in seconds for merge_asof (default increased to 300 seconds
        based on time difference analysis of SHL dataset)
    
    Returns:
    --------
    merged_df : DataFrame
        Merged dataframe with WiFi and location data
    """
    print("Merging WiFi and location data...")
    print(f"Using timestamp tolerance of {tolerance_seconds} seconds")
    
    # Ensure dataframes are sorted by timestamp
    wifi_df = wifi_df.sort_values('timestamp')
    location_df = location_df.sort_values('timestamp')
    
    try:
        # Use merge_asof with proper tolerance as pd.Timedelta
        merged_df = pd.merge_asof(
            wifi_df,
            location_df,
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(seconds=tolerance_seconds)
        )
        
        # Verify merge quality
        join_rate = merged_df['latitude_deg'].notna().mean()
        print(f"Join rate: {join_rate:.2%} of records have location data")
        
        if join_rate < 0.90:
            print("WARNING: Location join is sparse (<90% match rate)")
        
        # Require at least 50% join rate
        assert join_rate > 0.50, "Location join too sparse (<50% match rate). Aborting pipeline."
        
    except Exception as e:
        print(f"Error in merge_asof: {e}. Falling back to regular merge...")
        
        # Fall back to regular merge on timestamp_ms with a time window
        # Create time windows for matching
        wifi_df['time_window'] = (wifi_df['timestamp_ms'] // (tolerance_seconds * 1000)) * (tolerance_seconds * 1000)
        location_df['time_window'] = (location_df['timestamp_ms'] // (tolerance_seconds * 1000)) * (tolerance_seconds * 1000)
        
        # Merge on time window
        merged_df = pd.merge(
            wifi_df,
            location_df,
            on='time_window',
            how='inner'
        )
        
        # If we still have multiple location records per WiFi record, take the closest one
        if len(merged_df) > len(wifi_df):
            print("Multiple location matches found, keeping closest by timestamp...")
            merged_df['time_diff'] = abs(merged_df['timestamp_ms_x'] - merged_df['timestamp_ms_y'])
            merged_df = merged_df.sort_values('time_diff').groupby('timestamp_ms_x').first().reset_index()
        
        join_rate = len(merged_df) / len(wifi_df)
        print(f"Fallback join rate: {join_rate:.2%}")
        
        # Still require minimum join rate
        assert join_rate > 0.50, "Location join too sparse (<50% match rate). Aborting pipeline."
    
    # Sort for time-series operations
    merged_df = merged_df.sort_values(['bssid', 'timestamp'])
    
    return merged_df

def add_engineered_features(merged_df):
    """
    Add engineered features to the merged dataset
    """
    print("Adding engineered features...")
    
    # Calculate RSSI change between consecutive measurements (per BSSID)
    merged_df['rssi_change'] = merged_df.groupby('bssid')['rssi'].diff().fillna(0)
    
    # Calculate rolling standard deviation of RSSI (per BSSID)
    window_size = 5  # 5 consecutive measurements
    merged_df['rssi_rolling_std'] = merged_df.groupby('bssid')['rssi'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).std()
    ).fillna(0)
    
    # Add hour of day for temporal patterns
    merged_df['hour_of_day'] = merged_df['timestamp'].dt.hour
    
    # Add day of week (0=Monday, 6=Sunday)
    merged_df['day_of_week'] = merged_df['timestamp'].dt.dayofweek
    
    return merged_df

def create_grid_cells(merged_df, lat_grid_size=0.0002, lon_grid_size=0.0002):
    """
    Create grid cells for spatial analysis
    Approx grid cell size of 0.0002 degrees is about 25 meters
    """
    print("Creating spatial grid cells...")
    
    # Create grid cells by rounding coordinates
    merged_df['lat_grid'] = (merged_df['latitude_deg'] // lat_grid_size) * lat_grid_size
    merged_df['lon_grid'] = (merged_df['longitude_deg'] // lon_grid_size) * lon_grid_size
    
    # Create grid_id for grid-based cross validation to prevent spatial leakage
    merged_df['grid_id'] = merged_df['lat_grid'].astype(str) + '_' + merged_df['lon_grid'].astype(str)
    
    # Extract date for temporal validation
    merged_df['date'] = merged_df['timestamp'].dt.date
    
    return merged_df

def calculate_grid_statistics(merged_df, rssi_threshold=-75):
    """
    Calculate statistics for each grid cell and BSSID combination
    """
    print("Calculating grid statistics...")
    
    # Aggregate by grid cell and BSSID
    grid_stats = merged_df.groupby(['lat_grid', 'lon_grid', 'bssid']).agg({
        'rssi': ['mean', 'std', 'min', 'max', 'count'],
        'rssi_change': ['mean', 'std'],
        'rssi_rolling_std': ['mean'],
        'grid_id': 'first',  # Keep grid_id for spatial cross-validation
        'date': 'first'      # Keep date for temporal cross-validation
    }).reset_index()
    
    # Flatten column names
    grid_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in grid_stats.columns]
    
    # Verify grid_id column is present
    if 'grid_id' not in grid_stats.columns and 'grid_id_first' in grid_stats.columns:
        # Rename the column if it was flattened with a suffix
        grid_stats = grid_stats.rename(columns={'grid_id_first': 'grid_id'})
    
    # Create grid_id if it doesn't exist
    if 'grid_id' not in grid_stats.columns:
        print("Creating grid_id column from lat_grid and lon_grid...")
        grid_stats['grid_id'] = grid_stats['lat_grid'].astype(str) + '_' + grid_stats['lon_grid'].astype(str)
    
    # Handle any NaN values in standard deviation columns (single measurements)
    grid_stats['rssi_std'] = grid_stats['rssi_std'].fillna(0)
    grid_stats['rssi_change_std'] = grid_stats['rssi_change_std'].fillna(0)
    
    # Define low coverage areas based on threshold
    grid_stats['low_coverage_area'] = (grid_stats['rssi_mean'] < rssi_threshold).astype(int)
    
    # Add hourly variation if we have enough time range
    hour_counts = merged_df['hour_of_day'].nunique()
    if hour_counts > 3:  # Only if we have at least 3 different hours
        print(f"Adding hourly variations across {hour_counts} hours...")
        hourly_stats = merged_df.groupby(['lat_grid', 'lon_grid', 'bssid', 'hour_of_day']).agg({
            'rssi': ['mean', 'std']
        }).reset_index()
        
        # Rename columns
        hourly_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in hourly_stats.columns]
        
        # Calculate hourly variation per grid/bssid
        hourly_var = hourly_stats.groupby(['lat_grid', 'lon_grid', 'bssid']).agg({
            'rssi_mean': ['std', 'max', 'min']
        }).reset_index()
        
        # Rename columns
        hourly_var.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in hourly_var.columns]
        
        # Rename to more clear names
        hourly_var = hourly_var.rename(columns={
            'rssi_mean_std': 'hourly_variation',
            'rssi_mean_max': 'hourly_max',
            'rssi_mean_min': 'hourly_min'
        })
        
        # Merge with grid stats
        grid_stats = pd.merge(
            grid_stats,
            hourly_var[['lat_grid', 'lon_grid', 'bssid', 'hourly_variation', 'hourly_max', 'hourly_min']],
            on=['lat_grid', 'lon_grid', 'bssid'],
            how='left'
        )
        
        # Fill missing values
        grid_stats['hourly_variation'] = grid_stats['hourly_variation'].fillna(0)
    else:
        print("Not enough time diversity for hourly variations.")
        grid_stats['hourly_variation'] = 0
        grid_stats['hourly_max'] = grid_stats['rssi_max']
        grid_stats['hourly_min'] = grid_stats['rssi_min']
    
    # Make sure the grid_id and date columns exist
    if 'grid_id' not in grid_stats.columns:
        print("WARNING: Creating grid_id column from lat_grid and lon_grid...")
        grid_stats['grid_id'] = grid_stats['lat_grid'].astype(str) + '_' + grid_stats['lon_grid'].astype(str)
        
    if 'date' not in grid_stats.columns and 'date_first' in grid_stats.columns:
        grid_stats = grid_stats.rename(columns={'date_first': 'date'})
    
    # Print available columns for debugging
    print(f"Grid statistics columns: {grid_stats.columns.tolist()}")
    
    print(f"Created statistics for {len(grid_stats)} grid/BSSID combinations")
    return grid_stats

def apply_hampel_filter(series, window_size=5, threshold=3):
    """
    Apply Hampel filter to remove outliers
    A more robust outlier detection than simple thresholding
    """
    # Calculate rolling median and standard deviation
    rolling_median = series.rolling(window=window_size, center=True).median()
    rolling_std = series.rolling(window=window_size, center=True).std()
    
    # Replace std of 0 with a small value to avoid division by zero
    rolling_std = rolling_std.replace(0, 1e-6)
    
    # Identify outliers
    diff = (series - rolling_median).abs()
    outlier_idx = diff > (threshold * rolling_std)
    
    # Replace outliers with median
    return series.mask(outlier_idx, rolling_median)

def clean_signal_data(merged_df):
    """
    Clean signal data using Hampel filter for each BSSID
    """
    print("Cleaning signal data using Hampel filter...")
    
    # Apply Hampel filter to RSSI values grouped by BSSID
    merged_df['rssi'] = merged_df.groupby('bssid')['rssi'].transform(
        lambda x: apply_hampel_filter(x, window_size=5, threshold=3)
    )
    
    return merged_df 