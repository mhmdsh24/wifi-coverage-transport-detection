import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import warnings
from scipy import stats
import datetime
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

def load_data(file_path, sample_size=None):
    """
    Load data from CSV files with option to sample
    
    Args:
        file_path: Path to the CSV file
        sample_size: Number of rows to sample (None for all)
        
    Returns:
        DataFrame with the loaded data
    """
    try:
        if sample_size:
            df = pd.read_csv(file_path, nrows=sample_size)
        else:
            df = pd.read_csv(file_path)
        print(f"Successfully loaded {os.path.basename(file_path)}, shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {os.path.basename(file_path)}: {e}")
        return None

def analyze_dataframe(df, name):
    """
    Perform basic analysis on a dataframe
    
    Args:
        df: DataFrame to analyze
        name: Name of the dataset for printing
    """
    if df is None:
        print(f"Cannot analyze {name} - DataFrame is None")
        return
    
    print(f"\n{'='*50}")
    print(f"Analysis of {name} Dataset")
    print(f"{'='*50}")
    
    # Basic info
    print(f"\nShape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    
    # Data types
    print("\nData types:")
    print(df.dtypes)
    
    # Missing values
    missing = df.isnull().sum()
    print("\nMissing values:")
    print(missing[missing > 0] if len(missing[missing > 0]) > 0 else "No missing values")
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe().T)
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")
    
    return

def plot_distributions(df, name, columns=None, save_dir='plots'):
    """
    Plot distributions of numeric columns
    
    Args:
        df: DataFrame with data
        name: Name of the dataset for saving plots
        columns: List of columns to plot (None for all numeric)
        save_dir: Directory to save plots
    """
    if df is None:
        print(f"Cannot plot distributions for {name} - DataFrame is None")
        return
    
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Create distribution plots
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            plt.figure(figsize=(10, 6))
            
            # Plot histogram with KDE
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col} in {name}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{name}_{col}_distribution.png")
            plt.close()
            
            # Create boxplot to identify outliers
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[col].dropna())
            plt.title(f'Boxplot of {col} in {name}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{name}_{col}_boxplot.png")
            plt.close()
    
    print(f"Distribution plots for {name} saved to {save_dir}/")

def plot_correlations(df, name, save_dir='plots'):
    """
    Plot correlation matrix for numeric columns
    
    Args:
        df: DataFrame with data
        name: Name of the dataset for saving plot
        save_dir: Directory to save plots
    """
    if df is None:
        print(f"Cannot plot correlations for {name} - DataFrame is None")
        return
    
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.shape[1] < 2:
        print(f"Not enough numeric columns in {name} to plot correlations")
        return
    
    # Compute correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f")
    plt.title(f'Correlation Matrix for {name}')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}_correlation_matrix.png")
    plt.close()
    
    print(f"Correlation matrix for {name} saved to {save_dir}/")

def handle_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Handle outliers in the data
    
    Args:
        df: DataFrame with data
        columns: List of columns to handle outliers for (None for all numeric)
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers handled
    """
    if df is None:
        return None
    
    # Create a copy of the dataframe
    df_clean = df.copy()
    
    # Get numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    outliers_summary = {}
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            # Get original number of values
            original_count = df_clean[col].count()
            
            if method == 'iqr':
                # IQR method
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Replace outliers with NaN
                df_clean.loc[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), col] = np.nan
                
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                
                # Replace outliers with NaN
                df_clean.loc[z_scores > threshold, col] = np.nan
            
            # Record how many outliers were found
            new_count = df_clean[col].count()
            outliers_found = original_count - new_count
            outliers_summary[col] = outliers_found
    
    # Print summary
    if sum(outliers_summary.values()) > 0:
        print("\nOutliers handled (replaced with NaN):")
        for col, count in outliers_summary.items():
            if count > 0:
                print(f"{col}: {count} outliers ({count/original_count*100:.2f}%)")
    else:
        print("\nNo outliers found based on the specified criteria.")
    
    return df_clean

def handle_missing_values(df, strategy='auto'):
    """
    Handle missing values in the data
    
    Args:
        df: DataFrame with data
        strategy: Strategy to use ('mean', 'median', 'mode', 'auto')
        
    Returns:
        DataFrame with missing values handled
    """
    if df is None:
        return None
    
    # Create a copy of the dataframe
    df_filled = df.copy()
    
    # Check for missing values
    missing = df_filled.isnull().sum()
    missing_cols = missing[missing > 0]
    
    if len(missing_cols) == 0:
        print("No missing values found.")
        return df_filled
    
    print("\nHandling missing values:")
    for col in missing_cols.index:
        missing_count = missing_cols[col]
        col_dtype = df_filled[col].dtype
        
        print(f"{col}: {missing_count} missing values", end=" - ")
        
        if col_dtype in ['int64', 'float64']:
            # For numeric columns
            if strategy == 'mean' or (strategy == 'auto' and missing_count / len(df_filled) < 0.05):
                fill_value = df_filled[col].mean()
                method = 'mean'
            elif strategy == 'median' or (strategy == 'auto' and missing_count / len(df_filled) >= 0.05):
                fill_value = df_filled[col].median()
                method = 'median'
            else:
                fill_value = df_filled[col].median()
                method = 'median'
            
            df_filled[col].fillna(fill_value, inplace=True)
            print(f"filled with {method} ({fill_value:.2f})")
            
        elif col_dtype == 'object':
            # For categorical columns
            fill_value = df_filled[col].mode()[0]
            df_filled[col].fillna(fill_value, inplace=True)
            print(f"filled with mode ({fill_value})")
        
        else:
            # For other types, use a generic approach
            fill_value = df_filled[col].mode()[0] if len(df_filled[col].dropna()) > 0 else "UNKNOWN"
            df_filled[col].fillna(fill_value, inplace=True)
            print(f"filled with most common value ({fill_value})")
    
    return df_filled

def normalize_data(df, columns=None, method='standard'):
    """
    Normalize numeric data
    
    Args:
        df: DataFrame with data
        columns: List of columns to normalize (None for all numeric)
        method: Method to use ('standard', 'minmax')
        
    Returns:
        DataFrame with normalized data and the scaler used
    """
    if df is None:
        return None, None
    
    # Create a copy of the dataframe
    df_norm = df.copy()
    
    # Get numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Filter to only include columns that exist
    columns = [col for col in columns if col in df.columns]
    
    if len(columns) == 0:
        print("No numeric columns to normalize.")
        return df_norm, None
    
    # Select the scaler based on the method
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        print(f"Unknown normalization method '{method}', using StandardScaler.")
        scaler = StandardScaler()
    
    # Fit and transform the data
    df_norm[columns] = scaler.fit_transform(df_norm[columns])
    
    print(f"\nData normalized using {scaler.__class__.__name__} for columns: {columns}")
    
    return df_norm, scaler

def plot_signal_strength_map(location_df, wifi_df, save_dir='plots'):
    """
    Plot a map of signal strength
    
    Args:
        location_df: DataFrame with location data
        wifi_df: DataFrame with WiFi data
        save_dir: Directory to save plots
    """
    if location_df is None or wifi_df is None:
        print("Cannot plot signal strength map - DataFrames are None")
        return
    
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Merge datasets on timestamp
    # First, rename timestamp columns to match
    if 'timestamp_ms' in location_df.columns and 'timestamp' in wifi_df.columns:
        wifi_df_copy = wifi_df.copy()
        wifi_df_copy.rename(columns={'timestamp': 'timestamp_ms'}, inplace=True)
        
        # Create a simple mapping between timestamps using nearest match
        # This is a simplified approach - a more precise matching may be needed
        merged_df = pd.merge_asof(
            location_df.sort_values('timestamp_ms'),
            wifi_df_copy.sort_values('timestamp_ms'),
            on='timestamp_ms',
            direction='nearest'
        )
        
        if len(merged_df) > 0 and 'rssi' in merged_df.columns:
            plt.figure(figsize=(12, 10))
            
            # Create a scatter plot with signal strength
            scatter = plt.scatter(
                merged_df['longitude_deg'], 
                merged_df['latitude_deg'], 
                c=merged_df['rssi'], 
                cmap='RdYlGn',  # Red (weak) to Green (strong)
                alpha=0.7,
                s=50
            )
            
            # Add a colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Signal Strength (RSSI)')
            
            plt.title('WiFi Signal Strength Map')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/signal_strength_map.png")
            plt.close()
            
            print(f"Signal strength map saved to {save_dir}/signal_strength_map.png")
        else:
            print("Could not create signal strength map - missing data after merge")

def load_wifi_location_gps_data(wifi_path='Hips_WiFi.csv', location_path='Hips_Location.csv', gps_path='Hips_GPS.csv', 
              nrows=None, use_cleaned=True):
    """
    Load WiFi, location, and GPS data from CSV files.
    
    Parameters:
    -----------
    wifi_path : str
        Path to WiFi data file
    location_path : str
        Path to location data file
    gps_path : str
        Path to GPS data file
    nrows : int, optional
        Number of rows to load (for testing with smaller datasets)
    use_cleaned : bool, default=True
        Whether to try loading cleaned data files first
        
    Returns:
    --------
    tuple of pandas DataFrames: (wifi_df, location_df, gps_df)
    """
    # Try to load cleaned data first if requested
    if use_cleaned:
        try:
            wifi_df = pd.read_csv('cleaned_wifi_data.csv', nrows=nrows)
            location_df = pd.read_csv('cleaned_location_data.csv', nrows=nrows)
            gps_df = pd.read_csv('cleaned_gps_data.csv', nrows=nrows)
            print("Loaded cleaned data files successfully.")
            return wifi_df, location_df, gps_df
        except FileNotFoundError:
            print("Cleaned data files not found. Loading original data...")
    
    # Load original data
    print(f"Loading data from original files (nrows={nrows})...")
    wifi_df = pd.read_csv(wifi_path, nrows=nrows)
    location_df = pd.read_csv(location_path, nrows=nrows)
    gps_df = pd.read_csv(gps_path, nrows=nrows)
    
    return wifi_df, location_df, gps_df

def clean_wifi_data(wifi_df):
    """
    Clean and prepare WiFi data.
    
    Parameters:
    -----------
    wifi_df : pandas DataFrame
        Raw WiFi data
        
    Returns:
    --------
    pandas DataFrame: Cleaned WiFi data
    """
    print("Cleaning WiFi data...")
    
    # Make a copy to avoid modifying the original
    df = wifi_df.copy()
    
    # Rename timestamp column if needed
    if 'timestamp' in df.columns and 'timestamp_ms' not in df.columns:
        df.rename(columns={'timestamp': 'timestamp_ms'}, inplace=True)
    
    # Ensure timestamp is numeric
    df['timestamp_ms'] = pd.to_numeric(df['timestamp_ms'])
    
    # Add datetime column
    df['timestamp_dt'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    
    # Drop duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows from WiFi data.")
    
    # Handle missing values
    df = df.dropna(subset=['bssid', 'rssi'])
    
    # Filter out unrealistic RSSI values (typically between -100 and 0 dBm)
    df = df[(df['rssi'] >= -100) & (df['rssi'] <= 0)]
    
    # Add signal quality categories
    df['signal_quality'] = pd.cut(
        df['rssi'],
        bins=[-100, -85, -70, -55, 0],
        labels=['Very Poor', 'Poor', 'Good', 'Excellent']
    )
    
    # Add temporal features
    df['hour'] = df['timestamp_dt'].dt.hour
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
    
    # Detect potential outliers in RSSI
    z_scores = stats.zscore(df['rssi'])
    abs_z_scores = np.abs(z_scores)
    df['rssi_z_score'] = z_scores
    df['rssi_outlier'] = abs_z_scores > 3  # Flag RSSI values with z-score > 3
    
    # Add signal stability metrics (per BSSID)
    df = df.sort_values(['bssid', 'timestamp_ms'])
    
    # Calculate signal rate of change
    df['rssi_change'] = df.groupby('bssid')['rssi'].diff()
    
    # Calculate rolling statistics in a window
    window_size = 5
    df['rssi_rolling_mean'] = df.groupby('bssid')['rssi'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    df['rssi_rolling_std'] = df.groupby('bssid')['rssi'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).std()
    )
    
    return df

def clean_location_data(location_df):
    """
    Clean and prepare location data.
    
    Parameters:
    -----------
    location_df : pandas DataFrame
        Raw location data
        
    Returns:
    --------
    pandas DataFrame: Cleaned location data
    """
    print("Cleaning location data...")
    
    # Make a copy to avoid modifying the original
    df = location_df.copy()
    
    # Ensure timestamp is numeric
    df['timestamp_ms'] = pd.to_numeric(df['timestamp_ms'])
    
    # Add datetime column
    df['timestamp_dt'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    
    # Drop duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows from location data.")
    
    # Handle missing or invalid coordinates
    df = df.dropna(subset=['latitude_deg', 'longitude_deg'])
    
    # Filter out unrealistic coordinates (basic sanity check)
    df = df[(df['latitude_deg'] >= -90) & (df['latitude_deg'] <= 90)]
    df = df[(df['longitude_deg'] >= -180) & (df['longitude_deg'] <= 180)]
    
    # Detect outliers in coordinates using z-scores
    for col in ['latitude_deg', 'longitude_deg']:
        z_scores = stats.zscore(df[col])
        abs_z_scores = np.abs(z_scores)
        df[f'{col}_z_score'] = z_scores
        df[f'{col}_outlier'] = abs_z_scores > 3
    
    # Add temporal features
    df['hour'] = df['timestamp_dt'].dt.hour
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
    
    # Calculate speed between consecutive points
    df = df.sort_values('timestamp_ms')
    
    # Function to calculate haversine distance
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points on earth (in meters)"""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371000  # Radius of earth in meters
        
        return c * r
    
    # Calculate time differences and distances between consecutive points
    df['time_diff'] = df['timestamp_ms'].diff()
    df['lat_prev'] = df['latitude_deg'].shift(1)
    df['lon_prev'] = df['longitude_deg'].shift(1)
    
    # Apply haversine formula where we have previous coordinates
    mask = ~df['lat_prev'].isna()
    df.loc[mask, 'distance'] = df.loc[mask].apply(
        lambda row: haversine_distance(
            row['lat_prev'], row['lon_prev'], 
            row['latitude_deg'], row['longitude_deg']
        ), 
        axis=1
    )
    
    # Calculate speed in m/s and km/h
    df['time_diff_sec'] = df['time_diff'] / 1000  # Convert ms to seconds
    df['speed_ms'] = df['distance'] / df['time_diff_sec']
    df['speed_kmh'] = df['speed_ms'] * 3.6
    
    # Filter out unrealistic speeds (e.g., > 200 km/h might be GPS errors)
    df.loc[df['speed_kmh'] > 200, 'speed_kmh'] = np.nan
    df.loc[df['speed_ms'] > 55, 'speed_ms'] = np.nan
    
    return df

def clean_gps_data(gps_df):
    """
    Clean and prepare GPS data.
    
    Parameters:
    -----------
    gps_df : pandas DataFrame
        Raw GPS data
        
    Returns:
    --------
    pandas DataFrame: Cleaned GPS data
    """
    print("Cleaning GPS data...")
    
    # Make a copy to avoid modifying the original
    df = gps_df.copy()
    
    # Ensure timestamp is numeric
    df['timestamp_ms'] = pd.to_numeric(df['timestamp_ms'])
    
    # Add datetime column
    df['timestamp_dt'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    
    # Drop duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows from GPS data.")
    
    # Handle missing values in essential columns
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # Rename columns for consistency if needed
    if 'latitude' in df.columns and 'latitude_deg' not in df.columns:
        df.rename(columns={'latitude': 'latitude_deg', 'longitude': 'longitude_deg'}, inplace=True)
    
    # Add temporal features
    df['hour'] = df['timestamp_dt'].dt.hour
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
    
    return df

def merge_wifi_location_data(wifi_df, location_df, tolerance_ms=5000):
    """
    Merge WiFi and location data based on timestamps.
    
    Parameters:
    -----------
    wifi_df : pandas DataFrame
        Cleaned WiFi data
    location_df : pandas DataFrame
        Cleaned location data
    tolerance_ms : int, default=5000
        Time tolerance in milliseconds for merging
        
    Returns:
    --------
    pandas DataFrame: Merged data
    """
    print(f"Merging WiFi and location data with {tolerance_ms}ms tolerance...")
    
    # Ensure both dataframes are sorted by timestamp
    wifi_df = wifi_df.sort_values('timestamp_ms')
    location_df = location_df.sort_values('timestamp_ms')
    
    # Merge datasets using merge_asof
    merged_df = pd.merge_asof(
        wifi_df,
        location_df,
        on='timestamp_ms',
        direction='nearest',
        tolerance=tolerance_ms
    )
    
    # Drop rows where the merge didn't work (no location data within tolerance)
    merged_df = merged_df.dropna(subset=['latitude_deg', 'longitude_deg'])
    
    # Create grid cells for spatial aggregation
    grid_precision = 4
    merged_df['lat_grid'] = np.round(merged_df['latitude_deg'], grid_precision)
    merged_df['lon_grid'] = np.round(merged_df['longitude_deg'], grid_precision)
    
    print(f"Merged data shape: {merged_df.shape}")
    
    return merged_df

def compute_grid_statistics(merged_df, rssi_threshold=-75):
    """
    Compute aggregated statistics per grid cell for coverage analysis.
    
    Parameters:
    -----------
    merged_df : pandas DataFrame
        Merged WiFi and location data
    rssi_threshold : float, default=-75
        RSSI threshold for defining low coverage
        
    Returns:
    --------
    pandas DataFrame: Grid cell statistics
    """
    print("Computing grid statistics...")
    
    # Create low coverage flag
    merged_df['low_coverage'] = (merged_df['rssi'] < rssi_threshold).astype(int)
    
    # Aggregate by grid cells and BSSID
    grid_stats = merged_df.groupby(['lat_grid', 'lon_grid', 'bssid']).agg({
        'rssi': ['mean', 'std', 'min', 'max', 'count'],
        'low_coverage': 'mean',
        'rssi_change': ['mean', 'std'],
        'rssi_rolling_std': 'mean'
    }).reset_index()
    
    # Flatten the hierarchical column names
    grid_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in grid_stats.columns]
    
    # Create binary target (more than 50% measurements are low coverage)
    grid_stats['low_coverage_area'] = (grid_stats['low_coverage_mean'] > 0.5).astype(int)
    
    # Add extra stability indicators
    grid_stats['signal_stability'] = 1 / (grid_stats['rssi_std'] + 0.1)  # Higher values = more stable
    grid_stats['coverage_confidence'] = grid_stats['rssi_count'] / (grid_stats['rssi_std'] + 0.1)  # Higher values = more confident
    
    # Add a measure of temporal variability
    hourly_stats = merged_df.groupby(['lat_grid', 'lon_grid', 'bssid', 'hour']).agg({
        'rssi': ['mean', 'std', 'count']
    }).reset_index()
    
    hourly_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in hourly_stats.columns]
    
    # Calculate hourly variation
    hour_variation = hourly_stats.groupby(['lat_grid', 'lon_grid', 'bssid']).agg({
        'rssi_mean': ['std']
    }).reset_index()
    
    hour_variation.columns = ['lat_grid', 'lon_grid', 'bssid', 'hourly_variation']
    
    # Merge hourly variation into grid_stats
    grid_stats = pd.merge(
        grid_stats, 
        hour_variation, 
        on=['lat_grid', 'lon_grid', 'bssid'], 
        how='left'
    )
    
    # Identify potential anomalies in the grid statistics
    for col in ['rssi_mean', 'rssi_std', 'hourly_variation']:
        if col in grid_stats.columns:
            z_scores = stats.zscore(grid_stats[col].fillna(grid_stats[col].mean()))
            grid_stats[f'{col}_z_score'] = z_scores
            grid_stats[f'{col}_outlier'] = np.abs(z_scores) > 3
    
    # Calculate potential anomaly score (higher = more unusual behavior)
    grid_stats['grid_anomaly_score'] = (
        np.abs(grid_stats.get('rssi_mean_z_score', 0)) + 
        np.abs(grid_stats.get('rssi_std_z_score', 0)) + 
        np.abs(grid_stats.get('hourly_variation_z_score', 0))
    ) / 3.0
    
    return grid_stats

def visualize_coverage(merged_df, grid_stats, rssi_threshold=-75, save_path='plots'):
    """
    Create visualizations for coverage analysis.
    
    Parameters:
    -----------
    merged_df : pandas DataFrame
        Merged WiFi and location data
    grid_stats : pandas DataFrame
        Grid cell statistics
    rssi_threshold : float, default=-75
        RSSI threshold for defining low coverage
    save_path : str, default='plots'
        Directory to save plots
        
    Returns:
    --------
    None
    """
    print("Creating coverage visualizations...")
    
    os.makedirs(save_path, exist_ok=True)
    
    # Plot 1: Overall RSSI distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(merged_df['rssi'], bins=50, kde=True)
    plt.axvline(x=rssi_threshold, color='r', linestyle='--')
    plt.text(rssi_threshold-5, plt.ylim()[1]*0.9, f'Threshold: {rssi_threshold} dBm', color='r')
    plt.title('RSSI Distribution')
    plt.xlabel('RSSI (dBm)')
    plt.ylabel('Frequency')
    plt.savefig(f'{save_path}/rssi_distribution.png')
    plt.close()
    
    # Plot 2: Spatial distribution of signal strength
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        merged_df['longitude_deg'],
        merged_df['latitude_deg'],
        c=merged_df['rssi'],
        cmap='RdYlGn',
        alpha=0.6,
        s=10
    )
    plt.colorbar(scatter, label='RSSI (dBm)')
    plt.title('Spatial Distribution of Signal Strength')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/spatial_signal_strength.png')
    plt.close()
    
    # Plot 3: Coverage areas
    plt.figure(figsize=(12, 10))
    
    # Separate low coverage and good coverage points
    low_coverage = grid_stats[grid_stats['low_coverage_area'] == 1]
    good_coverage = grid_stats[grid_stats['low_coverage_area'] == 0]
    
    plt.scatter(
        good_coverage['lon_grid'],
        good_coverage['lat_grid'],
        c='green',
        marker='^',
        alpha=0.6,
        label='Good Coverage'
    )
    plt.scatter(
        low_coverage['lon_grid'],
        low_coverage['lat_grid'],
        c='red',
        marker='o',
        alpha=0.6,
        label='Low Coverage'
    )
    
    plt.title('Coverage Quality by Grid Cell')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/coverage_areas.png')
    plt.close()
    
    # Plot 4: Hourly RSSI patterns
    plt.figure(figsize=(14, 7))
    hourly_rssi = merged_df.groupby('hour')['rssi'].mean()
    hourly_rssi_std = merged_df.groupby('hour')['rssi'].std()
    
    plt.errorbar(
        hourly_rssi.index,
        hourly_rssi.values,
        yerr=hourly_rssi_std.values,
        fmt='o-',
        capsize=5
    )
    plt.axhline(y=rssi_threshold, color='r', linestyle='--')
    plt.title('Average RSSI by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average RSSI (dBm)')
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/hourly_rssi_pattern.png')
    plt.close()
    
    # Plot 5: Signal stability visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(
        grid_stats['rssi_mean'],
        grid_stats['rssi_std'],
        c=grid_stats['low_coverage_area'],
        cmap='coolwarm',
        alpha=0.7,
        s=grid_stats['rssi_count'] / 10
    )
    plt.colorbar(label='Low Coverage Area')
    plt.axvline(x=rssi_threshold, color='r', linestyle='--')
    plt.title('Signal Strength vs. Variability')
    plt.xlabel('Mean RSSI (dBm)')
    plt.ylabel('RSSI Standard Deviation')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/signal_stability.png')
    plt.close()
    
    # Plot 6: Grid anomaly scores visualization (potential anomalies)
    if 'grid_anomaly_score' in grid_stats.columns:
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            grid_stats['lon_grid'],
            grid_stats['lat_grid'],
            c=grid_stats['grid_anomaly_score'],
            cmap='viridis',
            alpha=0.7,
            s=30
        )
        plt.colorbar(scatter, label='Grid Anomaly Score')
        plt.title('Potential Anomalous Signal Behavior by Grid Cell')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_path}/grid_anomaly_scores.png')
        plt.close()

# Prepare data for anomaly detection - helper function for integration
def prepare_for_anomaly_detection(merged_df, rssi_threshold=-75):
    """
    Prepare merged data for anomaly detection analysis.
    
    Parameters:
    -----------
    merged_df : pandas DataFrame
        Merged WiFi and location data
    rssi_threshold : float, default=-75
        RSSI threshold for defining low coverage
        
    Returns:
    --------
    pandas DataFrame: Data prepared for anomaly detection
    """
    print("Preparing data for anomaly detection...")
    
    # Create a copy to avoid modifying the original
    df = merged_df.copy()
    
    # Ensure we have the necessary temporal features
    if 'timestamp_dt' not in df.columns and 'timestamp_ms' in df.columns:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp_dt'].dt.hour
    
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
    
    # Add or calculate signal variation features if not present
    if 'rssi_change' not in df.columns:
        df = df.sort_values(['bssid', 'timestamp_ms'])
        df['rssi_change'] = df.groupby('bssid')['rssi'].diff()
    
    if 'rssi_rolling_std' not in df.columns:
        window_size = 5
        df['rssi_rolling_std'] = df.groupby('bssid')['rssi'].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).std()
        )
    
    # Flag low coverage for reference
    df['low_coverage'] = (df['rssi'] < rssi_threshold).astype(int)
    
    return df

# Function to save cleaned and processed data
def save_processed_data(wifi_df, location_df, gps_df, merged_df=None, grid_stats=None):
    """
    Save processed dataframes to CSV files.
    
    Parameters:
    -----------
    wifi_df : pandas DataFrame
        Cleaned WiFi data
    location_df : pandas DataFrame
        Cleaned location data
    gps_df : pandas DataFrame
        Cleaned GPS data
    merged_df : pandas DataFrame, optional
        Merged WiFi and location data
    grid_stats : pandas DataFrame, optional
        Grid cell statistics
        
    Returns:
    --------
    None
    """
    print("Saving processed data to CSV files...")
    
    wifi_df.to_csv('cleaned_wifi_data.csv', index=False)
    location_df.to_csv('cleaned_location_data.csv', index=False)
    gps_df.to_csv('cleaned_gps_data.csv', index=False)
    
    if merged_df is not None:
        merged_df.to_csv('merged_wifi_location.csv', index=False)
    
    if grid_stats is not None:
        grid_stats.to_csv('grid_coverage_statistics.csv', index=False)
    
    print("Data saved successfully.") 