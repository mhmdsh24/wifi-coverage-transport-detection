import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import pickle
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_models():
    """Load the trained models for prediction"""
    print("Loading trained models...")
    
    models = {}
    
    # Load feature list
    with open('models/feature_list.txt', 'r') as f:
        models['feature_list'] = f.read().strip().split('\n')
    
    # Load feature scaler
    with open('models/feature_scaler.pkl', 'rb') as f:
        models['scaler'] = pickle.load(f)
    
    # Load RF model
    with open('models/rf_coverage_model.pkl', 'rb') as f:
        models['rf_model'] = pickle.load(f)
    
    # Load GB model
    with open('models/gb_coverage_model.pkl', 'rb') as f:
        models['gb_model'] = pickle.load(f)
    
    # Load KNN spatial model
    with open('models/knn_spatial_model.pkl', 'rb') as f:
        models['knn_spatial'] = pickle.load(f)
        
    # Load anomaly detection model if available
    try:
        with open('models/anomaly_detector.pkl', 'rb') as f:
            models['anomaly_model'] = pickle.load(f)
            
        with open('models/anomaly_scaler.pkl', 'rb') as f:
            models['anomaly_scaler'] = pickle.load(f)
            
        print("Loaded anomaly detection model.")
    except FileNotFoundError:
        print("Anomaly detection model not found.")
        models['anomaly_model'] = None
        models['anomaly_scaler'] = None
    
    return models

def prepare_data(wifi_path, location_path, gps_path=None, tolerance_ms=5000):
    """
    Prepare data for coverage prediction
    
    Parameters:
    -----------
    wifi_path : str
        Path to WiFi data CSV
    location_path : str
        Path to location data CSV
    gps_path : str, optional
        Path to GPS data CSV
    tolerance_ms : int, default=5000
        Tolerance in milliseconds for merging data
        
    Returns:
    --------
    tuple: (merged_df, grid_stats)
    """
    print("Loading and preparing data...")
    
    # Load data
    wifi_df = pd.read_csv(wifi_path)
    location_df = pd.read_csv(location_path)
    
    # Ensure timestamp_ms is in correct format
    if 'timestamp' in wifi_df.columns and 'timestamp_ms' not in wifi_df.columns:
        wifi_df.rename(columns={'timestamp': 'timestamp_ms'}, inplace=True)
        
    if 'timestamp' in location_df.columns and 'timestamp_ms' not in location_df.columns:
        location_df.rename(columns={'timestamp': 'timestamp_ms'}, inplace=True)
    
    # Ensure timestamp_ms is numeric
    wifi_df['timestamp_ms'] = pd.to_numeric(wifi_df['timestamp_ms'])
    location_df['timestamp_ms'] = pd.to_numeric(location_df['timestamp_ms'])
    
    # Sort by timestamp
    wifi_df = wifi_df.sort_values('timestamp_ms')
    location_df = location_df.sort_values('timestamp_ms')
    
    # Add datetime column if not present
    if 'timestamp_dt' not in wifi_df.columns:
        wifi_df['timestamp_dt'] = pd.to_datetime(wifi_df['timestamp_ms'], unit='ms')
        
    if 'timestamp_dt' not in location_df.columns:
        location_df['timestamp_dt'] = pd.to_datetime(location_df['timestamp_ms'], unit='ms')
    
    # Add hour and day columns if not present
    if 'hour' not in wifi_df.columns:
        wifi_df['hour'] = wifi_df['timestamp_dt'].dt.hour
        
    if 'day_of_week' not in wifi_df.columns:
        wifi_df['day_of_week'] = wifi_df['timestamp_dt'].dt.dayofweek
    
    # Merge datasets
    print("Merging WiFi and location data...")
    merged_df = pd.merge_asof(
        wifi_df,
        location_df[['timestamp_ms', 'latitude_deg', 'longitude_deg', 'altitude_m', 'speed_mps']],
        on='timestamp_ms',
        direction='nearest',
        tolerance=tolerance_ms
    )
    
    # Drop rows with missing location data
    merged_df = merged_df.dropna(subset=['latitude_deg', 'longitude_deg'])
    
    # Create grid cells
    grid_precision = 4
    merged_df['lat_grid'] = np.round(merged_df['latitude_deg'], grid_precision)
    merged_df['lon_grid'] = np.round(merged_df['longitude_deg'], grid_precision)
    
    # Fix column name issues by renaming if incorrect column names exist
    column_mapping = {
        'rss_change': 'rssi_change',
        'rss_rolling_std': 'rssi_rolling_std'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in merged_df.columns and new_col not in merged_df.columns:
            print(f"Renaming column '{old_col}' to '{new_col}'")
            merged_df = merged_df.rename(columns={old_col: new_col})
    
    # Add signal stability metrics if not present
    if 'rssi_change' not in merged_df.columns:
        merged_df = merged_df.sort_values(['bssid', 'timestamp_ms'])
        merged_df['rssi_change'] = merged_df.groupby('bssid')['rssi'].diff()
    
    if 'rssi_rolling_std' not in merged_df.columns:
        window_size = 5
        merged_df['rssi_rolling_std'] = merged_df.groupby('bssid')['rssi'].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).std()
        )
    
    # Define low signal threshold
    rssi_threshold = -75
    
    # Create target variable (1 for low coverage, 0 for good coverage)
    merged_df['low_coverage'] = (merged_df['rssi'] < rssi_threshold).astype(int)
    
    # Compute grid statistics
    print("Computing grid statistics...")
    grid_stats = merged_df.groupby(['lat_grid', 'lon_grid', 'bssid']).agg({
        'rssi': ['mean', 'std', 'min', 'max', 'count'],
        'low_coverage': 'mean',
        'rssi_change': ['mean', 'std'],
        'rssi_rolling_std': 'mean'
    }).reset_index()
    
    # Flatten the hierarchical column names
    grid_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in grid_stats.columns]
    
    # Add hourly variation if possible
    if 'hour' in merged_df.columns:
        # Calculate hourly variation
        hourly_stats = merged_df.groupby(['lat_grid', 'lon_grid', 'bssid', 'hour']).agg({
            'rssi': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
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
    
    return merged_df, grid_stats

def detect_anomalies(merged_df, models):
    """
    Detect signal anomalies in the data
    
    Parameters:
    -----------
    merged_df : pandas DataFrame
        Merged WiFi and location data
    models : dict
        Dictionary containing loaded models
        
    Returns:
    --------
    pandas DataFrame: Merged data with anomaly flags
    """
    # If no anomaly model is available, return the original dataframe
    if models['anomaly_model'] is None or models['anomaly_scaler'] is None:
        print("No anomaly detection model available. Skipping anomaly detection.")
        merged_df['is_anomaly'] = 0
        return merged_df
    
    print("Detecting signal anomalies...")
    
    # Fix column name issues by renaming if incorrect column names exist
    column_mapping = {
        'rss_change': 'rssi_change',
        'rss_rolling_std': 'rssi_rolling_std'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in merged_df.columns and new_col not in merged_df.columns:
            print(f"Renaming column '{old_col}' to '{new_col}'")
            merged_df = merged_df.rename(columns={old_col: new_col})
    
    # If the correct columns don't exist, create them
    if 'rssi_change' not in merged_df.columns:
        print("Creating 'rssi_change' column")
        merged_df = merged_df.sort_values(['bssid', 'timestamp_ms'])
        merged_df['rssi_change'] = merged_df.groupby('bssid')['rssi'].diff()
    
    if 'rssi_rolling_std' not in merged_df.columns:
        print("Creating 'rssi_rolling_std' column")
        window_size = 5
        merged_df['rssi_rolling_std'] = merged_df.groupby('bssid')['rssi'].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).std()
        )
    
    # Select features for anomaly detection
    anomaly_features = [
        'rssi', 'latitude_deg', 'longitude_deg', 
        'rssi_change', 'rssi_rolling_std'
    ]
    
    # Add temporal features if available
    if 'hour' in merged_df.columns:
        anomaly_features.extend(['hour', 'day_of_week'])
    
    # Handle missing features
    for feature in list(anomaly_features):
        if feature not in merged_df.columns:
            print(f"Warning: Feature '{feature}' not found, removing from anomaly detection")
            anomaly_features.remove(feature)
    
    # Prepare data for anomaly detection
    X_anomaly = merged_df[anomaly_features].copy()
    
    # Handle missing values
    X_anomaly = X_anomaly.fillna(method='ffill')
    X_anomaly = X_anomaly.fillna(method='bfill')
    X_anomaly = X_anomaly.fillna(0)  # Any remaining NaNs
    
    # Normalize features
    X_anomaly_scaled = models['anomaly_scaler'].transform(X_anomaly)
    
    # Detect anomalies
    anomaly_scores = models['anomaly_model'].decision_function(X_anomaly_scaled)
    anomaly_preds = models['anomaly_model'].predict(X_anomaly_scaled)
    
    # Add results to dataframe
    merged_df['anomaly_score'] = anomaly_scores
    merged_df['is_anomaly'] = (anomaly_preds == -1).astype(int)
    
    # Print summary
    anomaly_count = merged_df['is_anomaly'].sum()
    print(f"Detected {anomaly_count} anomalies out of {len(merged_df)} measurements ({anomaly_count/len(merged_df)*100:.2f}%)")
    
    return merged_df

def predict_coverage(grid_stats, models):
    """
    Predict low coverage areas using the trained models
    
    Parameters:
    -----------
    grid_stats : pandas DataFrame
        Grid cell statistics
    models : dict
        Dictionary containing loaded models
        
    Returns:
    --------
    pandas DataFrame: Grid statistics with predictions
    """
    print("Predicting coverage problems...")
    
    # Get feature list
    feature_list = models['feature_list']
    
    # Check if we have all required features
    missing_features = [f for f in feature_list if f not in grid_stats.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        
        # Try to fix missing features by renaming if they might be misspelled
        column_mapping = {
            'rss_change_mean': 'rssi_change_mean',
            'rss_change_std': 'rssi_change_std',
            'rss_rolling_std_mean': 'rssi_rolling_std_mean'
        }
        
        # Apply column renaming if needed
        for old_col, new_col in column_mapping.items():
            if old_col in grid_stats.columns and new_col in missing_features:
                print(f"Renaming column '{old_col}' to '{new_col}'")
                grid_stats = grid_stats.rename(columns={old_col: new_col})
                missing_features.remove(new_col)
        
        # Create any still-missing features with default values
        for feature in missing_features:
            print(f"Creating missing feature '{feature}' with default value 0")
            grid_stats[feature] = 0  # Add missing features with default values
    
    # Add anomaly density if we have anomaly data
    if 'is_anomaly_sum' in grid_stats.columns and 'rssi_count' in grid_stats.columns:
        grid_stats['anomaly_density'] = grid_stats['is_anomaly_sum'] / grid_stats['rssi_count']
    elif 'anomaly_density' in feature_list and 'anomaly_density' not in grid_stats.columns:
        grid_stats['anomaly_density'] = 0
    
    # Prepare features
    X = grid_stats[feature_list].copy()
    
    # Handle any missing values
    X = X.fillna(0)
    
    # Scale features
    X_scaled = models['scaler'].transform(X)
    
    # Make predictions with both models
    grid_stats['rf_prediction'] = models['rf_model'].predict(X_scaled)
    grid_stats['rf_prob'] = models['rf_model'].predict_proba(X_scaled)[:, 1]
    
    grid_stats['gb_prediction'] = models['gb_model'].predict(X_scaled)
    grid_stats['gb_prob'] = models['gb_model'].predict_proba(X_scaled)[:, 1]
    
    # Create ensemble prediction (average of probabilities)
    grid_stats['ensemble_prob'] = (grid_stats['rf_prob'] + grid_stats['gb_prob']) / 2
    grid_stats['ensemble_prediction'] = (grid_stats['ensemble_prob'] > 0.5).astype(int)
    
    # Print prediction summary
    low_coverage_count = grid_stats['ensemble_prediction'].sum()
    total_grids = len(grid_stats)
    print(f"Predicted {low_coverage_count} low coverage areas out of {total_grids} grid cells ({low_coverage_count/total_grids*100:.2f}%)")
    
    return grid_stats

def generate_coverage_map(grid_stats, merged_df, models, output_path='plots/predicted_coverage_map.png'):
    """
    Generate a coverage map visualization
    
    Parameters:
    -----------
    grid_stats : pandas DataFrame
        Grid cell statistics with predictions
    merged_df : pandas DataFrame
        Merged WiFi and location data
    models : dict
        Dictionary containing loaded models
    output_path : str, default='plots/predicted_coverage_map.png'
        Path to save the coverage map
        
    Returns:
    --------
    None
    """
    print(f"Generating coverage map at {output_path}...")
    
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use KNN spatial model for smooth interpolation
    # Extract coordinates from grid data
    min_lat, max_lat = grid_stats['lat_grid'].min(), grid_stats['lat_grid'].max()
    min_lon, max_lon = grid_stats['lon_grid'].min(), grid_stats['lon_grid'].max()
    
    lat_step = (max_lat - min_lat) / 100
    lon_step = (max_lon - min_lon) / 100
    
    lat_grid = np.arange(min_lat, max_lat, lat_step)
    lon_grid = np.arange(min_lon, max_lon, lon_step)
    
    lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
    prediction_points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
    
    # Use the pre-trained KNN model for spatial interpolation
    grid_predictions = models['knn_spatial'].predict(prediction_points)
    grid_predictions = grid_predictions.reshape(lon_mesh.shape)
    
    # Create the main coverage map
    plt.figure(figsize=(12, 10))
    
    # Plot interpolated coverage
    coverage_map = plt.contourf(lon_mesh, lat_mesh, grid_predictions, levels=50, cmap='RdYlGn_r', alpha=0.7)
    plt.colorbar(coverage_map, label='Low Coverage Probability')
    
    # Plot predicted grid cells
    low_coverage_points = grid_stats[grid_stats['ensemble_prediction'] == 1]
    good_coverage_points = grid_stats[grid_stats['ensemble_prediction'] == 0]
    
    plt.scatter(
        good_coverage_points['lon_grid'], 
        good_coverage_points['lat_grid'], 
        c='green', 
        marker='^', 
        edgecolors='k', 
        alpha=0.6, 
        label='Good Coverage'
    )
    plt.scatter(
        low_coverage_points['lon_grid'], 
        low_coverage_points['lat_grid'], 
        c='red', 
        marker='o', 
        edgecolors='k',
        alpha=0.6, 
        label='Low Coverage'
    )
    
    # If we have anomaly data, plot anomaly points
    if 'is_anomaly' in merged_df.columns and merged_df['is_anomaly'].sum() > 0:
        # Plot anomaly points
        anomaly_points = merged_df[merged_df['is_anomaly'] == 1]
        
        plt.scatter(
            anomaly_points['longitude_deg'], 
            anomaly_points['latitude_deg'], 
            c='purple', 
            marker='*', 
            s=100,
            edgecolors='k', 
            alpha=0.8, 
            label='Signal Anomaly'
        )
    
    plt.title('Predicted Low Coverage Areas and Signal Anomalies')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Create a separate anomaly map if we have anomaly data
    if 'anomaly_density' in grid_stats.columns:
        anomaly_output_path = output_path.replace('.png', '_anomalies.png')
        
        plt.figure(figsize=(12, 10))
        
        # Create a continuous colormap of anomaly density
        anomaly_map = plt.scatter(
            grid_stats['lon_grid'],
            grid_stats['lat_grid'],
            c=grid_stats['anomaly_density'],
            cmap='plasma',
            alpha=0.7,
            s=50,
            edgecolors='k'
        )
        plt.colorbar(anomaly_map, label='Anomaly Density')
        
        # Overlay low coverage contours
        contour = plt.contour(
            lon_mesh, 
            lat_mesh, 
            grid_predictions, 
            levels=[0.5], 
            colors='red', 
            linewidths=2
        )
        plt.clabel(contour, inline=True, fontsize=10, fmt='Low Coverage')
        
        plt.title('Signal Anomalies and Coverage Problems')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(anomaly_output_path)
        plt.close()
        
        print(f"Generated anomaly map at {anomaly_output_path}")
    
    print(f"Coverage map generated at {output_path}")

def save_prediction_results(grid_stats, output_dir='output'):
    """
    Save prediction results to CSV
    
    Parameters:
    -----------
    grid_stats : pandas DataFrame
        Grid cell statistics with predictions
    output_dir : str, default='output'
        Directory to save results
        
    Returns:
    --------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract main columns for output
    result_columns = [
        'lat_grid', 'lon_grid', 'bssid', 
        'rssi_mean', 'rssi_std', 'rssi_count',
        'rf_prediction', 'rf_prob',
        'gb_prediction', 'gb_prob',
        'ensemble_prediction', 'ensemble_prob'
    ]
    
    # Add anomaly columns if available
    if 'anomaly_density' in grid_stats.columns:
        result_columns.append('anomaly_density')
    if 'is_anomaly_sum' in grid_stats.columns:
        result_columns.append('is_anomaly_sum')
    
    # Save results
    output_path = f"{output_dir}/coverage_predictions.csv"
    grid_stats[result_columns].to_csv(output_path, index=False)
    print(f"Saved prediction results to {output_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Predict low cellular/WiFi coverage areas')
    parser.add_argument('--wifi', type=str, default='cleaned_wifi_data.csv', help='Path to WiFi data CSV')
    parser.add_argument('--location', type=str, default='cleaned_location_data.csv', help='Path to location data CSV')
    parser.add_argument('--output', type=str, default='plots/predicted_coverage_map.png', help='Path to save coverage map')
    args = parser.parse_args()
    
    # Load trained models
    models = load_models()
    
    # Prepare data
    merged_df, grid_stats = prepare_data(args.wifi, args.location)
    
    # Detect anomalies
    merged_df = detect_anomalies(merged_df, models)
    
    # Aggregate anomaly information to grid level if anomaly detection was performed
    if 'is_anomaly' in merged_df.columns and merged_df['is_anomaly'].sum() > 0:
        grid_anomaly_stats = merged_df.groupby(['lat_grid', 'lon_grid', 'bssid']).agg({
            'is_anomaly': ['mean', 'sum', 'count']
        }).reset_index()
        
        # Flatten column names
        grid_anomaly_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in grid_anomaly_stats.columns]
        
        # Merge anomaly information into grid_stats
        grid_stats = pd.merge(
            grid_stats,
            grid_anomaly_stats,
            on=['lat_grid', 'lon_grid', 'bssid'],
            how='left'
        )
        
        # Fill missing values for grids without anomaly information
        grid_stats = grid_stats.fillna({
            'is_anomaly_mean': 0,
            'is_anomaly_sum': 0
        })
        
        # Create an anomaly density feature (anomalies per measurement)
        grid_stats['anomaly_density'] = grid_stats['is_anomaly_sum'] / grid_stats['rssi_count']
    
    # Predict coverage
    grid_stats = predict_coverage(grid_stats, models)
    
    # Generate coverage map
    generate_coverage_map(grid_stats, merged_df, models, args.output)
    
    # Save prediction results
    save_prediction_results(grid_stats)

if __name__ == "__main__":
    main() 