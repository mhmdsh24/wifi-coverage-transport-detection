import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import os
import pickle
import warnings

# Create directories if they don't exist
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

def detect_signal_anomalies(wifi_df, location_df, threshold=-75, contamination=0.05):
    """
    Detect anomalies in WiFi signal data using Isolation Forest.
    
    Parameters:
    -----------
    wifi_df : pandas DataFrame
        DataFrame containing WiFi signal data with RSSI values
    location_df : pandas DataFrame
        DataFrame containing location data with GPS coordinates
    threshold : float, default=-75
        RSSI threshold for defining low coverage
    contamination : float, default=0.05
        Expected proportion of anomalies in the dataset
        
    Returns:
    --------
    tuple: (merged_df with anomaly flags, anomaly_df with only anomalies)
    """
    print("Starting signal anomaly detection...")
    
    # Ensure timestamp_ms is in correct format
    if 'timestamp' in wifi_df.columns and 'timestamp_ms' not in wifi_df.columns:
        wifi_df.rename(columns={'timestamp': 'timestamp_ms'}, inplace=True)
    
    # Ensure timestamp_ms is numeric
    wifi_df['timestamp_ms'] = pd.to_numeric(wifi_df['timestamp_ms'])
    location_df['timestamp_ms'] = pd.to_numeric(location_df['timestamp_ms'])
    
    # Sort dataframes by timestamp
    wifi_df = wifi_df.sort_values('timestamp_ms')
    location_df = location_df.sort_values('timestamp_ms')
    
    # Merge datasets using merge_asof with tolerance in milliseconds
    print("Merging WiFi and Location data for anomaly detection...")
    merged_df = pd.merge_asof(
        wifi_df,
        location_df,
        on='timestamp_ms',
        direction='nearest',
        tolerance=5000  # 5 second tolerance in milliseconds
    )
    
    # Add temporal features
    if 'timestamp_dt' not in merged_df.columns:
        merged_df['timestamp_dt'] = pd.to_datetime(merged_df['timestamp_ms'], unit='ms')
    
    merged_df['hour'] = merged_df['timestamp_dt'].dt.hour
    merged_df['minute'] = merged_df['timestamp_dt'].dt.minute
    merged_df['day_of_week'] = merged_df['timestamp_dt'].dt.dayofweek
    
    # Create grid cells for spatial aggregation
    grid_precision = 4
    merged_df['lat_grid'] = np.round(merged_df['latitude_deg'], grid_precision)
    merged_df['lon_grid'] = np.round(merged_df['longitude_deg'], grid_precision)
    
    # Create features for anomaly detection
    print("Creating features for anomaly detection...")
    features = [
        'rssi',                       # Signal strength
        'latitude_deg', 'longitude_deg',  # Location
        'hour', 'day_of_week'         # Temporal features
    ]
    
    # Feature engineering: Create additional features
    # Calculate signal rate of change (derivative) between consecutive readings
    merged_df = merged_df.sort_values(['bssid', 'timestamp_ms'])
    merged_df['rssi_change'] = merged_df.groupby('bssid')['rssi'].diff()
    
    # Calculate variance in a rolling window
    window_size = 5  # Adjust based on your needs
    merged_df['rssi_rolling_std'] = merged_df.groupby('bssid')['rssi'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).std()
    )
    
    # Add these to features
    features.extend(['rssi_change', 'rssi_rolling_std'])
    
    # Flag low coverage for reference
    merged_df['low_coverage'] = (merged_df['rssi'] < threshold).astype(int)
    
    # Prepare data for anomaly detection
    X = merged_df[features].copy()
    
    # Handle missing values
    X = X.fillna(method='ffill')
    X = X.fillna(method='bfill')
    X = X.fillna(0)  # Any remaining NaNs
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    print("Training Isolation Forest model...")
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)
    
    # Get anomaly predictions (-1 for anomalies, 1 for normal)
    anomaly_scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)
    
    # Add results to dataframe
    merged_df['anomaly_score'] = anomaly_scores
    merged_df['is_anomaly'] = (predictions == -1).astype(int)
    
    # Create visualizations
    visualize_anomalies(merged_df, features)
    
    # Extract only the anomalies
    anomaly_df = merged_df[merged_df['is_anomaly'] == 1].copy()
    
    print(f"Anomaly detection complete. Found {len(anomaly_df)} anomalies out of {len(merged_df)} points ({len(anomaly_df)/len(merged_df)*100:.2f}%).")
    
    return merged_df, anomaly_df

def visualize_anomalies(df, features):
    """Create visualizations for the anomalies detected"""
    # Create plots directory if it doesn't exist
    os.makedirs('plots/anomalies', exist_ok=True)
    
    # Plot 1: Anomaly scores distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['anomaly_score'], bins=50, kde=True)
    plt.title('Distribution of Anomaly Scores')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.savefig('plots/anomalies/anomaly_score_distribution.png')
    plt.close()
    
    # Plot 2: RSSI vs Anomaly Score scatter plot
    plt.figure(figsize=(10, 6))
    normal = df[df['is_anomaly'] == 0]
    anomalies = df[df['is_anomaly'] == 1]
    
    plt.scatter(normal['rssi'], normal['anomaly_score'], alpha=0.5, label='Normal', color='blue')
    plt.scatter(anomalies['rssi'], anomalies['anomaly_score'], alpha=0.7, label='Anomaly', color='red')
    plt.title('RSSI vs Anomaly Score')
    plt.xlabel('RSSI (dBm)')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.savefig('plots/anomalies/rssi_vs_anomaly_score.png')
    plt.close()
    
    # Plot 3: Spatial distribution of anomalies
    plt.figure(figsize=(12, 10))
    plt.scatter(
        normal['longitude_deg'],
        normal['latitude_deg'],
        alpha=0.3,
        label='Normal',
        color='blue',
        s=10
    )
    plt.scatter(
        anomalies['longitude_deg'],
        anomalies['latitude_deg'],
        alpha=0.7,
        label='Anomaly',
        color='red',
        s=30
    )
    plt.title('Spatial Distribution of Signal Anomalies')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/anomalies/anomaly_spatial_distribution.png')
    plt.close()
    
    # Plot 4: Temporal patterns of anomalies
    plt.figure(figsize=(14, 7))
    
    # Group by hour and calculate anomaly percentage
    hourly_anomalies = df.groupby('hour')['is_anomaly'].mean() * 100
    
    sns.barplot(x=hourly_anomalies.index, y=hourly_anomalies.values)
    plt.title('Percentage of Anomalies by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Anomaly Percentage (%)')
    plt.xticks(range(0, 24))
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig('plots/anomalies/hourly_anomaly_distribution.png')
    plt.close()
    
    # Plot 5: Feature correlations with anomaly scores
    correlation_df = df[features + ['anomaly_score']].copy()
    
    plt.figure(figsize=(12, 10))
    corr = correlation_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Feature Correlation with Anomaly Scores')
    plt.tight_layout()
    plt.savefig('plots/anomalies/feature_correlation.png')
    plt.close()

def detect_anomalies(merged_df, rssi_threshold=-75, contamination=0.03, random_state=42, 
                    output_file='signal_anomalies.csv', plot_dir='plots',
                    transport_mode_df=None):
    """
    Detect anomalous WiFi signal patterns
    
    Parameters:
    -----------
    merged_df : DataFrame
        Merged WiFi and location data
    rssi_threshold : float, default=-75
        RSSI threshold for defining low coverage
    contamination : float, default=0.03
        Expected fraction of anomalies in the data
    random_state : int, default=42
        Random seed for reproducibility
    output_file : str, default='signal_anomalies.csv'
        Output file to save anomalies
    plot_dir : str, default='plots'
        Directory to save plots
    transport_mode_df : DataFrame, optional
        Transport mode predictions with timestamps
        
    Returns:
    --------
    anomalies_df : DataFrame
        DataFrame with detected anomalies
    """
    print("Detecting WiFi signal anomalies...")
    
    # Create output directory
    os.makedirs(plot_dir, exist_ok=True)
    
    # Check for required columns
    required_cols = ['bssid', 'rssi', 'timestamp_ms', 'latitude', 'longitude']
    if not all(col in merged_df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {[col for col in required_cols if col not in merged_df.columns]}")
    
    # Initialize anomaly features
    features = ['rssi', 'latitude', 'longitude']
    
    # Add transport mode features if available
    has_transport_modes = False
    if transport_mode_df is not None:
        print("Incorporating transport mode data into anomaly detection...")
        
        # First check if we need to merge transport modes
        has_transport_modes = False
        
        if 'predicted_mode' in merged_df.columns:
            # Transport mode already merged
            has_transport_modes = True
            print("Using transport modes already in merged data.")
        else:
            # Need to merge transport modes with merged_df
            if 'start_ts' in transport_mode_df.columns and 'end_ts' in transport_mode_df.columns:
                # Create time windows in merged_df if not already there
                if 'time_window' not in merged_df.columns:
                    # Use start_ts as window identifier in transport_mode_df
                    transport_mode_df['time_window'] = transport_mode_df['start_ts']
                    
                    # For each row in merged_df, find which time window it belongs to
                    def assign_time_window(ts):
                        matching_windows = transport_mode_df[
                            (transport_mode_df['start_ts'] <= ts) & 
                            (transport_mode_df['end_ts'] > ts)
                        ]
                        if len(matching_windows) > 0:
                            return matching_windows.iloc[0]['time_window']
                        return None
                    
                    print("Mapping each data point to its transport mode time window...")
                    # This can be slow, so we'll use a more efficient approach
                    # First sort transport_mode_df by start_ts
                    transport_mode_df = transport_mode_df.sort_values('start_ts')
                    
                    # Then iterate through merged_df chronologically
                    merged_df = merged_df.sort_values('timestamp_ms')
                    merged_df['time_window'] = np.nan
                    
                    current_window_idx = 0
                    for i, row in merged_df.iterrows():
                        ts = row['timestamp_ms']
                        
                        # Move forward in transport_mode_df until finding a window that could contain ts
                        while (current_window_idx < len(transport_mode_df) and 
                              transport_mode_df.iloc[current_window_idx]['end_ts'] <= ts):
                            current_window_idx += 1
                        
                        # Check if we found a valid window
                        if current_window_idx < len(transport_mode_df):
                            current_window = transport_mode_df.iloc[current_window_idx]
                            if current_window['start_ts'] <= ts and current_window['end_ts'] > ts:
                                merged_df.at[i, 'time_window'] = current_window['time_window']
                
                # Now merge transport modes with merged_df
                transport_cols = ['time_window', 'predicted_mode']
                # Add probability columns if available
                prob_cols = [col for col in transport_mode_df.columns if col.startswith('prob_')]
                if prob_cols:
                    transport_cols.extend(prob_cols)
                
                # Also include threshold
                if 'rssi_threshold' in transport_mode_df.columns:
                    transport_cols.append('rssi_threshold')
                
                # Do the merge
                merged_df = pd.merge(
                    merged_df,
                    transport_mode_df[transport_cols],
                    on='time_window',
                    how='left'
                )
                
                # Check if merge was successful
                has_transport_modes = 'predicted_mode' in merged_df.columns
                
                if has_transport_modes:
                    print(f"Successfully merged transport modes. Modes found: {merged_df['predicted_mode'].unique()}")
                else:
                    print("Warning: Could not merge transport modes. Using default anomaly detection.")
        
        # If we have transport modes, include them as features
        if has_transport_modes:
            # Create dummy variables for transport modes
            transport_dummies = pd.get_dummies(merged_df['predicted_mode'], prefix='mode')
            merged_df = pd.concat([merged_df, transport_dummies], axis=1)
            
            # Add transport mode dummies to features
            mode_cols = transport_dummies.columns.tolist()
            features.extend(mode_cols)
            
            # If we have probability columns, add them too
            prob_cols = [col for col in merged_df.columns if col.startswith('prob_')]
            if prob_cols:
                features.extend(prob_cols)
                
            # Add dynamic threshold feature
            if 'rssi_threshold' in merged_df.columns:
                # Create a new feature: rssi_relative_to_threshold
                merged_df['rssi_relative_to_threshold'] = merged_df['rssi'] - merged_df['rssi_threshold']
                features.append('rssi_relative_to_threshold')
                
                print("Added mobility-aware threshold feature: rssi_relative_to_threshold")
    
    # Feature engineering - add basic features
    if 'rssi_change' not in merged_df.columns:
        merged_df = merged_df.sort_values(['bssid', 'timestamp_ms'])
        merged_df['rssi_change'] = merged_df.groupby('bssid')['rssi'].diff()
    
    if 'rssi_rolling_std' not in merged_df.columns:
        window_size = 5
        merged_df['rssi_rolling_std'] = merged_df.groupby('bssid')['rssi'].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).std()
        )
    
    # Add these to features
    features.extend(['rssi_change', 'rssi_rolling_std'])
    
    # Drop missing values
    merged_df_clean = merged_df.dropna(subset=features)
    
    # Check if we have enough data
    if len(merged_df_clean) == 0:
        warnings.warn("No data available after dropping missing values")
        return pd.DataFrame()
    
    # Prepare data for anomaly detection
    X = merged_df_clean[features].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train isolation forest model
    if has_transport_modes:
        print("Training Isolation Forest with transport mode features...")
    else:
        print("Training standard Isolation Forest...")
        
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        max_samples='auto'
    )
    
    # Fit model
    model.fit(X_scaled)
    
    # Predict anomalies
    # -1 for anomalies, 1 for normal
    y_pred = model.predict(X_scaled)
    
    # Convert to 0 for normal, 1 for anomaly
    merged_df_clean['anomaly'] = (y_pred == -1).astype(int)
    
    # Get anomaly scores (decision_function returns negative values for anomalies)
    anomaly_scores = -model.decision_function(X_scaled)
    merged_df_clean['anomaly_score'] = anomaly_scores
    
    # Add contextual information about anomalies
    if has_transport_modes:
        # Use different RSSI thresholds based on transport mode
        def get_threshold(row):
            if pd.isna(row['predicted_mode']):
                return rssi_threshold
            
            # Use specific threshold if available
            if 'rssi_threshold' in row and not pd.isna(row['rssi_threshold']):
                return row['rssi_threshold']
            
            # Otherwise use mapping based on mode
            mode_thresholds = {
                'still': -75,
                'walk': -75,
                'run': -78,
                'bike': -80,
                'car': -83,
                'bus': -83,
                'train': -87,
                'subway': -87,
                # For backward compatibility
                'vehicle': -83
            }
            
            return mode_thresholds.get(row['predicted_mode'], rssi_threshold)
        
        merged_df_clean['mode_rssi_threshold'] = merged_df_clean.apply(get_threshold, axis=1)
        
        # Mark RSSI below dynamic threshold
        merged_df_clean['below_threshold'] = (merged_df_clean['rssi'] < merged_df_clean['mode_rssi_threshold']).astype(int)
        
        # Label anomaly types
        merged_df_clean['anomaly_type'] = 'Normal'
        
        # RSSI anomalies
        merged_df_clean.loc[
            (merged_df_clean['anomaly'] == 1) & 
            (merged_df_clean['below_threshold'] == 1), 
            'anomaly_type'
        ] = 'RSSI Anomaly'
        
        # Spatial anomalies (high anomaly score but normal RSSI)
        merged_df_clean.loc[
            (merged_df_clean['anomaly'] == 1) & 
            (merged_df_clean['below_threshold'] == 0), 
            'anomaly_type'
        ] = 'Spatial Anomaly'
        
        # Mobility context anomalies
        merged_df_clean.loc[
            (merged_df_clean['anomaly'] == 1) & 
            (merged_df_clean['rssi'] < rssi_threshold) &  # Below static threshold
            (merged_df_clean['rssi'] >= merged_df_clean['mode_rssi_threshold']),  # But above dynamic threshold
            'anomaly_type'
        ] = 'Mobility Context (Not Anomalous)'
        
        print("Applied mobility-aware thresholds for anomaly detection.")
    else:
        # Just use the static threshold
        merged_df_clean['below_threshold'] = (merged_df_clean['rssi'] < rssi_threshold).astype(int)
        
        # Label anomaly types
        merged_df_clean['anomaly_type'] = 'Normal'
        merged_df_clean.loc[
            (merged_df_clean['anomaly'] == 1) & 
            (merged_df_clean['below_threshold'] == 1), 
            'anomaly_type'
        ] = 'RSSI Anomaly'
        
        merged_df_clean.loc[
            (merged_df_clean['anomaly'] == 1) & 
            (merged_df_clean['below_threshold'] == 0), 
            'anomaly_type'
        ] = 'Spatial Anomaly'
    
    # Extract anomalies
    anomalies_df = merged_df_clean[merged_df_clean['anomaly'] == 1].copy()
    
    # Add percentile rank of anomaly score
    anomalies_df['score_percentile'] = anomalies_df['anomaly_score'].rank(pct=True) * 100
    
    # Save anomalies to CSV
    if output_file:
        anomalies_df.to_csv(output_file, index=False)
        print(f"Saved {len(anomalies_df)} anomalies to {output_file}")
    
    # Plot anomalies
    plot_anomalies(merged_df_clean, anomalies_df, plot_dir, has_transport_modes)
    
    # Save the model
    with open(os.path.join(os.path.dirname(output_file) if '/' in output_file else '', 'anomaly_detector.pkl'), 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'features': features
        }, f)
    
    return anomalies_df

def plot_anomalies(merged_df, anomalies_df, plot_dir, has_transport_modes=False):
    """
    Create visualizations of the detected anomalies
    
    Parameters:
    -----------
    merged_df : DataFrame
        Full merged dataset with anomaly flags
    anomalies_df : DataFrame
        DataFrame containing only anomalies
    plot_dir : str
        Directory to save plots
    has_transport_modes : bool
        Whether transport mode data is available
    """
    print("Creating anomaly visualization plots...")
    
    # Create directory
    os.makedirs(plot_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    
    # Plot RSSI distribution with anomalies highlighted
    plt.figure(figsize=(10, 6))
    sns.histplot(data=merged_df, x='rssi', hue='anomaly', 
               palette={0: 'blue', 1: 'red'}, 
               element='step', bins=30, alpha=0.6)
    plt.title('RSSI Distribution with Anomalies')
    plt.xlabel('RSSI (dBm)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plot_dir, 'rssi_anomalies_distribution.png'))
    plt.close()
    
    # Plot anomaly types
    if 'anomaly_type' in anomalies_df.columns:
        plt.figure(figsize=(10, 6))
        anomaly_counts = anomalies_df['anomaly_type'].value_counts()
        anomaly_counts.plot(kind='bar', color=sns.color_palette("Set2"))
        plt.title('Types of Detected Anomalies')
        plt.xlabel('Anomaly Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'anomaly_types.png'))
        plt.close()
    
    # If transport modes are available, create mode-specific plots
    if has_transport_modes and 'predicted_mode' in merged_df.columns:
        # Plot anomalies by transport mode
        plt.figure(figsize=(12, 6))
        anomaly_by_mode = pd.crosstab(
            merged_df['predicted_mode'], 
            merged_df['anomaly']
        )
        
        # Calculate percentage of anomalies by mode
        anomaly_pct = anomaly_by_mode[1] / anomaly_by_mode.sum(axis=1) * 100
        anomaly_pct = anomaly_pct.sort_values(ascending=False)
        
        anomaly_pct.plot(kind='bar', color='firebrick')
        plt.title('Percentage of Anomalies by Transport Mode')
        plt.xlabel('Transport Mode')
        plt.ylabel('Anomaly Percentage')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'anomalies_by_transport_mode.png'))
        plt.close()
        
        # Plot RSSI by transport mode with anomalies
        plt.figure(figsize=(12, 8))
        
        # Use boxplot to show RSSI distribution by mode
        ax = sns.boxplot(x='predicted_mode', y='rssi', data=merged_df, 
                      palette='viridis', showfliers=False)
        
        # Overlay scatterplot of anomalies
        sns.stripplot(x='predicted_mode', y='rssi', data=anomalies_df,
                    jitter=True, alpha=0.7, color='red', size=4)
        
        # Add mode-specific thresholds as horizontal lines
        unique_modes = merged_df['predicted_mode'].unique()
        
        if 'mode_rssi_threshold' in merged_df.columns:
            for mode in unique_modes:
                threshold = merged_df[merged_df['predicted_mode'] == mode]['mode_rssi_threshold'].iloc[0]
                plt.axhline(y=threshold, color='blue', linestyle='--', alpha=0.5)
        
        plt.title('RSSI by Transport Mode with Anomalies')
        plt.xlabel('Transport Mode')
        plt.ylabel('RSSI (dBm)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'rssi_anomalies_by_mode.png'))
        plt.close()
        
        # Plot anomaly percentage with and without mode-aware thresholds
        if 'mode_rssi_threshold' in merged_df.columns:
            plt.figure(figsize=(12, 6))
            
            # Static threshold anomalies
            static_anomalies = (merged_df['rssi'] < -75)
            static_by_mode = merged_df.groupby('predicted_mode')[static_anomalies].mean() * 100
            
            # Dynamic threshold anomalies
            dynamic_anomalies = (merged_df['rssi'] < merged_df['mode_rssi_threshold'])
            dynamic_by_mode = merged_df.groupby('predicted_mode')[dynamic_anomalies].mean() * 100
            
            # Combine into DataFrame
            comparison_df = pd.DataFrame({
                'Static Threshold (-75 dBm)': static_by_mode,
                'Dynamic Threshold': dynamic_by_mode
            })
            
            # Sort by difference
            comparison_df['Difference'] = comparison_df['Static Threshold (-75 dBm)'] - comparison_df['Dynamic Threshold']
            comparison_df = comparison_df.sort_values('Difference', ascending=False)
            
            # Plot comparison
            comparison_df[['Static Threshold (-75 dBm)', 'Dynamic Threshold']].plot(
                kind='bar', figsize=(12, 6)
            )
            plt.title('Low Coverage Detection: Static vs. Dynamic Thresholds by Transport Mode')
            plt.xlabel('Transport Mode')
            plt.ylabel('Percentage of Samples Below Threshold (%)')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Threshold Type')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'static_vs_dynamic_thresholds.png'))
            plt.close()
    
    # Scatter plot of anomaly score vs RSSI
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_df['rssi'], merged_df['anomaly_score'], 
               c=merged_df['anomaly'], cmap='coolwarm', alpha=0.6)
    plt.colorbar(label='Anomaly (1) / Normal (0)')
    plt.title('Anomaly Score vs RSSI')
    plt.xlabel('RSSI (dBm)')
    plt.ylabel('Anomaly Score')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'anomaly_score_vs_rssi.png'))
    plt.close()

# Add a new function to detect anomalies with mobility-aware thresholds
def detect_anomalies_with_mobility(merged_df, transport_mode_df, rssi_threshold=-75,
                                 contamination=0.03, random_state=42,
                                 output_file='signal_anomalies.csv', plot_dir='plots'):
    """
    Detect WiFi signal anomalies with mobility-aware thresholds
    
    Parameters:
    -----------
    merged_df : DataFrame
        Merged WiFi and location data
    transport_mode_df : DataFrame
        Transport mode predictions with timestamps
    rssi_threshold : float, default=-75
        Static RSSI threshold (will be adjusted based on transport mode)
    contamination : float, default=0.03
        Expected fraction of anomalies in the data
    random_state : int, default=42
        Random seed for reproducibility
    output_file : str, default='signal_anomalies.csv'
        Output file to save anomalies
    plot_dir : str, default='plots'
        Directory to save plots
        
    Returns:
    --------
    anomalies_df : DataFrame
        DataFrame with detected anomalies
    """
    # This is just a wrapper around the main function for clarity
    return detect_anomalies(merged_df, rssi_threshold, contamination, random_state,
                         output_file, plot_dir, transport_mode_df)

if __name__ == "__main__":
    # Test the module with sample data
    try:
        print("Loading WiFi data...")
        wifi_df = pd.read_csv('cleaned_wifi_data.csv')
        
        print("Loading location data...")
        location_df = pd.read_csv('cleaned_location_data.csv')
        
        merged_df, anomaly_df = detect_signal_anomalies(wifi_df, location_df)
        
        # Save anomalies to CSV
        anomaly_df.to_csv('signal_anomalies.csv', index=False)
        print("Anomalies saved to signal_anomalies.csv")
        
    except Exception as e:
        print(f"Error during anomaly detection: {e}") 