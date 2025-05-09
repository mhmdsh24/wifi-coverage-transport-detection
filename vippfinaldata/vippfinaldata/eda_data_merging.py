import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from eda_utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Create plots directory
os.makedirs('plots', exist_ok=True)

print("Starting Data Integration and Relationship Analysis...")

# Load the cleaned data files if they exist, otherwise load and clean the original files
try:
    location_df = pd.read_csv('cleaned_location_data.csv')
    print("Loaded cleaned location data")
except FileNotFoundError:
    location_df = load_data('Hips_Location.csv')
    print("Loaded original location data")
    
try:
    gps_df = pd.read_csv('cleaned_gps_data.csv')
    print("Loaded cleaned GPS data")
except FileNotFoundError:
    gps_df = load_data('Hips_GPS.csv')
    print("Loaded original GPS data")
    
try:
    # Load a sample of the cleaned WiFi data (it's large)
    wifi_df = pd.read_csv('cleaned_wifi_data.csv', nrows=100000)
    print("Loaded cleaned WiFi data sample")
except FileNotFoundError:
    # Load a sample of the original WiFi data
    wifi_df = load_data('Hips_WiFi.csv', sample_size=100000)
    print("Loaded original WiFi data sample")

# Ensure all datasets have compatible timestamp formats
if location_df is not None:
    if 'timestamp_ms' not in location_df.columns:
        print("Error: location_df doesn't have timestamp_ms column")
    else:
        # Ensure timestamp is numeric
        location_df['timestamp_ms'] = pd.to_numeric(location_df['timestamp_ms'], errors='coerce')

if gps_df is not None:
    if 'timestamp_ms' not in gps_df.columns:
        print("Error: gps_df doesn't have timestamp_ms column")
    else:
        # Ensure timestamp is numeric
        gps_df['timestamp_ms'] = pd.to_numeric(gps_df['timestamp_ms'], errors='coerce')

if wifi_df is not None:
    # Rename timestamp column if needed
    if 'timestamp' in wifi_df.columns and 'timestamp_ms' not in wifi_df.columns:
        wifi_df.rename(columns={'timestamp': 'timestamp_ms'}, inplace=True)
    
    if 'timestamp_ms' not in wifi_df.columns:
        print("Error: wifi_df doesn't have timestamp_ms column")
    else:
        # Ensure timestamp is numeric
        wifi_df['timestamp_ms'] = pd.to_numeric(wifi_df['timestamp_ms'], errors='coerce')

# Merge Location and WiFi data
if location_df is not None and wifi_df is not None:
    print("\nMerging Location and WiFi data...")
    
    # Sort dataframes by timestamp
    location_df = location_df.sort_values('timestamp_ms')
    wifi_df = wifi_df.sort_values('timestamp_ms')
    
    # Use merge_asof to match each WiFi reading with the nearest location reading
    wifi_location_df = pd.merge_asof(
        wifi_df,
        location_df,
        on='timestamp_ms',
        direction='nearest',
        tolerance=pd.Timedelta('5s').total_seconds() * 1000  # 5 second tolerance
    )
    
    print(f"Merged WiFi-Location dataframe shape: {wifi_location_df.shape}")
    
    # Check how many WiFi readings got matched with location data
    match_rate = (wifi_location_df['latitude_deg'].notna().sum() / len(wifi_location_df)) * 100
    print(f"WiFi-Location match rate: {match_rate:.2f}%")
    
    if match_rate > 0:
        # Analyze signal strength distribution spatially
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            wifi_location_df['longitude_deg'],
            wifi_location_df['latitude_deg'],
            c=wifi_location_df['rssi'],
            cmap='RdYlGn_r',  # Red for weak signals, green for strong
            alpha=0.7,
            s=30
        )
        plt.colorbar(scatter, label='RSSI (dBm)')
        plt.title('WiFi Signal Strength Distribution')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/wifi_signal_spatial_distribution.png")
        plt.close()
        
        # Create a heatmap using hexbin
        plt.figure(figsize=(12, 10))
        hb = plt.hexbin(
            wifi_location_df['longitude_deg'],
            wifi_location_df['latitude_deg'],
            C=wifi_location_df['rssi'],
            reduce_C_function=np.mean,
            gridsize=50,
            cmap='RdYlGn_r'
        )
        plt.colorbar(hb, label='Average RSSI (dBm)')
        plt.title('WiFi Signal Strength Heatmap')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/wifi_signal_heatmap.png")
        plt.close()
        
        # Analyze signal strength by altitude
        plt.figure(figsize=(12, 6))
        plt.scatter(wifi_location_df['altitude_m'], wifi_location_df['rssi'], alpha=0.5)
        plt.title('Signal Strength vs. Altitude')
        plt.xlabel('Altitude (m)')
        plt.ylabel('RSSI (dBm)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/wifi_signal_vs_altitude.png")
        plt.close()
        
        # Use binned altitude for better visualization
        wifi_location_df['altitude_bin'] = pd.cut(
            wifi_location_df['altitude_m'],
            bins=10,
            labels=False
        )
        
        plt.figure(figsize=(14, 6))
        sns.boxplot(x='altitude_bin', y='rssi', data=wifi_location_df)
        plt.title('Signal Strength by Altitude Bin')
        plt.xlabel('Altitude Bin (Higher Number = Higher Altitude)')
        plt.ylabel('RSSI (dBm)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/wifi_signal_by_altitude_bin.png")
        plt.close()
        
        # Signal strength variability
        # Group by location grid cells and compute signal stats
        # Create grid cells by rounding coordinates
        decimals = 4  # Adjust based on desired grid size
        wifi_location_df['lat_grid'] = np.round(wifi_location_df['latitude_deg'], decimals)
        wifi_location_df['lon_grid'] = np.round(wifi_location_df['longitude_deg'], decimals)
        
        # Group by grid cells and compute signal statistics
        grid_stats = wifi_location_df.groupby(['lat_grid', 'lon_grid']).agg({
            'rssi': ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()
        
        # Flatten the hierarchical column names
        grid_stats.columns = ['lat_grid', 'lon_grid', 'rssi_mean', 'rssi_std', 'rssi_min', 'rssi_max', 'count']
        
        # Filter to cells with sufficient measurements
        grid_stats = grid_stats[grid_stats['count'] >= 5]
        
        # Create a map of signal variability
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            grid_stats['lon_grid'],
            grid_stats['lat_grid'],
            c=grid_stats['rssi_std'],
            cmap='plasma',
            alpha=0.7,
            s=grid_stats['count'],  # Size by number of measurements
            edgecolor='k',
            linewidth=0.5
        )
        plt.colorbar(scatter, label='RSSI Standard Deviation (dBm)')
        plt.title('WiFi Signal Variability by Location')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/wifi_signal_variability_map.png")
        plt.close()
        
        # Create a map of average signal strength
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            grid_stats['lon_grid'],
            grid_stats['lat_grid'],
            c=grid_stats['rssi_mean'],
            cmap='RdYlGn_r',
            alpha=0.7,
            s=grid_stats['count'],  # Size by number of measurements
            edgecolor='k',
            linewidth=0.5
        )
        plt.colorbar(scatter, label='Average RSSI (dBm)')
        plt.title('WiFi Signal Strength by Location')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/wifi_signal_average_map.png")
        plt.close()
        
        # Detect low coverage areas using clustering
        # Prepare data for clustering (locations with weak signals)
        weak_signal_threshold = -75  # Define weak signal threshold (dBm)
        weak_signals = wifi_location_df[wifi_location_df['rssi'] < weak_signal_threshold]
        
        if len(weak_signals) > 0:
            # Extract coordinates for clustering
            X = weak_signals[['latitude_deg', 'longitude_deg']].values
            
            # Normalize coordinates for clustering
            X_normalized = StandardScaler().fit_transform(X)
            
            # Apply DBSCAN clustering
            db = DBSCAN(eps=0.08, min_samples=5).fit(X_normalized)
            
            # Add cluster labels to dataframe
            weak_signals = weak_signals.copy()
            weak_signals['cluster'] = db.labels_
            
            # Plot clusters of weak signal areas
            plt.figure(figsize=(12, 10))
            # Plot all data points as background
            plt.scatter(
                wifi_location_df['longitude_deg'],
                wifi_location_df['latitude_deg'],
                c='lightgrey',
                alpha=0.2,
                s=10
            )
            # Filter out noise points (cluster=-1)
            cluster_points = weak_signals[weak_signals['cluster'] >= 0]
            # Plot clustered weak signal points
            if len(cluster_points) > 0:
                scatter = plt.scatter(
                    cluster_points['longitude_deg'],
                    cluster_points['latitude_deg'],
                    c=cluster_points['cluster'],
                    cmap='tab10',
                    alpha=0.8,
                    s=50,
                    edgecolor='k',
                    linewidth=0.5
                )
                plt.colorbar(scatter, label='Cluster')
            plt.title('Clusters of Low WiFi Coverage Areas')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("plots/low_coverage_clusters.png")
            plt.close()
            
            # Analyze signal quality by distance to nearest cluster center
            # Find cluster centers
            if len(cluster_points) > 0:
                cluster_centers = {}
                for cluster_id in cluster_points['cluster'].unique():
                    cluster_data = cluster_points[cluster_points['cluster'] == cluster_id]
                    center_lat = cluster_data['latitude_deg'].mean()
                    center_lon = cluster_data['longitude_deg'].mean()
                    cluster_centers[cluster_id] = (center_lat, center_lon)
                
                # Function to calculate distance to nearest cluster center
                def min_distance_to_cluster(lat, lon):
                    if not cluster_centers:
                        return np.nan
                    
                    min_dist = float('inf')
                    for _, (center_lat, center_lon) in cluster_centers.items():
                        dist = haversine_distance(lat, lon, center_lat, center_lon)
                        min_dist = min(min_dist, dist)
                    
                    return min_dist
                
                # Calculate distance to nearest cluster for all WiFi points
                wifi_location_df['dist_to_cluster'] = wifi_location_df.apply(
                    lambda row: min_distance_to_cluster(row['latitude_deg'], row['longitude_deg']),
                    axis=1
                )
                
                # Bin the distances for visualization
                wifi_location_df['dist_bin'] = pd.cut(
                    wifi_location_df['dist_to_cluster'],
                    bins=[0, 100, 200, 300, 500, 1000, 2000, np.inf],
                    labels=['0-100m', '100-200m', '200-300m', '300-500m', '500-1000m', '1000-2000m', '>2000m']
                )
                
                # Plot signal strength by distance to nearest low coverage cluster
                plt.figure(figsize=(14, 6))
                sns.boxplot(x='dist_bin', y='rssi', data=wifi_location_df.dropna(subset=['dist_bin']))
                plt.title('Signal Strength by Distance to Nearest Low Coverage Area')
                plt.xlabel('Distance to Nearest Low Coverage Cluster')
                plt.ylabel('RSSI (dBm)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig("plots/signal_by_distance_to_low_coverage.png")
                plt.close()
                
                # Save merged dataframe with cluster information
                wifi_location_df.to_csv('merged_wifi_location.csv', index=False)
                print("Merged WiFi-Location data saved to merged_wifi_location.csv")
    else:
        print("Insufficient matches between WiFi and Location data for spatial analysis")

# Merge GPS and Location data
if gps_df is not None and location_df is not None:
    print("\nComparing GPS and Location data...")
    
    # Sort dataframes by timestamp
    gps_df = gps_df.sort_values('timestamp_ms')
    location_df = location_df.sort_values('timestamp_ms')
    
    # Use merge_asof to match each GPS reading with the nearest location reading
    gps_location_df = pd.merge_asof(
        gps_df,
        location_df,
        on='timestamp_ms',
        direction='nearest',
        tolerance=pd.Timedelta('1s').total_seconds() * 1000,  # 1 second tolerance
        suffixes=('_gps', '_loc')
    )
    
    print(f"Merged GPS-Location dataframe shape: {gps_location_df.shape}")
    
    # Check how many GPS readings got matched with location data
    match_rate = (gps_location_df['latitude_deg_loc'].notna().sum() / len(gps_location_df)) * 100
    print(f"GPS-Location match rate: {match_rate:.2f}%")
    
    if match_rate > 0:
        # Compare GPS and Location coordinates
        plt.figure(figsize=(10, 10))
        plt.scatter(
            gps_location_df['longitude_deg_gps'],
            gps_location_df['latitude_deg_gps'],
            alpha=0.5,
            label='GPS'
        )
        plt.scatter(
            gps_location_df['longitude_deg_loc'],
            gps_location_df['latitude_deg_loc'],
            alpha=0.5,
            label='Location'
        )
        plt.title('GPS vs. Location Coordinates')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/gps_vs_location_coordinates.png")
        plt.close()
        
        # Calculate distance between GPS and Location coordinates
        gps_location_df['coord_distance_m'] = gps_location_df.apply(
            lambda row: haversine_distance(
                row['latitude_deg_gps'], row['longitude_deg_gps'],
                row['latitude_deg_loc'], row['longitude_deg_loc']
            ) if pd.notna(row['latitude_deg_loc']) and pd.notna(row['longitude_deg_loc']) else np.nan,
            axis=1
        )
        
        # Plot histogram of distance between GPS and Location coordinates
        plt.figure(figsize=(12, 6))
        sns.histplot(
            gps_location_df['coord_distance_m'][gps_location_df['coord_distance_m'] < 1000],
            kde=True,
            bins=30
        )
        plt.title('Distance Between GPS and Location Coordinates')
        plt.xlabel('Distance (m)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/gps_location_distance.png")
        plt.close()
        
        # Compare accuracy
        plt.figure(figsize=(12, 6))
        plt.scatter(
            gps_location_df['accuracy_m_gps'],
            gps_location_df['accuracy_m_loc'],
            alpha=0.5
        )
        plt.plot([0, 100], [0, 100], 'r--')  # Line of perfect agreement
        plt.title('GPS vs. Location Accuracy')
        plt.xlabel('GPS Accuracy (m)')
        plt.ylabel('Location Accuracy (m)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/gps_vs_location_accuracy.png")
        plt.close()
        
        # Save merged GPS-Location dataframe
        gps_location_df.to_csv('merged_gps_location.csv', index=False)
        print("Merged GPS-Location data saved to merged_gps_location.csv")
    else:
        print("Insufficient matches between GPS and Location data for comparison")

print("Data Integration and Relationship Analysis completed!") 