import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from eda_utils import *
from sklearn.cluster import DBSCAN

# Create plots directory
os.makedirs('plots', exist_ok=True)

print("Starting Location Data Analysis...")

# Load location data
location_df = load_data('Hips_Location.csv')

# Basic analysis
analyze_dataframe(location_df, "Location")

# Handle outliers in location data
location_cleaned = handle_outliers(location_df, 
                                   columns=['latitude_deg', 'longitude_deg', 'altitude_m', 'accuracy_m'],
                                   method='iqr',
                                   threshold=2.0)  # Using 2.0 for threshold to be less aggressive with outlier removal

# Handle any missing values
location_cleaned = handle_missing_values(location_cleaned)

if location_cleaned is not None:
    # Plot geographical distribution
    plt.figure(figsize=(12, 10))
    plt.scatter(location_cleaned['longitude_deg'], location_cleaned['latitude_deg'], 
                alpha=0.5, c=location_cleaned['altitude_m'], cmap='viridis')
    plt.colorbar(label='Altitude (m)')
    plt.title('Geographical Distribution of Data Points')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/location_geographical_distribution.png")
    plt.close()
    
    # Plot accuracy distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(location_cleaned['accuracy_m'], kde=True, bins=30)
    plt.title('Distribution of GPS Accuracy')
    plt.xlabel('Accuracy (m)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/location_accuracy_distribution.png")
    plt.close()
    
    # Plot altitude distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(location_cleaned['altitude_m'], kde=True, bins=30)
    plt.title('Distribution of Altitude')
    plt.xlabel('Altitude (m)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/location_altitude_distribution.png")
    plt.close()
    
    # Add a distance field to analyze movement patterns
    # First, convert to proper coordinates for distance calculation
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points on the earth"""
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        r = 6371  # Radius of earth in kilometers
        
        return c * r * 1000  # Convert to meters
    
    # Calculate distance between consecutive points
    location_cleaned['prev_lat'] = location_cleaned['latitude_deg'].shift(1)
    location_cleaned['prev_lon'] = location_cleaned['longitude_deg'].shift(1)
    
    # Apply haversine distance function
    location_cleaned['distance_m'] = location_cleaned.apply(
        lambda row: haversine_distance(
            row['prev_lat'], row['prev_lon'], row['latitude_deg'], row['longitude_deg']
        ) if pd.notna(row['prev_lat']) and pd.notna(row['prev_lon']) else 0,
        axis=1
    )
    
    # Plot distance histogram
    plt.figure(figsize=(12, 6))
    sns.histplot(location_cleaned['distance_m'][location_cleaned['distance_m'] < 100], kde=True, bins=30)
    plt.title('Distribution of Distance Between Consecutive Points (< 100m)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/location_distance_distribution.png")
    plt.close()
    
    # Calculate speed using timestamp difference
    location_cleaned['prev_timestamp'] = location_cleaned['timestamp_ms'].shift(1)
    location_cleaned['time_diff_sec'] = (location_cleaned['timestamp_ms'] - location_cleaned['prev_timestamp']) / 1000.0
    
    # Avoid division by zero
    location_cleaned['speed_mps'] = np.where(
        location_cleaned['time_diff_sec'] > 0,
        location_cleaned['distance_m'] / location_cleaned['time_diff_sec'],
        0
    )
    
    # Plot speed histogram (filtering out unrealistic values)
    plt.figure(figsize=(12, 6))
    sns.histplot(location_cleaned['speed_mps'][
        (location_cleaned['speed_mps'] > 0) & (location_cleaned['speed_mps'] < 30)
    ], kde=True, bins=30)
    plt.title('Distribution of Speed (0-30 m/s)')
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/location_speed_distribution.png")
    plt.close()
    
    # Plot accuracy vs. time
    location_cleaned['timestamp_dt'] = pd.to_datetime(location_cleaned['timestamp_ms'], unit='ms')
    
    plt.figure(figsize=(14, 6))
    plt.plot(location_cleaned['timestamp_dt'], location_cleaned['accuracy_m'], '-', alpha=0.7)
    plt.title('GPS Accuracy Over Time')
    plt.xlabel('Time')
    plt.ylabel('Accuracy (m)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/location_accuracy_over_time.png")
    plt.close()
    
    # Analyze data density by clustering
    # Use DBSCAN for clustering locations
    coords = location_cleaned[['latitude_deg', 'longitude_deg']].values
    
    # Normalize coordinates for clustering
    coords_normalized = StandardScaler().fit_transform(coords)
    
    # Apply DBSCAN clustering
    db = DBSCAN(eps=0.05, min_samples=10).fit(coords_normalized)
    
    # Add cluster labels to dataframe
    location_cleaned['cluster'] = db.labels_
    
    # Plot clusters
    plt.figure(figsize=(12, 10))
    # Filter out noise points (cluster=-1)
    cluster_points = location_cleaned[location_cleaned['cluster'] >= 0]
    plt.scatter(
        cluster_points['longitude_deg'], 
        cluster_points['latitude_deg'], 
        c=cluster_points['cluster'], 
        cmap='viridis',
        alpha=0.7,
        s=50
    )
    plt.title('DBSCAN Clustering of Location Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Cluster')
    plt.tight_layout()
    plt.savefig("plots/location_dbscan_clusters.png")
    plt.close()
    
    # Analyze clusters
    cluster_stats = location_cleaned.groupby('cluster').agg({
        'latitude_deg': ['mean', 'min', 'max', 'count'],
        'longitude_deg': ['mean', 'min', 'max'],
        'altitude_m': ['mean', 'min', 'max'],
        'accuracy_m': ['mean', 'min', 'max']
    })
    
    print("\nCluster Statistics:")
    print(cluster_stats)
    
    # Save cleaned dataframe
    location_cleaned.to_csv('cleaned_location_data.csv', index=False)
    print("\nCleaned location data saved to cleaned_location_data.csv")
    
print("Location Data Analysis completed!") 