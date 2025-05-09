import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from eda_utils import *

# Create plots directory
os.makedirs('plots', exist_ok=True)

print("Starting GPS Data Analysis...")

# Load GPS data
gps_df = load_data('Hips_GPS.csv')

# Basic analysis
analyze_dataframe(gps_df, "GPS")

# Handle outliers in GPS data
gps_cleaned = handle_outliers(gps_df, 
                              columns=['latitude_deg', 'longitude_deg', 'speed_mps', 'accuracy_m'],
                              method='iqr',
                              threshold=2.0)

# Handle any missing values
gps_cleaned = handle_missing_values(gps_cleaned)

if gps_cleaned is not None:
    # Plot geographical distribution
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        gps_cleaned['longitude_deg'], 
        gps_cleaned['latitude_deg'], 
        c=gps_cleaned['speed_mps'], 
        cmap='plasma',
        alpha=0.7,
        s=50
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Speed (m/s)')
    plt.title('GPS Data Points with Speed Information')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/gps_speed_map.png")
    plt.close()
    
    # Plot speed distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(gps_cleaned['speed_mps'], kde=True, bins=30)
    plt.title('Distribution of Speed')
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/gps_speed_distribution.png")
    plt.close()
    
    # Plot accuracy distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(gps_cleaned['accuracy_m'], kde=True, bins=20)
    plt.title('Distribution of GPS Accuracy')
    plt.xlabel('Accuracy (m)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/gps_accuracy_distribution.png")
    plt.close()
    
    # Convert timestamp to datetime for time-based analysis
    gps_cleaned['timestamp_dt'] = pd.to_datetime(gps_cleaned['timestamp_ms'], unit='ms')
    
    # Add hour of day
    gps_cleaned['hour'] = gps_cleaned['timestamp_dt'].dt.hour
    
    # Plot speed by hour of day
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='hour', y='speed_mps', data=gps_cleaned)
    plt.title('Speed Distribution by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Speed (m/s)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/gps_speed_by_hour.png")
    plt.close()
    
    # Plot accuracy by hour of day
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='hour', y='accuracy_m', data=gps_cleaned)
    plt.title('GPS Accuracy by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Accuracy (m)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/gps_accuracy_by_hour.png")
    plt.close()
    
    # Analyze GPS provider information
    provider_counts = gps_cleaned['provider_id'].value_counts()
    
    plt.figure(figsize=(10, 6))
    provider_counts.plot(kind='bar')
    plt.title('GPS Provider Distribution')
    plt.xlabel('Provider ID')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("plots/gps_provider_distribution.png")
    plt.close()
    
    # Calculate time difference between consecutive GPS readings
    gps_cleaned = gps_cleaned.sort_values('timestamp_ms')
    gps_cleaned['prev_timestamp'] = gps_cleaned['timestamp_ms'].shift(1)
    gps_cleaned['time_diff_sec'] = (gps_cleaned['timestamp_ms'] - gps_cleaned['prev_timestamp']) / 1000.0
    
    # Plot time difference distribution (excluding outliers)
    plt.figure(figsize=(12, 6))
    sns.histplot(gps_cleaned['time_diff_sec'][
        (gps_cleaned['time_diff_sec'] > 0) & (gps_cleaned['time_diff_sec'] < 10)
    ], kde=True, bins=30)
    plt.title('Distribution of Time Between GPS Readings (0-10 sec)')
    plt.xlabel('Time Difference (sec)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/gps_time_diff_distribution.png")
    plt.close()
    
    # Calculate distance between consecutive points
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
    
    gps_cleaned['prev_lat'] = gps_cleaned['latitude_deg'].shift(1)
    gps_cleaned['prev_lon'] = gps_cleaned['longitude_deg'].shift(1)
    
    # Apply haversine distance function
    gps_cleaned['distance_m'] = gps_cleaned.apply(
        lambda row: haversine_distance(
            row['prev_lat'], row['prev_lon'], row['latitude_deg'], row['longitude_deg']
        ) if pd.notna(row['prev_lat']) and pd.notna(row['prev_lon']) else 0,
        axis=1
    )
    
    # Plot distance histogram (excluding outliers)
    plt.figure(figsize=(12, 6))
    sns.histplot(gps_cleaned['distance_m'][
        (gps_cleaned['distance_m'] > 0) & (gps_cleaned['distance_m'] < 1000)
    ], kde=True, bins=30)
    plt.title('Distribution of Distance Between Consecutive GPS Points (0-1000m)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/gps_distance_distribution.png")
    plt.close()
    
    # Calculate speed from distance/time
    gps_cleaned['calculated_speed_mps'] = np.where(
        gps_cleaned['time_diff_sec'] > 0,
        gps_cleaned['distance_m'] / gps_cleaned['time_diff_sec'],
        0
    )
    
    # Compare reported speed with calculated speed
    valid_speed_mask = (
        (gps_cleaned['speed_mps'] > 0) & 
        (gps_cleaned['calculated_speed_mps'] > 0) & 
        (gps_cleaned['calculated_speed_mps'] < 50)  # Filter out unrealistic calculated speeds
    )
    
    if valid_speed_mask.sum() > 0:
        plt.figure(figsize=(10, 10))
        plt.scatter(
            gps_cleaned.loc[valid_speed_mask, 'speed_mps'],
            gps_cleaned.loc[valid_speed_mask, 'calculated_speed_mps'],
            alpha=0.5
        )
        plt.plot([0, 50], [0, 50], 'r--')  # Line of perfect agreement
        plt.title('Reported Speed vs. Calculated Speed')
        plt.xlabel('Reported Speed (m/s)')
        plt.ylabel('Calculated Speed (m/s)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/gps_speed_comparison.png")
        plt.close()
    
    # Save cleaned dataframe
    gps_cleaned.to_csv('cleaned_gps_data.csv', index=False)
    print("\nCleaned GPS data saved to cleaned_gps_data.csv")
    
print("GPS Data Analysis completed!") 