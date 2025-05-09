import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os

print("Available files:")
for file in os.listdir('.'):
    if file.endswith('.csv'):
        file_size = os.path.getsize(file) / (1024 * 1024)  # size in MB
        print(f"{file} - {file_size:.2f} MB")

# Load location data
try:
    print("\nLoading location data...")
    location_df = pd.read_csv('Hips_Location.csv')
    print(f"Location data shape: {location_df.shape}")
    print("Location data columns:")
    print(location_df.columns.tolist())
    print("Location data sample:")
    print(location_df.head())
    print("Location data info:")
    location_df.info()
    print("\nLocation data statistics:")
    print(location_df.describe())
except Exception as e:
    print(f"Error loading location data: {e}")

# Load GPS data
try:
    print("\nLoading GPS data...")
    gps_df = pd.read_csv('Hips_GPS.csv')
    print(f"GPS data shape: {gps_df.shape}")
    print("GPS data columns:")
    print(gps_df.columns.tolist())
    print("GPS data sample:")
    print(gps_df.head())
    print("GPS data info:")
    gps_df.info()
    print("\nGPS data statistics:")
    print(gps_df.describe())
except Exception as e:
    print(f"Error loading GPS data: {e}")

# Try to load a sample of WiFi data (it's large)
try:
    print("\nLoading a sample of WiFi data...")
    # Read only the first 10,000 rows to avoid memory issues
    wifi_df = pd.read_csv('Hips_WiFi.csv', nrows=10000)
    print(f"WiFi data sample shape: {wifi_df.shape}")
    print("WiFi data columns:")
    print(wifi_df.columns.tolist())
    print("WiFi data sample:")
    print(wifi_df.head())
    print("WiFi data info:")
    wifi_df.info()
except Exception as e:
    print(f"Error loading WiFi data: {e}")

# Basic plotting function to visualize locations
def plot_locations(df, title='Location Data'):
    plt.figure(figsize=(10, 8))
    plt.scatter(df['longitude_deg'], df['latitude_deg'], c='blue', alpha=0.5)
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('location_map.png')
    print("Location map saved as location_map.png")

# Try to plot location data
try:
    if 'location_df' in locals() and 'latitude_deg' in location_df.columns and 'longitude_deg' in location_df.columns:
        plot_locations(location_df)
except Exception as e:
    print(f"Error plotting location data: {e}") 