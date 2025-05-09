import os
import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import eda_utils as utils
import signal_anomaly_detection as anomaly_detector

print("Starting EDA Process...")

# Check if required packages are installed
required_packages = [
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn'
]

# Function to install packages
def install_package(package):
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Check and install required packages
for package in required_packages:
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"{package} is not installed. Installing...")
        install_package(package)
        print(f"{package} has been installed.")

# Run the integrated EDA script
print("\nRunning complete EDA analysis...")
try:
    exec(open("eda_integration.py").read())
    print("\nEDA completed successfully!")
except Exception as e:
    print(f"\nError running EDA: {e}")
    print("Please check the individual scripts for more details.")

def run_full_eda_pipeline(nrows=None, rssi_threshold=-75):
    """
    Run the complete EDA pipeline including anomaly detection preparation.
    
    Parameters:
    -----------
    nrows : int, optional
        Number of rows to load (for testing with smaller datasets)
    rssi_threshold : float, default=-75
        RSSI threshold for defining low coverage
    """
    print("\n=== Starting Full EDA Pipeline ===\n")
    
    # 1. Load Data
    print("\n--- Loading Data ---\n")
    wifi_df, location_df, gps_df = utils.load_wifi_location_gps_data(nrows=nrows)
    
    # 2. Clean Data
    print("\n--- Cleaning Data ---\n")
    cleaned_wifi_df = utils.clean_wifi_data(wifi_df)
    cleaned_location_df = utils.clean_location_data(location_df)
    cleaned_gps_df = utils.clean_gps_data(gps_df)
    
    # 3. Merge Data
    print("\n--- Merging Data ---\n")
    merged_df = utils.merge_wifi_location_data(cleaned_wifi_df, cleaned_location_df)
    
    # 4. Compute Grid Statistics
    print("\n--- Computing Grid Statistics ---\n")
    grid_stats = utils.compute_grid_statistics(merged_df, rssi_threshold)
    
    # 5. Generate Visualizations
    print("\n--- Generating Visualizations ---\n")
    utils.visualize_coverage(merged_df, grid_stats, rssi_threshold)
    
    # 6. Save Processed Data
    print("\n--- Saving Processed Data ---\n")
    utils.save_processed_data(
        cleaned_wifi_df, 
        cleaned_location_df, 
        cleaned_gps_df, 
        merged_df, 
        grid_stats
    )
    
    # 7. Prepare for Anomaly Detection
    print("\n--- Preparing for Anomaly Detection ---\n")
    merged_df_prepared = utils.prepare_for_anomaly_detection(merged_df, rssi_threshold)
    
    # 8. Run Anomaly Detection
    print("\n--- Running Anomaly Detection ---\n")
    try:
        merged_with_anomalies, anomalies_df = anomaly_detector.detect_signal_anomalies(
            cleaned_wifi_df, 
            cleaned_location_df, 
            threshold=rssi_threshold
        )
        
        # Save anomalies to CSV
        if len(anomalies_df) > 0:
            anomalies_df.to_csv('signal_anomalies.csv', index=False)
            print(f"Saved {len(anomalies_df)} detected anomalies to signal_anomalies.csv")
    except Exception as e:
        print(f"Error during anomaly detection: {e}")
        print("Continuing with rest of pipeline...")
    
    print("\n=== EDA Pipeline Completed ===\n")
    
    return (cleaned_wifi_df, cleaned_location_df, cleaned_gps_df, 
            merged_df, grid_stats)

if __name__ == "__main__":
    # Set nrows to None to process all data, or a number for testing
    run_full_eda_pipeline(nrows=None) 