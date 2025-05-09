"""
Generate a summary of EDA results from processed data files.
This script reads various data files and generates a summary JSON file for the web visualization.
"""

import os
import json
import pandas as pd
import numpy as np

def generate_eda_summary():
    """Generate a summary of EDA results and save to JSON file"""
    eda_summary = {}
    
    # WiFi data summary
    if os.path.exists('cleaned_wifi_data.csv'):
        print("Processing WiFi data...")
        try:
            # Read with lower memory usage by only loading necessary columns
            wifi_df = pd.read_csv('cleaned_wifi_data.csv', usecols=['bssid', 'ssid', 'rssi'])
            eda_summary['total_wifi_records'] = len(wifi_df)
            eda_summary['unique_bssids'] = len(wifi_df['bssid'].unique())
            hidden_ssids = wifi_df['ssid'].isna().sum()
            eda_summary['hidden_ssids_count'] = int(hidden_ssids)
            eda_summary['hidden_ssids_percentage'] = (hidden_ssids / len(wifi_df)) * 100
            eda_summary['rssi_min_wifi'] = float(wifi_df['rssi'].min())
            eda_summary['rssi_max_wifi'] = float(wifi_df['rssi'].max())
            eda_summary['rssi_mean_wifi'] = float(wifi_df['rssi'].mean())
            print(f"WiFi summary: {len(wifi_df)} records, {eda_summary['unique_bssids']} unique BSSIDs")
        except Exception as e:
            print(f"Error processing WiFi data: {e}")
    
    # Location data summary
    if os.path.exists('cleaned_location_data.csv'):
        print("Processing location data...")
        try:
            location_df = pd.read_csv('cleaned_location_data.csv')
            eda_summary['total_location_records'] = len(location_df)
            print(f"Location summary: {len(location_df)} records")
        except Exception as e:
            print(f"Error processing location data: {e}")
    
    # GPS data summary
    if os.path.exists('cleaned_gps_data.csv'):
        print("Processing GPS data...")
        try:
            gps_df = pd.read_csv('cleaned_gps_data.csv')
            eda_summary['total_gps_records'] = len(gps_df)
            print(f"GPS summary: {len(gps_df)} records")
        except Exception as e:
            print(f"Error processing GPS data: {e}")
    
    # Merged data summary
    if os.path.exists('merged_wifi_location.csv'):
        print("Processing merged data...")
        try:
            # Read with lower memory usage by only loading necessary columns
            merged_df = pd.read_csv('merged_wifi_location.csv', 
                                   usecols=['bssid', 'rssi', 'lat', 'lon'])
            eda_summary['total_merged_records'] = len(merged_df)
            if 'total_wifi_records' in eda_summary and eda_summary['total_wifi_records'] > 0:
                eda_summary['join_rate'] = (eda_summary['total_merged_records'] / eda_summary['total_wifi_records']) * 100
            print(f"Merged summary: {len(merged_df)} records")
        except Exception as e:
            print(f"Error processing merged data: {e}")
    
    # Grid coverage statistics
    if os.path.exists('grid_coverage_statistics.csv'):
        print("Processing grid statistics...")
        try:
            grid_stats = pd.read_csv('grid_coverage_statistics.csv')
            eda_summary['total_grid_cells'] = len(grid_stats)
            eda_summary['grid_cell_size_meters'] = 10  # Default value, should be adjusted if known
            
            if 'poor_coverage' in grid_stats.columns:
                eda_summary['low_coverage_cells'] = int(grid_stats['poor_coverage'].sum())
                if eda_summary['total_grid_cells'] > 0:
                    eda_summary['low_coverage_percentage'] = (eda_summary['low_coverage_cells'] / eda_summary['total_grid_cells']) * 100
            
            if 'rssi_mean' in grid_stats.columns:
                eda_summary['rssi_mean'] = float(grid_stats['rssi_mean'].mean())
                eda_summary['rssi_min'] = float(grid_stats['rssi_min'].min() if 'rssi_min' in grid_stats.columns else grid_stats['rssi_mean'].min())
                eda_summary['rssi_max'] = float(grid_stats['rssi_max'].max() if 'rssi_max' in grid_stats.columns else grid_stats['rssi_mean'].max())
            
            print(f"Grid summary: {len(grid_stats)} cells")
        except Exception as e:
            print(f"Error processing grid statistics: {e}")
    
    # Anomaly data summary
    if os.path.exists('signal_anomalies.csv'):
        print("Processing anomaly data...")
        try:
            anomalies_df = pd.read_csv('signal_anomalies.csv')
            eda_summary['anomaly_count'] = len(anomalies_df)
            if 'total_wifi_records' in eda_summary and eda_summary['total_wifi_records'] > 0:
                eda_summary['anomaly_percentage'] = (eda_summary['anomaly_count'] / eda_summary['total_wifi_records']) * 100
            print(f"Anomaly summary: {len(anomalies_df)} anomalies")
        except Exception as e:
            print(f"Error processing anomaly data: {e}")
    
    # Save summary to JSON
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'eda_summary.json')
    
    # Convert NumPy types to Python native types for JSON serialization
    for key, value in eda_summary.items():
        if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
            eda_summary[key] = int(value)
        elif isinstance(value, (np.float64, np.float32, np.float16)):
            eda_summary[key] = float(value)
    
    with open(summary_path, 'w') as f:
        json.dump(eda_summary, f, indent=4)
    
    print(f"EDA summary saved to {summary_path}")
    return eda_summary

if __name__ == "__main__":
    print("Generating EDA summary...")
    summary = generate_eda_summary()
    print("Done!") 