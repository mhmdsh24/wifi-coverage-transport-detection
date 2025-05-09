import argparse
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from wifi_coverage_utils import (
    load_wifi_data, load_location_data, merge_wifi_and_location,
    add_engineered_features, create_grid_cells, calculate_grid_statistics,
    clean_signal_data
)
from wifi_anomaly_detection import detect_anomalies, add_anomaly_features_to_grid
from wifi_coverage_model import CoveragePredictionModel

def run_pipeline(wifi_path, location_path, rssi_threshold=-75, 
                 output_dir='output', models_dir='models', plots_dir='plots',
                 nrows=None, skip_anomaly=False, 
                 wifi_start=None, wifi_end=None, location_start=None, location_end=None):
    """
    Run the complete WiFi coverage prediction pipeline
    
    Parameters:
    -----------
    wifi_path : str
        Path to WiFi data CSV
    location_path : str
        Path to location data CSV
    rssi_threshold : float
        RSSI threshold for defining low coverage areas
    output_dir : str
        Directory to save output files
    models_dir : str
        Directory to save trained models
    plots_dir : str
        Directory to save plots
    nrows : int, optional
        Number of rows to load for testing
    skip_anomaly : bool
        Whether to skip anomaly detection
    wifi_start : int, optional
        Start timestamp for WiFi data filtering (ms)
    wifi_end : int, optional
        End timestamp for WiFi data filtering (ms)
    location_start : int, optional
        Start timestamp for location data filtering (ms)
    location_end : int, optional
        End timestamp for location data filtering (ms)
    
    Returns:
    --------
    results : dict
        Results from the pipeline including:
        - merged_df: Merged data
        - grid_stats: Grid statistics
        - model: Trained model
    """
    # Start time
    start_time = time.time()
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\n{'='*20} WiFi Coverage Prediction Pipeline {'='*20}\n")
    
    # Step 1: Load data
    print("\n--- Step 1: Loading and Cleaning Data ---")
    wifi_df = load_wifi_data(wifi_path, nrows)
    location_df = load_location_data(location_path, nrows)
    
    # Apply timestamp filtering if provided
    if wifi_start and wifi_end:
        print(f"Filtering WiFi data from {pd.to_datetime(wifi_start, unit='ms')} to {pd.to_datetime(wifi_end, unit='ms')}")
        wifi_df = wifi_df[(wifi_df['timestamp_ms'] >= wifi_start) & (wifi_df['timestamp_ms'] <= wifi_end)]
        print(f"Filtered WiFi data: {len(wifi_df)} records")
        
    if location_start and location_end:
        print(f"Filtering location data from {pd.to_datetime(location_start, unit='ms')} to {pd.to_datetime(location_end, unit='ms')}")
        location_df = location_df[(location_df['timestamp_ms'] >= location_start) & (location_df['timestamp_ms'] <= location_end)]
        print(f"Filtered location data: {len(location_df)} records")
    
    # Step 2: Merge data
    print("\n--- Step 2: Merging WiFi and Location Data ---")
    merged_df = merge_wifi_and_location(wifi_df, location_df)
    
    # Apply Hampel filter to clean signal data
    merged_df = clean_signal_data(merged_df)
    
    # Save merged data
    merged_path = os.path.join(output_dir, 'merged_wifi_location.csv')
    merged_df.to_csv(merged_path, index=False)
    print(f"Merged data saved to {merged_path}")
    
    # Step 3: Add engineered features
    print("\n--- Step 3: Feature Engineering ---")
    merged_df = add_engineered_features(merged_df)
    
    # Step 4: Create spatial grid cells
    print("\n--- Step 4: Creating Spatial Grid ---")
    merged_df = create_grid_cells(merged_df)
    
    # Step 5: Calculate grid statistics
    print("\n--- Step 5: Calculating Grid Statistics ---")
    grid_stats = calculate_grid_statistics(merged_df, rssi_threshold)
    
    # Save grid statistics
    grid_path = os.path.join(output_dir, 'grid_coverage_statistics.csv')
    grid_stats.to_csv(grid_path, index=False)
    print(f"Grid statistics saved to {grid_path}")
    
    # Step 6: Anomaly detection (optional)
    if not skip_anomaly:
        print("\n--- Step 6: Signal Anomaly Detection ---")
        merged_df, anomaly_detector = detect_anomalies(merged_df, output_dir=models_dir)
        
        # Add anomaly features to grid statistics
        grid_stats = add_anomaly_features_to_grid(grid_stats, merged_df)
        
        # Save updated grid statistics
        grid_stats.to_csv(grid_path, index=False)
    else:
        print("\n--- Step 6: Skipping Anomaly Detection ---")
    
    # Step 7: Coverage prediction modeling
    print("\n--- Step 7: Coverage Prediction Modeling ---")
    model = CoveragePredictionModel(
        output_dir=output_dir,
        models_dir=models_dir,
        plots_dir=plots_dir
    )
    
    # Run the full model pipeline
    grid_stats = model.run_full_pipeline(grid_stats)
    
    # Print completion message
    duration = time.time() - start_time
    print(f"\n{'='*20} Pipeline Complete {'='*20}")
    print(f"Total runtime: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Results saved to:")
    print(f"  - Output data: {output_dir}")
    print(f"  - Models: {models_dir}")
    print(f"  - Plots: {plots_dir}")
    
    # Return results
    return {
        'merged_df': merged_df,
        'grid_stats': grid_stats,
        'model': model
    }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='WiFi Coverage Prediction Pipeline')
    parser.add_argument('--wifi', type=str, required=True, help='Path to WiFi data CSV')
    parser.add_argument('--location', type=str, required=True, help='Path to location data CSV')
    parser.add_argument('--threshold', type=float, default=-75, help='RSSI threshold for low coverage (default: -75)')
    parser.add_argument('--output', type=str, default='output', help='Output directory (default: output)')
    parser.add_argument('--models', type=str, default='models', help='Models directory (default: models)')
    parser.add_argument('--plots', type=str, default='plots', help='Plots directory (default: plots)')
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to load (for testing)')
    parser.add_argument('--skip-anomaly', action='store_true', help='Skip anomaly detection')
    parser.add_argument('--wifi-start', type=int, default=None, help='Start timestamp for WiFi data filtering (ms)')
    parser.add_argument('--wifi-end', type=int, default=None, help='End timestamp for WiFi data filtering (ms)')
    parser.add_argument('--location-start', type=int, default=None, help='Start timestamp for location data filtering (ms)')
    parser.add_argument('--location-end', type=int, default=None, help='End timestamp for location data filtering (ms)')
    
    args = parser.parse_args()
    
    # Run the pipeline
    run_pipeline(
        wifi_path=args.wifi,
        location_path=args.location,
        rssi_threshold=args.threshold,
        output_dir=args.output,
        models_dir=args.models,
        plots_dir=args.plots,
        nrows=args.nrows,
        skip_anomaly=args.skip_anomaly,
        wifi_start=args.wifi_start,
        wifi_end=args.wifi_end,
        location_start=args.location_start,
        location_end=args.location_end
    ) 