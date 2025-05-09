import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
from datetime import datetime

# Import our modules
import eda_utils
import signal_anomaly_detection
import run_eda
import coverage_prediction_model
import predict_coverage
import transport_mode_detection
import threshold_sensitivity
import enhanced_visualization

def run_integrated_pipeline(wifi_path='Hips_WiFi.csv', 
                            location_path='Hips_Location.csv', 
                            gps_path='Hips_GPS.csv',
                            nrows=None, 
                            rssi_threshold=-75,
                            output_dir='output',
                            skip_eda=False,
                            skip_anomaly=False,
                            skip_model=False,
                            skip_transport_mode=False,
                            skip_threshold_analysis=False):
    """
    Run the complete integrated pipeline with anomaly detection
    
    Parameters:
    -----------
    wifi_path : str, default='Hips_WiFi.csv'
        Path to WiFi data CSV
    location_path : str, default='Hips_Location.csv'
        Path to location data CSV
    gps_path : str, default='Hips_GPS.csv'
        Path to GPS data CSV
    nrows : int, optional
        Number of rows to load (for testing with smaller datasets)
    rssi_threshold : float, default=-75
        RSSI threshold for defining low coverage
    output_dir : str, default='output'
        Directory to save results
    skip_eda : bool, default=False
        Skip EDA processing if data already exists
    skip_anomaly : bool, default=False
        Skip anomaly detection
    skip_model : bool, default=False
        Skip model training if models already exist
    skip_transport_mode : bool, default=False
        Skip transport mode detection
    skip_threshold_analysis : bool, default=False
        Skip threshold sensitivity analysis
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Start timer
    start_time = time.time()
    
    # Setup logging
    log_file = f"{output_dir}/pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    def log_message(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    log_message("Starting integrated WiFi/cellular coverage prediction pipeline")
    log_message(f"Parameters: rssi_threshold={rssi_threshold}, nrows={nrows if nrows else 'all'}")
    
    # Step 1: Run EDA and data preparation
    if not skip_eda and (not os.path.exists('cleaned_wifi_data.csv') or 
                         not os.path.exists('cleaned_location_data.csv') or
                         not os.path.exists('cleaned_gps_data.csv')):
        log_message("\n=== STEP 1: EDA and Data Preparation ===")
        try:
            cleaned_wifi_df, cleaned_location_df, cleaned_gps_df, merged_df, grid_stats = run_eda.run_full_eda_pipeline(
                nrows=nrows, 
                rssi_threshold=rssi_threshold
            )
            log_message("EDA and data preparation completed successfully")
        except Exception as e:
            log_message(f"Error during EDA: {str(e)}")
            return
    else:
        log_message("\n=== STEP 1: Loading Preprocessed Data ===")
        try:
            cleaned_wifi_df = pd.read_csv('cleaned_wifi_data.csv')
            cleaned_location_df = pd.read_csv('cleaned_location_data.csv')
            cleaned_gps_df = pd.read_csv('cleaned_gps_data.csv')
            
            try:
                merged_df = pd.read_csv('merged_wifi_location.csv')
                
                # Fix missing required columns
                log_message("Checking for required columns and adding if missing...")
                # Ensure we have timestamps in correct format
                if 'timestamp_dt' not in merged_df.columns and 'timestamp_ms' in merged_df.columns:
                    merged_df['timestamp_dt'] = pd.to_datetime(merged_df['timestamp_ms'], unit='ms')
                    log_message("Added timestamp_dt column")
                
                # Sort data by BSSID and timestamp for time-series operations
                merged_df = merged_df.sort_values(['bssid', 'timestamp_ms'])
                
                # Add rssi_change column if missing
                if 'rssi_change' not in merged_df.columns:
                    merged_df['rssi_change'] = merged_df.groupby('bssid')['rssi'].diff()
                    log_message("Added rssi_change column")
                
                # Add rssi_rolling_std column if missing
                if 'rssi_rolling_std' not in merged_df.columns:
                    window_size = 5
                    merged_df['rssi_rolling_std'] = merged_df.groupby('bssid')['rssi'].transform(
                        lambda x: x.rolling(window=window_size, min_periods=1).std()
                    )
                    log_message("Added rssi_rolling_std column")
                    
                    # Save the updated merged data
                    merged_df.to_csv('merged_wifi_location.csv', index=False)
                    log_message("Updated merged_wifi_location.csv with missing columns")
                
                grid_stats = pd.read_csv('grid_coverage_statistics.csv')
                log_message(f"Loaded preprocessed data: {len(cleaned_wifi_df)} WiFi records, {len(cleaned_location_df)} location records")
            except FileNotFoundError:
                log_message("Merging data and computing grid statistics...")
                merged_df = eda_utils.merge_wifi_location_data(cleaned_wifi_df, cleaned_location_df)
                
                # Add required columns
                if 'rssi_change' not in merged_df.columns:
                    merged_df = merged_df.sort_values(['bssid', 'timestamp_ms'])
                    merged_df['rssi_change'] = merged_df.groupby('bssid')['rssi'].diff()
                    log_message("Added rssi_change column")
                
                if 'rssi_rolling_std' not in merged_df.columns:
                    window_size = 5
                    merged_df['rssi_rolling_std'] = merged_df.groupby('bssid')['rssi'].transform(
                        lambda x: x.rolling(window=window_size, min_periods=1).std()
                    )
                    log_message("Added rssi_rolling_std column")
                
                grid_stats = eda_utils.compute_grid_statistics(merged_df, rssi_threshold)
                
                # Save the merged data with all required columns
                merged_df.to_csv('merged_wifi_location.csv', index=False)
                log_message("Saved merged_wifi_location.csv with all required columns")
        except FileNotFoundError:
            log_message("Preprocessed data not found. Please run EDA first or disable skip_eda.")
            return
    
    # Step 2: Run Transport Mode Detection
    if not skip_transport_mode:
        log_message("\n=== STEP 2: Transport Mode Detection ===")
        try:
            # Check if transport modes already exist
            transport_file = f"{output_dir}/transport_modes.csv"
            if os.path.exists(transport_file):
                log_message(f"Loading existing transport mode data from {transport_file}")
                transport_df = pd.read_csv(transport_file)
                
                # Merge with merged_df if needed
                if 'time_window' in transport_df.columns and 'time_window' in merged_df.columns:
                    # First create time_window if missing
                    if 'time_window' not in merged_df.columns:
                        window_size_ms = 5000
                        merged_df['time_window'] = (merged_df['timestamp_ms'] // window_size_ms) * window_size_ms
                    
                    # Keep only necessary columns from transport_df for merging
                    transport_cols = ['time_window', 'predicted_mode', 'predicted_mode_code'] + \
                                     [col for col in transport_df.columns if col.startswith('prob_')]
                    
                    merged_df = pd.merge(
                        merged_df,
                        transport_df[transport_cols],
                        on='time_window',
                        how='left'
                    )
                    log_message(f"Merged transport mode data with main dataframe (modes: {merged_df['predicted_mode'].unique()})")
                else:
                    log_message("Could not merge transport modes - missing time_window column")
            else:
                log_message("Detecting transport modes...")
                transport_mode_start = time.time()
                
                # Run transport mode detection
                merged_df, detector = transport_mode_detection.detect_transport_modes(
                    cleaned_gps_df, 
                    cleaned_wifi_df,
                    merged_df=merged_df,
                    output_dir=output_dir,
                    models_dir='models',
                    plots_dir='plots'
                )
                
                # Save transport mode data
                merged_df.to_csv(transport_file, index=False)
                
                # Update merged_wifi_location.csv with transport modes
                merged_df.to_csv('merged_wifi_location.csv', index=False)
                
                transport_mode_time = time.time() - transport_mode_start
                log_message(f"Transport mode detection completed in {transport_mode_time:.2f} seconds")
                log_message(f"Transport modes detected: {merged_df['predicted_mode'].unique()}")
                
                # Aggregate transport modes to grid level
                log_message("Aggregating transport modes to grid level...")
                
                # Ensure we have grid coordinates
                if 'lat_grid' not in merged_df.columns or 'lon_grid' not in merged_df.columns:
                    grid_precision = 4
                    merged_df['lat_grid'] = np.round(merged_df['latitude_deg'], grid_precision)
                    merged_df['lon_grid'] = np.round(merged_df['longitude_deg'], grid_precision)
                
                # Aggregate transport modes by grid cell
                grid_transport = merged_df.groupby(['lat_grid', 'lon_grid']).agg({
                    'predicted_mode': lambda x: x.value_counts().index[0],  # Most common mode
                    'predicted_mode_code': 'mean'
                }).reset_index()
                
                # Merge transport modes to grid stats
                grid_stats = pd.merge(
                    grid_stats,
                    grid_transport[['lat_grid', 'lon_grid', 'predicted_mode', 'predicted_mode_code']],
                    on=['lat_grid', 'lon_grid'],
                    how='left'
                )
                
                # Save updated grid stats
                grid_stats.to_csv('grid_coverage_statistics.csv', index=False)
                log_message("Updated grid statistics with transport mode information")
            
        except Exception as e:
            log_message(f"Error during transport mode detection: {str(e)}")
            log_message("Continuing pipeline without transport mode information...")
    else:
        log_message("\n=== STEP 2: Skipping Transport Mode Detection ===")
    
    # Step 3: Run Anomaly Detection
    if not skip_anomaly:
        log_message("\n=== STEP 3: Signal Anomaly Detection ===")
        try:
            anomaly_start_time = time.time()
            merged_with_anomalies, anomalies_df = signal_anomaly_detection.detect_signal_anomalies(
                cleaned_wifi_df, 
                cleaned_location_df, 
                threshold=rssi_threshold
            )
            
            # Save anomalies to CSV
            if len(anomalies_df) > 0:
                anomalies_df.to_csv(f'{output_dir}/signal_anomalies.csv', index=False)
                log_message(f"Detected and saved {len(anomalies_df)} signal anomalies")
                
                # Create summary visualization
                plt.figure(figsize=(12, 10))
                plt.scatter(
                    merged_df[merged_df['rssi'] >= rssi_threshold]['longitude_deg'], 
                    merged_df[merged_df['rssi'] >= rssi_threshold]['latitude_deg'], 
                    c='green', 
                    alpha=0.3, 
                    s=10,
                    label='Good Signal'
                )
                plt.scatter(
                    merged_df[merged_df['rssi'] < rssi_threshold]['longitude_deg'], 
                    merged_df[merged_df['rssi'] < rssi_threshold]['latitude_deg'], 
                    c='red', 
                    alpha=0.3, 
                    s=10,
                    label='Low Signal'
                )
                plt.scatter(
                    anomalies_df['longitude_deg'], 
                    anomalies_df['latitude_deg'], 
                    c='purple', 
                    marker='*', 
                    s=50, 
                    alpha=0.7, 
                    label='Signal Anomaly'
                )
                plt.title('Signal Strength and Detected Anomalies')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f'{output_dir}/anomaly_detection_summary.png')
                plt.close()
                
                # Integrate anomalies with transport modes if available
                if 'predicted_mode' in merged_df.columns:
                    log_message("Analyzing anomalies by transport mode...")
                    anomalies_with_modes = pd.merge(
                        anomalies_df,
                        merged_df[['timestamp_ms', 'predicted_mode']],
                        on='timestamp_ms',
                        how='left'
                    )
                    
                    # Count anomalies by transport mode
                    anomalies_by_mode = anomalies_with_modes['predicted_mode'].value_counts()
                    log_message("Anomalies by transport mode:")
                    for mode, count in anomalies_by_mode.items():
                        log_message(f"  {mode}: {count} anomalies")
                    
                    # Create visualization of anomalies by transport mode
                    plt.figure(figsize=(10, 6))
                    anomalies_by_mode.plot(kind='bar')
                    plt.title('Signal Anomalies by Transport Mode')
                    plt.xlabel('Transport Mode')
                    plt.ylabel('Number of Anomalies')
                    plt.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/anomalies_by_transport_mode.png')
                    plt.close()
            else:
                log_message("No signal anomalies detected")
                
            anomaly_time = time.time() - anomaly_start_time
            log_message(f"Anomaly detection completed in {anomaly_time:.2f} seconds")
        except Exception as e:
            log_message(f"Error during anomaly detection: {str(e)}")
            log_message("Continuing without anomaly detection...")
    else:
        log_message("\n=== STEP 3: Skipping Anomaly Detection ===")
    
    # Step 4: Run Threshold Sensitivity Analysis
    if not skip_threshold_analysis:
        log_message("\n=== STEP 4: Threshold Sensitivity Analysis ===")
        try:
            threshold_start_time = time.time()
            
            # Run threshold analysis
            analyzer, recommendations = threshold_sensitivity.run_threshold_analysis(
                merged_df,
                threshold_range=(-85, -65, 1),
                output_dir=output_dir,
                plots_dir='plots'
            )
            
            # Log recommendations
            log_message("\nRSSI Threshold Recommendations:")
            for key, recs in recommendations.items():
                log_message(f"\nAnalysis: {key}")
                for scenario, values in recs.items():
                    log_message(f"  {scenario}:")
                    for metric, value in values.items():
                        log_message(f"    {metric}: {value}")
            
            threshold_time = time.time() - threshold_start_time
            log_message(f"Threshold sensitivity analysis completed in {threshold_time:.2f} seconds")
        except Exception as e:
            log_message(f"Error during threshold sensitivity analysis: {str(e)}")
            log_message("Continuing without threshold sensitivity analysis...")
    else:
        log_message("\n=== STEP 4: Skipping Threshold Sensitivity Analysis ===")
    
    # Step 5: Train or load prediction models
    log_message("\n=== STEP 5: Coverage Prediction Modeling ===")
    models_exist = (os.path.exists('models/rf_coverage_model.pkl') and 
                   os.path.exists('models/gb_coverage_model.pkl') and
                   os.path.exists('models/knn_spatial_model.pkl'))
    
    if not skip_model or not models_exist:
        try:
            log_message("Training coverage prediction models...")
            model_file = f"{output_dir}/model_training_output.txt"
            # Redirect standard output to capture model training results
            original_stdout = sys.stdout
            with open(model_file, 'w') as f:
                sys.stdout = f
                # Import and run the model training script
                import coverage_prediction_model
                sys.stdout = original_stdout
            
            log_message(f"Model training completed. Full output saved to {model_file}")
        except Exception as e:
            log_message(f"Error during model training: {str(e)}")
            return
    else:
        log_message("Using pre-trained models")
    
    # Step 6: Generate enhanced predictions and visualizations
    log_message("\n=== STEP 6: Enhanced Coverage Prediction and Visualization ===")
    try:
        # Load models
        models = predict_coverage.load_models()
        
        # Prepare data from cleaned sources
        merged_df, grid_stats = predict_coverage.prepare_data(
            'cleaned_wifi_data.csv', 
            'cleaned_location_data.csv'
        )
        
        # Detect anomalies if not skipped
        if not skip_anomaly:
            merged_df = predict_coverage.detect_anomalies(merged_df, models)
        
        # Create enhanced visualizations
        log_message("Generating enhanced visualizations...")
        visualizer = enhanced_visualization.enhance_visualizations(
            grid_stats,
            merged_df,
            output_dir=output_dir,
            plots_dir='plots'
        )
        log_message("Enhanced visualizations created successfully")
        
        # Generate coverage map
        log_message("Generating coverage prediction map...")
        coverage_map = predict_coverage.generate_coverage_map(
            grid_stats, 
            models,
            with_anomalies=not skip_anomaly,
            with_transport_modes=not skip_transport_mode
        )
        log_message("Coverage prediction map generated successfully")
        
    except Exception as e:
        log_message(f"Error during prediction and visualization: {str(e)}")
    
    # Calculate total run time
    total_time = time.time() - start_time
    log_message(f"\nTotal pipeline execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    log_message("\n=== Pipeline Completed Successfully ===")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the integrated WiFi coverage prediction pipeline')
    parser.add_argument('--wifi', type=str, default='Hips_WiFi.csv', help='WiFi data CSV file')
    parser.add_argument('--location', type=str, default='Hips_Location.csv', help='Location data CSV file')
    parser.add_argument('--gps', type=str, default='Hips_GPS.csv', help='GPS data CSV file')
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to load (for testing)')
    parser.add_argument('--threshold', type=float, default=-75, help='RSSI threshold for low coverage')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--skip-eda', action='store_true', help='Skip EDA processing')
    parser.add_argument('--skip-anomaly', action='store_true', help='Skip anomaly detection')
    parser.add_argument('--skip-model', action='store_true', help='Skip model training')
    parser.add_argument('--skip-transport-mode', action='store_true', help='Skip transport mode detection')
    parser.add_argument('--skip-threshold-analysis', action='store_true', help='Skip threshold sensitivity analysis')
    
    args = parser.parse_args()
    
    run_integrated_pipeline(
        wifi_path=args.wifi,
        location_path=args.location,
        gps_path=args.gps,
        nrows=args.nrows,
        rssi_threshold=args.threshold,
        output_dir=args.output,
        skip_eda=args.skip_eda,
        skip_anomaly=args.skip_anomaly,
        skip_model=args.skip_model,
        skip_transport_mode=args.skip_transport_mode,
        skip_threshold_analysis=args.skip_threshold_analysis
    ) 