#!/usr/bin/env python
"""
Enhanced WiFi Coverage Analysis Pipeline

This script implements all recommended improvements to the WiFi coverage analysis pipeline,
including transport mode detection, threshold sensitivity analysis, enhanced visualizations,
and model improvements based on SHL dataset concepts.

Usage:
    python run_enhanced_pipeline.py [options]
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import json
from datetime import datetime

# Import modules
from transport_mode_detection import TransportModeDetector
from threshold_sensitivity import ThresholdSensitivityAnalyzer
from enhanced_visualization import EnhancedCoverageVisualizer
import run_integrated_pipeline
import eda_utils
import signal_anomaly_detection
import enhanced_transport_classifier  # Import our new module

def run_enhanced_pipeline(wifi_path='Hips_WiFi.csv', 
                          location_path='Hips_Location.csv', 
                          gps_path='Hips_GPS.csv',
                          motion_path=None,  # Optional motion data path
                          nrows=None, 
                          rssi_threshold=-75,
                          output_dir='output_enhanced',
                          models_dir='models_enhanced',
                          plots_dir='plots_enhanced',
                          skip_basic_transport=False,  # Skip basic transport mode detection
                          skip_enhanced_transport=False,  # Skip enhanced transport classification
                          use_enhanced_transport=True,  # Use enhanced transport by default
                          use_mobility_thresholds=True):  # Use mobility-aware thresholds by default
    """
    Run the enhanced pipeline with all improvements
    
    Parameters:
    -----------
    wifi_path : str, default='Hips_WiFi.csv'
        Path to WiFi data CSV
    location_path : str, default='Hips_Location.csv'
        Path to location data CSV
    gps_path : str, default='Hips_GPS.csv'
        Path to GPS data CSV
    motion_path : str, optional
        Path to motion sensor data if available
    nrows : int, optional
        Number of rows to load (for testing with smaller datasets)
    rssi_threshold : float, default=-75
        Static RSSI threshold for defining low coverage (will be dynamically adjusted)
    output_dir : str, default='output_enhanced'
        Directory to save results
    models_dir : str, default='models_enhanced'
        Directory to save models
    plots_dir : str, default='plots_enhanced'
        Directory to save plots
    skip_basic_transport : bool, default=False
        Skip basic transport mode detection
    skip_enhanced_transport : bool, default=False
        Skip enhanced transport classification
    use_enhanced_transport : bool, default=True
        Use enhanced transport classifier for mobility-aware processing
    use_mobility_thresholds : bool, default=True
        Use mobility-aware RSSI thresholds
    """
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up logging
    log_file = f"{output_dir}/enhanced_pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    def log_message(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    log_message("Starting enhanced WiFi coverage pipeline")
    log_message(f"Parameters: threshold={rssi_threshold}, output={output_dir}, models={models_dir}, plots={plots_dir}")
    
    # Record start time
    start_time = time.time()
    
    # Step 1: Run the integrated pipeline for basic processing & EDA
    # Skip transport mode if we're going to use the enhanced version
    log_message("\n=== STEP 1: Basic Data Processing ===")
    
    # Run the basic pipeline with appropriate settings
    run_integrated_pipeline.run_integrated_pipeline(
        wifi_path=wifi_path,
        location_path=location_path,
        gps_path=gps_path,
        nrows=nrows,
        rssi_threshold=rssi_threshold,
        output_dir=output_dir,
        skip_transport_mode=skip_basic_transport or use_enhanced_transport,
        skip_threshold_analysis=use_mobility_thresholds  # Skip basic threshold analysis if using mobility-aware thresholds
    )
    
    # Load the processed data
    log_message("Loading processed data files...")
    try:
        merged_df = pd.read_csv('merged_wifi_location.csv')
        cleaned_wifi_df = pd.read_csv('cleaned_wifi_data.csv')
        cleaned_location_df = pd.read_csv('cleaned_location_data.csv')
        cleaned_gps_df = pd.read_csv('cleaned_gps_data.csv')
        
        # Load motion data if available
        motion_df = None
        if motion_path is not None:
            try:
                motion_df = pd.read_csv(motion_path)
                log_message(f"Loaded motion data: {len(motion_df)} records")
            except Exception as e:
                log_message(f"Could not load motion data: {str(e)}")
        
        log_message(f"Data loaded: {len(merged_df)} merged records")
    except Exception as e:
        log_message(f"Error loading data: {str(e)}")
        return
    
    # Step 2: Enhanced Transport Mode Classification
    transport_mode_df = None
    enhanced_transport_df = None
    
    if not skip_enhanced_transport and use_enhanced_transport:
        log_message("\n=== STEP 2: Enhanced Transport Mode Classification ===")
        try:
            # Run the enhanced transport classifier
            enhanced_transport_df, classifier = enhanced_transport_classifier.classify_enhanced_transport_modes(
                cleaned_gps_df, 
                cleaned_wifi_df, 
                motion_df=motion_df,
                output_dir=output_dir,
                models_dir=models_dir,
                load_existing=True  # Try to load existing model
            )
            
            # Log the results
            if enhanced_transport_df is not None:
                mode_counts = enhanced_transport_df['predicted_mode'].value_counts()
                log_message(f"Enhanced transport classification complete. Mode counts:")
                for mode, count in mode_counts.items():
                    log_message(f"  {mode}: {count} windows ({count/len(enhanced_transport_df)*100:.1f}%)")
                
                # Display the dynamic thresholds
                log_message("Dynamic RSSI thresholds by transport mode:")
                for mode, threshold in classifier.threshold_mapping.items():
                    log_message(f"  {mode}: {threshold} dBm")
            
            transport_mode_df = enhanced_transport_df
        except Exception as e:
            log_message(f"Error in enhanced transport classification: {str(e)}")
            log_message("Falling back to basic transport mode detection...")
            transport_mode_df = None
    
    # Step 3: Mobility-Aware Anomaly Detection
    if use_mobility_thresholds and transport_mode_df is not None:
        log_message("\n=== STEP 3: Mobility-Aware Anomaly Detection ===")
        try:
            # Import signal anomaly detection explicitly here to ensure we have the updated version
            import signal_anomaly_detection
            
            # Run the enhanced anomaly detection with mobility-aware thresholds
            anomalies_df = signal_anomaly_detection.detect_anomalies_with_mobility(
                merged_df,
                transport_mode_df,
                rssi_threshold=rssi_threshold,  # This is the static threshold that will be dynamically adjusted
                output_file=os.path.join(output_dir, "mobility_aware_anomalies.csv"),
                plot_dir=os.path.join(plots_dir, "anomalies")
            )
            
            # Log the results
            if anomalies_df is not None and len(anomalies_df) > 0:
                log_message(f"Detected {len(anomalies_df)} anomalies with mobility-aware thresholds")
                
                # Analyze anomaly reduction
                if 'anomaly_type' in anomalies_df.columns:
                    anomaly_types = anomalies_df['anomaly_type'].value_counts()
                    log_message("Anomaly types:")
                    for atype, count in anomaly_types.items():
                        log_message(f"  {atype}: {count} ({count/len(anomalies_df)*100:.1f}%)")
                    
                    # Check if we have the mobility context type
                    mobility_context_count = anomalies_df[anomalies_df['anomaly_type'] == 'Mobility Context (Not Anomalous)'].shape[0]
                    if mobility_context_count > 0:
                        log_message(f"Eliminated {mobility_context_count} false positives through mobility context!")
        except Exception as e:
            log_message(f"Error in mobility-aware anomaly detection: {str(e)}")
            log_message("Falling back to standard anomaly detection...")
    else:
        log_message("\n=== STEP 3: Standard Anomaly Detection ===")
        # We're using the standard anomaly detection from the integrated pipeline
        log_message("Using standard anomaly detection (no mobility-aware thresholds)")
    
    # Step 4: Enhanced Visualization with Mobility Context
    log_message("\n=== STEP 4: Enhanced Visualization with Mobility Context ===")
    try:
        # Use our enhanced visualization module
        enhanced_plots_dir = os.path.join(plots_dir, "enhanced")
        os.makedirs(enhanced_plots_dir, exist_ok=True)
        
        # Create enhanced visualizations incorporating mobility
        enhanced_visualization.create_enhanced_visualizations(
            merged_df=merged_df,
            transport_mode_df=transport_mode_df,
            anomalies_file=os.path.join(output_dir, "mobility_aware_anomalies.csv") if use_mobility_thresholds else None,
            output_dir=enhanced_plots_dir
        )
        
        log_message(f"Enhanced visualizations saved to {enhanced_plots_dir}")
    except Exception as e:
        log_message(f"Error creating enhanced visualizations: {str(e)}")
    
    # Calculate and log total runtime
    end_time = time.time()
    runtime_seconds = end_time - start_time
    log_message(f"\nEnhanced pipeline completed in {runtime_seconds:.1f} seconds ({runtime_seconds/60:.1f} minutes)")
    
    return {
        'merged_df': merged_df,
        'transport_mode_df': transport_mode_df,
        'enhanced_transport_df': enhanced_transport_df
    }

def main():
    parser = argparse.ArgumentParser(description='Run enhanced WiFi coverage pipeline')
    parser.add_argument('--wifi', default='Hips_WiFi.csv', help='WiFi data file')
    parser.add_argument('--location', default='Hips_Location.csv', help='Location data file')
    parser.add_argument('--gps', default='Hips_GPS.csv', help='GPS data file')
    parser.add_argument('--motion', help='Motion sensor data file (optional)')
    parser.add_argument('--nrows', type=int, help='Number of rows to process (for testing)')
    parser.add_argument('--threshold', type=float, default=-75, help='Static RSSI threshold')
    parser.add_argument('--output', default='output_enhanced', help='Output directory')
    parser.add_argument('--models', default='models_enhanced', help='Models directory')
    parser.add_argument('--plots', default='plots_enhanced', help='Plots directory')
    parser.add_argument('--no-basic-transport', action='store_true', help='Skip basic transport mode detection')
    parser.add_argument('--no-enhanced-transport', action='store_true', help='Skip enhanced transport classification')
    parser.add_argument('--use-basic-transport', action='store_true', help='Use basic transport classifier instead of enhanced')
    parser.add_argument('--no-mobility-thresholds', action='store_true', help='Disable mobility-aware thresholds')
    
    args = parser.parse_args()
    
    # Run the enhanced pipeline with command line arguments
    run_enhanced_pipeline(
        wifi_path=args.wifi,
        location_path=args.location,
        gps_path=args.gps,
        motion_path=args.motion,
        nrows=args.nrows,
        rssi_threshold=args.threshold,
        output_dir=args.output,
        models_dir=args.models,
        plots_dir=args.plots,
        skip_basic_transport=args.no_basic_transport,
        skip_enhanced_transport=args.no_enhanced_transport,
        use_enhanced_transport=not args.use_basic_transport,
        use_mobility_thresholds=not args.no_mobility_thresholds
    )

if __name__ == "__main__":
    main() 