#!/usr/bin/env python
"""
WiFi Coverage Analysis - Demo Script

This script demonstrates the key new features of the WiFi coverage analysis system:
1. Transport Mode Detection
2. Threshold Sensitivity Analysis
3. Enhanced Visualizations with Sample Density
4. Model Evaluation with Probability Calibration

Usage:
    python run_demo.py --demo [transport|threshold|visualization|complete]
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Import our modules
import transport_mode_detection
import threshold_sensitivity
import enhanced_visualization
import run_integrated_pipeline

def demo_transport_mode(merged_df=None, gps_df=None, wifi_df=None):
    """Run a demonstration of transport mode detection"""
    print("\n" + "="*80)
    print("Transport Mode Detection Demo".center(80))
    print("="*80 + "\n")
    
    # Load data if not provided
    if merged_df is None:
        try:
            print("Loading merged data...")
            merged_df = pd.read_csv('merged_wifi_location.csv')
            print(f"Loaded {len(merged_df)} rows of merged data")
        except FileNotFoundError:
            if gps_df is None or wifi_df is None:
                print("Loading GPS and WiFi data...")
                try:
                    gps_df = pd.read_csv('cleaned_gps_data.csv')
                    wifi_df = pd.read_csv('cleaned_wifi_data.csv')
                except FileNotFoundError:
                    print("ERROR: Required data files not found. Run preprocessing first.")
                    return
    
    # Start timing
    start_time = time.time()
    
    # Run transport mode detection
    print("\nDetecting transport modes based on GPS and WiFi metadata...")
    if merged_df is not None:
        transport_data, detector = transport_mode_detection.detect_transport_modes(
            None, None, merged_df=merged_df
        )
    else:
        transport_data, detector = transport_mode_detection.detect_transport_modes(
            gps_df, wifi_df
        )
    
    # Save results
    transport_data.to_csv('transport_modes.csv', index=False)
    print(f"Saved transport mode data to transport_modes.csv ({len(transport_data)} rows)")
    
    # Print a summary of detected modes
    mode_counts = transport_data['predicted_mode'].value_counts()
    total = len(transport_data)
    
    print("\nTransport Mode Distribution:")
    print("-" * 40)
    for mode, count in mode_counts.items():
        print(f"{mode.capitalize()}: {count} ({count/total*100:.1f}%)")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    mode_counts.plot(kind='bar', color='skyblue')
    plt.title('Transport Mode Distribution')
    plt.xlabel('Transport Mode')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('transport_mode_distribution.png')
    print("Saved transport mode distribution plot to transport_mode_distribution.png")
    
    # End timing
    end_time = time.time()
    print(f"\nTransport mode detection completed in {end_time - start_time:.2f} seconds")
    
    return transport_data, detector

def demo_threshold_sensitivity(merged_df=None):
    """Run a demonstration of threshold sensitivity analysis"""
    print("\n" + "="*80)
    print("Threshold Sensitivity Analysis Demo".center(80))
    print("="*80 + "\n")
    
    # Load data if not provided
    if merged_df is None:
        try:
            print("Loading merged data...")
            merged_df = pd.read_csv('merged_wifi_location.csv')
            print(f"Loaded {len(merged_df)} rows of merged data")
        except FileNotFoundError:
            print("ERROR: Required data files not found. Run preprocessing first.")
            return
    
    # Start timing
    start_time = time.time()
    
    # Run threshold sensitivity analysis
    print("\nAnalyzing RSSI threshold sensitivity...")
    analyzer, recommendations = threshold_sensitivity.run_threshold_analysis(
        merged_df,
        threshold_range=(-85, -65, 1)
    )
    
    # Print recommendations
    print("\nRecommended RSSI Thresholds:")
    print("-" * 40)
    for key, recs in recommendations.items():
        print(f"\nAnalysis: {key}")
        for scenario, values in recs.items():
            print(f"  {scenario}:")
            for metric, value in values.items():
                print(f"    {metric}: {value}")
    
    # End timing
    end_time = time.time()
    print(f"\nThreshold sensitivity analysis completed in {end_time - start_time:.2f} seconds")
    print("\nSaved threshold analysis plots to plots/threshold_analysis/")
    
    return analyzer, recommendations

def demo_enhanced_visualization(grid_stats=None, merged_df=None):
    """Run a demonstration of enhanced visualizations"""
    print("\n" + "="*80)
    print("Enhanced Visualization Demo".center(80))
    print("="*80 + "\n")
    
    # Load data if not provided
    if grid_stats is None:
        try:
            print("Loading grid statistics...")
            grid_stats = pd.read_csv('grid_coverage_statistics.csv')
            print(f"Loaded {len(grid_stats)} rows of grid statistics")
        except FileNotFoundError:
            print("ERROR: Required data files not found. Run preprocessing first.")
            return
    
    # Load merged data with transport modes if available
    if merged_df is None:
        try:
            print("Loading merged data with transport modes...")
            merged_df = pd.read_csv('transport_modes.csv')
            print(f"Loaded {len(merged_df)} rows of merged data with transport modes")
        except FileNotFoundError:
            try:
                print("Loading basic merged data...")
                merged_df = pd.read_csv('merged_wifi_location.csv')
                print(f"Loaded {len(merged_df)} rows of merged data")
            except FileNotFoundError:
                print("WARNING: No merged data available. Some visualizations will be limited.")
                merged_df = None
    
    # Start timing
    start_time = time.time()
    
    # Create enhanced visualizations
    print("\nCreating enhanced visualizations...")
    visualizer = enhanced_visualization.enhance_visualizations(
        grid_stats,
        merged_df
    )
    
    # End timing
    end_time = time.time()
    print(f"\nEnhanced visualization creation completed in {end_time - start_time:.2f} seconds")
    print("\nSaved enhanced maps to plots/enhanced_maps/")
    
    print("\nCreated the following visualization types:")
    print("1. Coverage map with sample density overlay")
    print("2. Confidence-based coverage visualization")
    print("3. Transport mode filtered coverage maps")
    print("4. Static density-aware coverage plot")
    
    return visualizer

def run_complete_demo():
    """Run a complete demonstration of all features"""
    print("\n" + "="*80)
    print("Complete WiFi Coverage Analysis Demo".center(80))
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs('demo_output', exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    print("Running complete demo with all new features...")
    run_integrated_pipeline.run_integrated_pipeline(
        output_dir='demo_output',
        skip_eda=False  # Ensure we run the full pipeline
    )
    
    # End timing
    end_time = time.time()
    print(f"\nComplete demo completed in {end_time - start_time:.2f} seconds")
    print(f"All outputs saved to demo_output/")

def main():
    parser = argparse.ArgumentParser(description='WiFi Coverage Analysis Demo')
    parser.add_argument('--demo', type=str, default='complete',
                        choices=['transport', 'threshold', 'visualization', 'complete'],
                        help='Which demo to run (default: complete)')
    
    args = parser.parse_args()
    
    if args.demo == 'transport':
        demo_transport_mode()
    elif args.demo == 'threshold':
        demo_threshold_sensitivity()
    elif args.demo == 'visualization':
        demo_enhanced_visualization()
    elif args.demo == 'complete':
        run_complete_demo()
    else:
        print(f"Unknown demo type: {args.demo}")

if __name__ == "__main__":
    main() 