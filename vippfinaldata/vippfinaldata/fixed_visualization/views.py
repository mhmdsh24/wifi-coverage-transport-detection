from django.shortcuts import render
import os
import json
import pandas as pd

from visualization.views import eda_steps_view as original_eda_steps_view

# Set up paths similar to the original app
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def fixed_eda_steps_view(request):
    """Fixed version of the EDA steps view"""
    # Reuse logic from the original view
    context = {}
    
    # Get EDA summary data
    eda_summary = {}
    try:
        # Try to read from EDA summary file if it exists
        if os.path.exists(os.path.join(OUTPUT_DIR, 'eda_summary.json')):
            with open(os.path.join(OUTPUT_DIR, 'eda_summary.json'), 'r') as f:
                eda_summary = json.load(f)
        else:
            # Generate summary from available data
            if os.path.exists('cleaned_wifi_data.csv'):
                wifi_df = pd.read_csv('cleaned_wifi_data.csv', nrows=10000)
                eda_summary['total_wifi_records'] = len(wifi_df)
                eda_summary['unique_bssids'] = len(wifi_df['bssid'].unique()) if 'bssid' in wifi_df.columns else 'N/A'
                eda_summary['hidden_ssids_count'] = wifi_df['ssid'].isna().sum() if 'ssid' in wifi_df.columns else 'N/A'
                if 'hidden_ssids_count' in eda_summary and eda_summary['total_wifi_records'] > 0:
                    eda_summary['hidden_ssids_percentage'] = (eda_summary['hidden_ssids_count'] / eda_summary['total_wifi_records']) * 100
    except Exception as e:
        print(f"Error generating EDA summary: {e}")

    # Get EDA step plots
    eda_plots = {
        'wifi': [f for f in os.listdir(PLOTS_DIR) if f.startswith('wifi_')],
        'location': [f for f in os.listdir(PLOTS_DIR) if f.startswith('location_')],
        'gps': [f for f in os.listdir(PLOTS_DIR) if f.startswith('gps_')],
        'merged': [f for f in os.listdir(PLOTS_DIR) if f.startswith('merged_')],
    }
    
    # Process links to static files
    for category in eda_plots:
        eda_plots[category] = [os.path.join('plots', f) for f in eda_plots[category]]
    
    # Read anomaly detection results if available
    anomalies = None
    if os.path.exists('signal_anomalies.csv'):
        try:
            anomalies = pd.read_csv('signal_anomalies.csv', nrows=10)
        except Exception as e:
            print(f"Error loading anomalies: {e}")
    
    context = {
        'eda_summary': eda_summary,
        'eda_plots': eda_plots,
        'anomalies': anomalies,
        'has_wifi_data': os.path.exists('cleaned_wifi_data.csv'),
        'has_location_data': os.path.exists('cleaned_location_data.csv'),
        'has_merged_data': os.path.exists('merged_wifi_location.csv'),
        'has_grid_stats': os.path.exists('grid_coverage_statistics.csv'),
    }
    
    return render(request, 'fixed_visualization/fixed_eda_steps.html', context)
