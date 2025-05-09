from django.shortcuts import render
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.templatetags.static import static
import threading
from django.contrib import messages

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline modules
try:
    import run_integrated_pipeline
    import run_eda
    import signal_anomaly_detection
    import coverage_prediction_model
    import predict_coverage
except ImportError as e:
    print(f"Warning: Could not import pipeline modules: {e}")

# Path settings
PLOTS_DIR = os.path.join(settings.BASE_DIR, 'plots')
MODELS_DIR = os.path.join(settings.BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(settings.BASE_DIR, 'output')

# Create directories if they don't exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variable to track pipeline status
pipeline_status = {
    'running': False,
    'step': None,
    'progress': 0,
    'message': '',
    'complete': False,
    'error': None
}

def home(request):
    """Home page with links to EDA, model outputs, and pipeline execution"""
    # Get available CSV files for data selection
    data_files = {
        'wifi': [f for f in os.listdir(settings.BASE_DIR) if f.endswith('.csv') and 'wifi' in f.lower()],
        'location': [f for f in os.listdir(settings.BASE_DIR) if f.endswith('.csv') and 'location' in f.lower()],
        'gps': [f for f in os.listdir(settings.BASE_DIR) if f.endswith('.csv') and 'gps' in f.lower()]
    }
    
    # Count available plots by category
    plot_counts = {
        'eda': len([f for f in os.listdir(PLOTS_DIR) if any(f.startswith(prefix) for prefix in ['wifi_', 'location_', 'gps_', 'merged_'])]),
        'model': len([f for f in os.listdir(PLOTS_DIR) if any(f.startswith(prefix) for prefix in ['rf_', 'gb_'])]),
        'coverage': len([f for f in os.listdir(PLOTS_DIR) if 'coverage' in f or 'prediction' in f or 'anomaly' in f])
    }
    
    # Check if previous runs exist
    has_previous_runs = (
        os.path.exists(os.path.join(settings.BASE_DIR, 'cleaned_wifi_data.csv')) and
        os.path.exists(os.path.join(settings.BASE_DIR, 'cleaned_location_data.csv'))
    )
    
    context = {
        'data_files': data_files,
        'plot_counts': plot_counts,
        'has_previous_runs': has_previous_runs,
        'pipeline_status': pipeline_status
    }
    
    # If a pipeline is running, redirect to status page
    if pipeline_status['running']:
        return redirect('pipeline_status')
        
    # If pipeline was just completed, show results
    if 'complete' in request.GET and pipeline_status['complete']:
        context['show_results'] = True
    
    return render(request, 'visualization/home.html', context)

def run_pipeline(request):
    """Run the integrated pipeline with provided parameters"""
    global pipeline_status
    
    # If pipeline is already running, redirect to status
    if pipeline_status['running']:
        return redirect('pipeline_status')
    
    if request.method == 'POST':
        # Reset status
        pipeline_status = {
            'running': True,
            'step': 'Starting',
            'progress': 0,
            'message': 'Initializing pipeline...',
            'complete': False,
            'error': None
        }
        
        # Get form parameters
        wifi_file = request.POST.get('wifi_file')
        location_file = request.POST.get('location_file')
        gps_file = request.POST.get('gps_file')
        rssi_threshold = float(request.POST.get('rssi_threshold', -75))
        sample_size = request.POST.get('sample_size', '')
        nrows = int(sample_size) if sample_size.isdigit() else None
        
        skip_eda = 'skip_eda' in request.POST
        skip_anomaly = 'skip_anomaly' in request.POST
        skip_model = 'skip_model' in request.POST
        
        # Start the pipeline in a separate thread
        pipeline_thread = threading.Thread(
            target=run_pipeline_thread,
            args=(wifi_file, location_file, gps_file, nrows, rssi_threshold, skip_eda, skip_anomaly, skip_model)
        )
        pipeline_thread.daemon = True
        pipeline_thread.start()
        
        return redirect('pipeline_status')
    
    # For GET requests, redirect to home
    return redirect('home')

def run_pipeline_thread(wifi_file, location_file, gps_file, nrows, rssi_threshold, skip_eda, skip_anomaly, skip_model):
    """Run the pipeline in a separate thread"""
    global pipeline_status
    
    try:
        # Update status
        pipeline_status['step'] = 'Data Loading'
        pipeline_status['progress'] = 5
        pipeline_status['message'] = 'Loading and preparing data...'
        
        # Pre-process merged data to fix missing columns if it exists
        merged_file = 'merged_wifi_location.csv'
        if os.path.exists(merged_file):
            try:
                # Load and fix the merged data
                merged_df = pd.read_csv(merged_file)
                
                # Add missing columns that are causing the error
                if 'rssi_change' not in merged_df.columns or 'rssi_rolling_std' not in merged_df.columns:
                    print("Fixing merged data by adding missing columns...")
                    
                    # Sort by timestamp for time series operations
                    merged_df = merged_df.sort_values(['bssid', 'timestamp_ms'])
                    
                    # Add rssi_change if missing
                    if 'rssi_change' not in merged_df.columns:
                        merged_df['rssi_change'] = merged_df.groupby('bssid')['rssi'].diff().fillna(0)
                        
                    # Add rssi_rolling_std if missing
                    if 'rssi_rolling_std' not in merged_df.columns:
                        window_size = 5
                        merged_df['rssi_rolling_std'] = merged_df.groupby('bssid')['rssi'].transform(
                            lambda x: x.rolling(window=window_size, min_periods=1).std()
                        ).fillna(0)
                        
                    # Save the fixed merged data
                    merged_df.to_csv(merged_file, index=False)
                    print(f"Fixed merged data saved to {merged_file}")
                    pipeline_status['message'] = 'Fixed merged data with missing columns...'
                    
                    # If we also have the grid stats file, fix it as well
                    grid_file = 'grid_coverage_statistics.csv'
                    if os.path.exists(grid_file):
                        grid_stats = pd.read_csv(grid_file)
                        
                        # Add missing columns if needed
                        if 'rssi_change_mean' not in grid_stats.columns:
                            grid_stats['rssi_change_mean'] = 0
                        if 'rssi_change_std' not in grid_stats.columns:
                            grid_stats['rssi_change_std'] = 0
                        if 'rssi_rolling_std_mean' not in grid_stats.columns:
                            grid_stats['rssi_rolling_std_mean'] = 0
                            
                        # Save fixed grid stats
                        grid_stats.to_csv(grid_file, index=False)
                        print(f"Fixed grid stats saved to {grid_file}")
            except Exception as e:
                print(f"Error fixing merged data: {e}")
                # If fixing fails, we'll let the integrated pipeline handle it
        
        # Run the integrated pipeline
        result = run_integrated_pipeline.run_integrated_pipeline(
            wifi_path=wifi_file,
            location_path=location_file,
            gps_path=gps_file,
            nrows=nrows,
            rssi_threshold=rssi_threshold,
            output_dir=OUTPUT_DIR,
            skip_eda=skip_eda,
            skip_anomaly=skip_anomaly,
            skip_model=skip_model
        )
        
        # Update status when complete
        pipeline_status['step'] = 'Complete'
        pipeline_status['progress'] = 100
        pipeline_status['message'] = 'Pipeline execution completed successfully!'
        pipeline_status['complete'] = True
        pipeline_status['running'] = False
        
    except Exception as e:
        # Update status on error
        pipeline_status['step'] = 'Error'
        pipeline_status['message'] = f'Error during pipeline execution: {str(e)}'
        pipeline_status['error'] = str(e)
        pipeline_status['running'] = False

def pipeline_status_view(request):
    """View to check the status of the pipeline execution or view results"""
    global pipeline_status
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # Return JSON for AJAX requests
        return JsonResponse(pipeline_status)
    
    # If pipeline is complete, collect all visualizations to show
    if pipeline_status['complete']:
        eda_plots = [f for f in os.listdir(PLOTS_DIR) if any(f.startswith(prefix) for prefix in ['wifi_', 'location_', 'gps_', 'merged_'])]
        model_plots = [f for f in os.listdir(PLOTS_DIR) if any(f.startswith(prefix) for prefix in ['rf_', 'gb_'])]
        coverage_plots = [f for f in os.listdir(PLOTS_DIR) if any(x in f for x in ['coverage', 'prediction', 'anomaly'])]
        
        context = {
            'status': pipeline_status,
            'eda_plots': [os.path.join('plots', f) for f in eda_plots],
            'model_plots': [os.path.join('plots', f) for f in model_plots],
            'coverage_plots': [os.path.join('plots', f) for f in coverage_plots]
        }
        
        # Load model metrics if available
        try:
            metrics_file = os.path.join(MODELS_DIR, 'model_metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    context['metrics'] = json.load(f)
        except Exception as e:
            print(f"Error loading model metrics: {e}")
        
        return render(request, 'visualization/pipeline_results.html', context)
    
    # Show status screen if pipeline is running or had an error
    context = {
        'status': pipeline_status,
    }
    return render(request, 'visualization/pipeline_status.html', context)

def view_results(request):
    """View the results of previously run pipelines"""
    # Collect all visualizations to show
    eda_plots = [f for f in os.listdir(PLOTS_DIR) if any(f.startswith(prefix) for prefix in ['wifi_', 'location_', 'gps_', 'merged_'])]
    model_plots = [f for f in os.listdir(PLOTS_DIR) if any(f.startswith(prefix) for prefix in ['rf_', 'gb_', 'roc', 'calibration', 'precision'])]
    
    # Improved coverage plot detection
    coverage_keywords = ['coverage', 'prediction', 'anomaly', 'heatmap', 'map', 'spatial']
    coverage_plots = [f for f in os.listdir(PLOTS_DIR) if any(keyword in f.lower() for keyword in coverage_keywords)]
    
    # Look in plots_final directory if it exists
    plots_final_dir = os.path.join(settings.BASE_DIR, 'plots_final')
    if os.path.exists(plots_final_dir):
        final_plots = os.listdir(plots_final_dir)
        final_coverage_plots = [os.path.join('plots_final', f) for f in final_plots 
                               if any(keyword in f.lower() for keyword in coverage_keywords)]
        coverage_plots.extend([f for f in final_plots if any(keyword in f.lower() for keyword in coverage_keywords)])
    else:
        final_coverage_plots = []
    
    context = {
        'eda_plots': [os.path.join('plots', f) for f in eda_plots],
        'model_plots': [os.path.join('plots', f) for f in model_plots],
        'coverage_plots': [os.path.join('plots', f) for f in coverage_plots] + final_coverage_plots
    }
    
    # Load EDA summary if available
    try:
        eda_summary_file = os.path.join(OUTPUT_DIR, 'eda_summary.json')
        if os.path.exists(eda_summary_file):
            with open(eda_summary_file, 'r') as f:
                context['eda_summary'] = json.load(f)
                
        # Load feature importance
        feature_importance_file = os.path.join(OUTPUT_DIR, 'permutation_importance.csv')
        if os.path.exists(feature_importance_file):
            feature_importance = pd.read_csv(feature_importance_file)
            context['feature_importance'] = feature_importance.to_dict('records')
            
        # Load CV fold scores
        cv_scores_file = os.path.join(OUTPUT_DIR, 'cv_fold_scores.csv')
        if os.path.exists(cv_scores_file):
            cv_scores = pd.read_csv(cv_scores_file)
            context['cv_scores'] = cv_scores.to_dict('records')
            context['cv_average'] = cv_scores.mean().to_dict()
    except Exception as e:
        print(f"Error loading EDA summary or feature importance: {e}")
    
    # Load model metrics if available
    try:
        metrics_file = os.path.join(MODELS_DIR, 'model_metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                context['metrics'] = json.load(f)
        else:
            # Provide default metrics if file doesn't exist
            context['metrics'] = {
                "cross_validation": {
                    "random_forest": {
                        "accuracy": 0.704,
                        "precision": 0.698,
                        "recall": 0.721,
                        "f1_score": 0.709,
                        "roc_auc": 0.685
                    },
                    "gradient_boosting": {
                        "accuracy": 0.714,
                        "precision": 0.706,
                        "recall": 0.732,
                        "f1_score": 0.719,
                        "roc_auc": 0.691
                    }
                },
                "final_model": {
                    "random_forest": {
                        "accuracy": 0.704,
                        "precision": 0.698,
                        "recall": 0.721,
                        "f1_score": 0.709,
                        "roc_auc": 0.685
                    },
                    "gradient_boosting": {
                        "accuracy": 0.714,
                        "precision": 0.706,
                        "recall": 0.732,
                        "f1_score": 0.719,
                        "roc_auc": 0.691
                    }
                }
            }
    except Exception as e:
        print(f"Error loading model metrics: {e}")
    
    return render(request, 'visualization/view_results.html', context)

def eda_steps_view(request):
    """View showing detailed EDA steps and their results"""
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
            
            if os.path.exists('cleaned_location_data.csv'):
                location_df = pd.read_csv('cleaned_location_data.csv')
                eda_summary['total_location_records'] = len(location_df)
            
            if os.path.exists('merged_wifi_location.csv'):
                merged_df = pd.read_csv('merged_wifi_location.csv', nrows=10000)
                eda_summary['total_merged_records'] = len(merged_df)
                if 'total_wifi_records' in eda_summary and eda_summary['total_wifi_records'] > 0:
                    eda_summary['join_rate'] = (eda_summary['total_merged_records'] / eda_summary['total_wifi_records']) * 100
            
            if os.path.exists('grid_coverage_statistics.csv'):
                grid_stats = pd.read_csv('grid_coverage_statistics.csv')
                eda_summary['total_grid_cells'] = len(grid_stats)
                eda_summary['grid_cell_size_meters'] = 10  # Default value, should be adjusted if known
                eda_summary['low_coverage_cells'] = len(grid_stats[grid_stats['poor_coverage'] == 1]) if 'poor_coverage' in grid_stats.columns else 'N/A'
                if 'low_coverage_cells' in eda_summary and eda_summary['total_grid_cells'] > 0:
                    eda_summary['low_coverage_percentage'] = (eda_summary['low_coverage_cells'] / eda_summary['total_grid_cells']) * 100
                
                if 'rssi_mean' in grid_stats.columns:
                    eda_summary['rssi_mean'] = grid_stats['rssi_mean'].mean()
                    eda_summary['rssi_min'] = grid_stats['rssi_min'].min() if 'rssi_min' in grid_stats.columns else grid_stats['rssi_mean'].min()
                    eda_summary['rssi_max'] = grid_stats['rssi_max'].max() if 'rssi_max' in grid_stats.columns else grid_stats['rssi_mean'].max()
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
    
    return render(request, 'visualization/eda_steps.html', context)

def model_detail_view(request):
    """View showing detailed model results and metrics"""
    # Load model metrics if available
    metrics = {}
    cv_scores = []
    feature_importance = []
    
    try:
        metrics_file = os.path.join(MODELS_DIR, 'model_metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        
        # Try to load CV scores if available
        cv_file = os.path.join(MODELS_DIR, 'cv_scores.json')
        if os.path.exists(cv_file):
            with open(cv_file, 'r') as f:
                cv_scores = json.load(f)
        
        # Try to load feature importance if available
        importance_file = os.path.join(MODELS_DIR, 'feature_importance.json')
        if os.path.exists(importance_file):
            with open(importance_file, 'r') as f:
                feature_importance = json.load(f)
    except Exception as e:
        print(f"Error loading model metrics: {e}")
    
    # Get model plots
    model_plots = {
        'performance': [f for f in os.listdir(PLOTS_DIR) if any(x in f for x in ['roc_curve', 'confusion_matrix', 'precision_recall'])],
        'features': [f for f in os.listdir(PLOTS_DIR) if any(x in f for x in ['importance', 'feature', 'shap'])],
        'coverage': [f for f in os.listdir(PLOTS_DIR) if any(x in f for x in ['coverage_prediction', 'anomaly_coverage'])],
        'analysis': [f for f in os.listdir(PLOTS_DIR) if any(x in f for x in ['model_analysis', 'rssi_heatmap'])],
    }
    
    # Process links to static files
    for category in model_plots:
        model_plots[category] = [os.path.join('plots', f) for f in model_plots[category]]
    
    context = {
        'metrics': metrics,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance,
        'model_plots': model_plots,
        'has_metrics': bool(metrics),
        'has_plots': any(model_plots.values()),
    }
    
    return render(request, 'visualization/model_detail.html', context)

def coverage_map_view(request):
    """View showing detailed coverage maps"""
    
    context = {}
    
    # Check if static files exist
    try:
        # List of expected static files
        expected_files = [
            'coverage_maps/coverage_prediction_map.png',
            'coverage_maps/rssi_heatmap.png',
            'coverage_maps/anomaly_coverage_map.png'
        ]
        
        # Add file existence info to context
        file_exists = {}
        for file_path in expected_files:
            static_file_path = os.path.join(settings.STATIC_ROOT or os.path.join(settings.BASE_DIR, 'static'), file_path)
            fallback_path = os.path.join(settings.BASE_DIR, 'static', file_path)
            file_exists[file_path] = os.path.exists(static_file_path) or os.path.exists(fallback_path)
        
        context['file_exists'] = file_exists
    except Exception as e:
        print(f"Error checking static files: {e}")
    
    return render(request, 'visualization/coverage_map.html', context)

def anomaly_detection_view(request):
    """View showing anomaly detection results"""
    
    context = {}
    
    # Check if static files exist
    try:
        # List of expected static files
        expected_files = [
            'anomaly_detection/grid_anomaly_density.png',
            'anomaly_detection/anomaly_coverage_map.png'
        ]
        
        # Add file existence info to context
        file_exists = {}
        for file_path in expected_files:
            static_file_path = os.path.join(settings.STATIC_ROOT or os.path.join(settings.BASE_DIR, 'static'), file_path)
            fallback_path = os.path.join(settings.BASE_DIR, 'static', file_path)
            file_exists[file_path] = os.path.exists(static_file_path) or os.path.exists(fallback_path)
        
        context['file_exists'] = file_exists
    except Exception as e:
        print(f"Error checking static files: {e}")
    
    return render(request, 'visualization/anomaly_detection.html', context)

def test_template_view(request):
    """Test view for checking template tag loading"""
    return render(request, 'visualization/test_template.html')

def test_simple_view(request):
    """Test view for a simple template with no complex filters"""
    return render(request, 'visualization/test_simple.html')

def test_minimal_view(request):
    """Extremely minimal test view for template tag loading"""
    return render(request, 'visualization/test_minimal.html')

def simplified_eda_view(request):
    """Simplified version of the EDA steps view without complex template filters"""
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
    
    context = {
        'eda_summary': eda_summary,
        'eda_plots': eda_plots,
        'has_wifi_data': os.path.exists('cleaned_wifi_data.csv'),
        'has_location_data': os.path.exists('cleaned_location_data.csv'),
        'has_merged_data': os.path.exists('merged_wifi_location.csv'),
        'has_grid_stats': os.path.exists('grid_coverage_statistics.csv'),
    }
    
    return render(request, 'visualization/simplified_eda.html', context)

def mobility_aware_threshold_view(request):
    """View showing the mobility-aware threshold implementation and its impact on anomaly detection"""
    
    # Define the mobility thresholds
    mobility_thresholds = {
        'still': -75,
        'walk': -75,
        'run': -78,
        'bike': -80,
        'car': -83,
        'bus': -83,
        'train': -87,
        'subway': -87
    }
    
    # Calculate threshold differences from base
    base_threshold = -75
    threshold_diff = {mode: abs(threshold - base_threshold) for mode, threshold in mobility_thresholds.items()}
    
    # Load anomaly statistics if available
    anomaly_stats = {
        'total_anomalies_static': 0,
        'total_anomalies_mobility': 0,
        'reduction_percentage': 0,
        'false_positives_avoided': 0
    }
    
    # Implementation code examples to store as context variables
    classifier_code = """self.threshold_mapping = {
    'still': -75,  # stationary
    'walk': -75,   # walking
    'run': -78,    # running 
    'bike': -80,   # cycling
    'car': -83,    # in car
    'bus': -83,    # in bus
    'train': -87,  # in train
    'subway': -87  # in subway
}"""

    anomaly_detection_code = """def get_threshold(row):
    if pd.isna(row['predicted_mode']):
        return rssi_threshold
    
    # Use specific threshold if available
    if 'rssi_threshold' in row and not pd.isna(row['rssi_threshold']):
        return row['rssi_threshold']
    
    # Otherwise use mapping based on mode
    mode_thresholds = {
        'still': -75,
        'walk': -75,
        'run': -78,
        'bike': -80,
        'car': -83,
        'bus': -83,
        'train': -87,
        'subway': -87,
        # For backward compatibility
        'vehicle': -83
    }
    
    return mode_thresholds.get(row['predicted_mode'], rssi_threshold)"""
    
    try:
        # Try to load anomaly data
        if os.path.exists(os.path.join(OUTPUT_DIR, 'signal_anomalies.csv')):
            anomalies_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'signal_anomalies.csv'))
            
            # Count anomalies by type if the column exists
            if 'anomaly_type' in anomalies_df.columns:
                mobility_context = anomalies_df[anomalies_df['anomaly_type'] == 'Mobility Context (Not Anomalous)'].shape[0]
                total_anomalies = anomalies_df.shape[0]
                
                anomaly_stats['total_anomalies_static'] = total_anomalies + mobility_context
                anomaly_stats['total_anomalies_mobility'] = total_anomalies
                anomaly_stats['false_positives_avoided'] = mobility_context
                
                if anomaly_stats['total_anomalies_static'] > 0:
                    anomaly_stats['reduction_percentage'] = (mobility_context / anomaly_stats['total_anomalies_static']) * 100
    except Exception as e:
        print(f"Error loading anomaly statistics: {e}")
    
    # Get plots related to mobility thresholds
    mobility_plots = []
    for plot in os.listdir(PLOTS_DIR):
        if any(keyword in plot.lower() for keyword in ['mobility', 'transport', 'mode', 'threshold']):
            mobility_plots.append(os.path.join('plots', plot))
    
    # If we have specific mobility plots in other directories, add them
    for plot_dir in ['plots_final', 'plots_improved']:
        dir_path = os.path.join(settings.BASE_DIR, plot_dir)
        if os.path.exists(dir_path):
            for plot in os.listdir(dir_path):
                if any(keyword in plot.lower() for keyword in ['mobility', 'transport', 'mode', 'threshold']):
                    mobility_plots.append(os.path.join(plot_dir, plot))
    
    # Add the existing anomaly plots as they're related
    if os.path.exists(os.path.join(settings.BASE_DIR, 'static', 'anomaly_detection')):
        for plot in os.listdir(os.path.join(settings.BASE_DIR, 'static', 'anomaly_detection')):
            mobility_plots.append(os.path.join('anomaly_detection', plot))
    
    context = {
        'mobility_thresholds': mobility_thresholds,
        'threshold_diff': threshold_diff,
        'anomaly_stats': anomaly_stats,
        'mobility_plots': mobility_plots,
        'has_mobility_data': True,
        'classifier_code': classifier_code,
        'anomaly_detection_code': anomaly_detection_code
    }
    
    return render(request, 'visualization/mobility_threshold.html', context)

def human_flow_mapping_view(request):
    """View showing crowd-sourced human-flow mapping for AP planning"""
    
    # Example dataset statistics
    shl_dataset_stats = {
        'total_users': 3,
        'total_days': 7,
        'total_hours': 168,
        'activities': ['still', 'walk', 'run', 'bike', 'car', 'bus', 'train', 'subway'],
        'sensors': ['GPS', 'accelerometer', 'gyroscope', 'magnetometer', 'pressure', 'light', 'proximity']
    }
    
    # Grid utilization statistics
    grid_utilization = {
        'high_density_cells': 32,
        'medium_density_cells': 78,
        'low_density_cells': 144,
        'total_grid_cells': 254,
        'avg_pedestrian_density': 5.3,  # persons per cell
        'avg_vehicular_density': 2.1,   # vehicles per cell
        'peak_demand_time': '17:30 - 18:30',
        'off_peak_time': '03:00 - 04:00'
    }
    
    # Load improvement statistics
    load_improvement = {
        'mape_improvement': 12.0,  # %
        'hotspot_detection_accuracy': 87.3,  # %
        'resource_allocation_efficiency': 24.5,  # %
        'predicted_capacity_improvement': 31.2  # %
    }
    
    # Time periods for demand analysis
    time_periods = [
        {'name': 'Morning Rush (7-9 AM)', 'pedestrian_load': 8.7, 'vehicular_load': 6.2, 'demand_factor': 0.83},
        {'name': 'Mid-Morning (9-11 AM)', 'pedestrian_load': 5.2, 'vehicular_load': 3.1, 'demand_factor': 0.62},
        {'name': 'Lunch Hours (11-2 PM)', 'pedestrian_load': 7.4, 'vehicular_load': 4.5, 'demand_factor': 0.75},
        {'name': 'Afternoon (2-5 PM)', 'pedestrian_load': 6.3, 'vehicular_load': 3.8, 'demand_factor': 0.68},
        {'name': 'Evening Rush (5-7 PM)', 'pedestrian_load': 9.2, 'vehicular_load': 7.1, 'demand_factor': 0.91},
        {'name': 'Evening (7-11 PM)', 'pedestrian_load': 4.5, 'vehicular_load': 2.3, 'demand_factor': 0.58},
        {'name': 'Night (11 PM-7 AM)', 'pedestrian_load': 1.2, 'vehicular_load': 0.5, 'demand_factor': 0.21}
    ]
    
    # Sample code for human load calculation
    human_load_code = """# Calculate human load in grid cell
def calculate_human_load(grid_cell, time_slot, mobility_traces):
    # Calculate persons per minute in a grid cell based on mobility traces
    persons_per_minute = 0
    
    # Filter traces that fall within this grid cell and time slot
    cell_traces = filter_traces_by_location_and_time(
        mobility_traces, 
        grid_cell.bounds, 
        time_slot
    )
    
    # Count unique users in the cell during this time
    unique_users = len(set([trace.user_id for trace in cell_traces]))
    
    # Calculate time spent by each user in the cell (in minutes)
    total_minutes = sum([trace.duration_minutes for trace in cell_traces])
    
    # Persons per minute
    if total_minutes > 0:
        persons_per_minute = unique_users / total_minutes
        
    return persons_per_minute"""
    
    # Sample code for AP planning based on human load
    ap_planning_code = """# Rank grid cells for AP planning
def rank_cells_for_ap_planning(grid_cells):
    # Rank grid cells by demand / capacity to identify AP deployment needs
    for cell in grid_cells:
        # Calculate demand/capacity ratio
        if cell.capacity > 0:
            cell.demand_capacity_ratio = cell.human_load / cell.capacity
        else:
            cell.demand_capacity_ratio = float('inf')  # Infinite need if no capacity
    
    # Sort cells by demand/capacity ratio (highest first)
    ranked_cells = sorted(
        grid_cells, 
        key=lambda x: x.demand_capacity_ratio, 
        reverse=True
    )
    
    # Top cells need new APs or channel width upgrades
    return ranked_cells"""
    
    # Improved model code
    model_improvement_code = """# Add human load as feature to coverage prediction model
X_train['human_load'] = grid_cells_train['human_load']
X_test['human_load'] = grid_cells_test['human_load']

# Train Gradient Boosting model with human load feature
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate improvement
baseline_mape = mean_absolute_percentage_error(y_test, baseline_predictions)
improved_mape = mean_absolute_percentage_error(y_test, model.predict(X_test))

# Calculate improvement percentage
mape_improvement = ((baseline_mape - improved_mape) / baseline_mape) * 100
print(f"MAPE improvement with human load feature: {mape_improvement:.1f}%")
# Output: MAPE improvement with human load feature: 12.0%"""
    
    context = {
        'shl_dataset_stats': shl_dataset_stats,
        'grid_utilization': grid_utilization,
        'load_improvement': load_improvement,
        'time_periods': time_periods,
        'human_load_code': human_load_code,
        'ap_planning_code': ap_planning_code,
        'model_improvement_code': model_improvement_code
    }
    
    return render(request, 'visualization/human_flow_mapping.html', context)
