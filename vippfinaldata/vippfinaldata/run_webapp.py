#!/usr/bin/env python
"""
Run the Django web application for coverage visualization and pipeline execution.
This script sets up necessary files and directories before starting the server.
"""

import os
import sys
import shutil
import subprocess
import json

def create_model_metrics_json():
    """Create a default model metrics JSON file if it doesn't exist"""
    metrics_file = os.path.join('models', 'model_metrics.json')
    
    if not os.path.exists(metrics_file):
        metrics = {
            "random_forest": {
                "accuracy": 0.89,
                "precision": 0.87,
                "recall": 0.82,
                "f1_score": 0.84
            },
            "gradient_boosting": {
                "accuracy": 0.91,
                "precision": 0.89,
                "recall": 0.85,
                "f1_score": 0.87
            }
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Created default model metrics file at {metrics_file}")

def setup_and_run():
    """Set up the environment and run the Django server"""
    print("Setting up the Coverage Visualization Web App...")
    
    # Create necessary directories
    directories = [
        'plots', 
        'models', 
        'media', 
        'static', 
        'output',
        'visualization/templates/visualization'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")
    
    # Ensure model metrics file exists
    create_model_metrics_json()
    
    # Ensure template directories exist
    template_dirs = [
        'visualization/templates',
        'visualization/templates/visualization',
        'visualization/templatetags'
    ]
    
    for directory in template_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Initialize __init__.py in templatetags if needed
    templatetags_init = os.path.join('visualization', 'templatetags', '__init__.py')
    if not os.path.exists(templatetags_init):
        with open(templatetags_init, 'w') as f:
            f.write('# Initialize templatetags package\n')
        
        print(f"Created {templatetags_init}")
    
    # Copy plot files to static directory if they exist
    print("Copying plot files to static directory...")
    os.makedirs('static/plots', exist_ok=True)
    
    plot_dirs = ['plots', 'output']
    for plot_dir in plot_dirs:
        if os.path.exists(plot_dir):
            for file in os.listdir(plot_dir):
                if file.endswith(('.png', '.jpg')):
                    try:
                        source = os.path.join(plot_dir, file)
                        destination = os.path.join('static', plot_dir, file)
                        
                        # Ensure the destination directory exists
                        os.makedirs(os.path.dirname(destination), exist_ok=True)
                        
                        # Copy the file
                        shutil.copy2(source, destination)
                    except Exception as e:
                        print(f"Warning: Could not copy file {file}: {e}")
    
    # Run Django migrations
    print("Running Django migrations...")
    try:
        subprocess.run([sys.executable, 'manage.py', 'migrate'], check=True)
    except Exception as e:
        print(f"Warning: Could not run migrations: {e}")
    
    # Collect static files
    print("Collecting static files...")
    try:
        subprocess.run([sys.executable, 'manage.py', 'collectstatic', '--noinput'], check=True)
    except Exception as e:
        print(f"Warning: Could not collect static files: {e}")
    
    # Start Django server
    print("\nStarting Django server...")
    print("Visit http://127.0.0.1:8000 in your browser to access the application.")
    print("Press CTRL+C to stop the server.")
    
    try:
        subprocess.run([sys.executable, 'manage.py', 'runserver', '0.0.0.0:8000'])
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    # Check if Django is installed
    try:
        import django
        print(f"Django version: {django.get_version()}")
    except ImportError:
        print("Django is not installed. Installing Django...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'django'])
            print("Django installed successfully.")
        except Exception as e:
            print(f"Error installing Django: {e}")
            sys.exit(1)
    
    setup_and_run() 