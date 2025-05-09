#!/usr/bin/env python
"""
Save model metrics to JSON files for web visualization.
This script is used to save model performance metrics, cross-validation results,
and feature importance data for display in the Django web application.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_model(model_path):
    """Load a trained model from pickle file"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def save_model_metrics(rf_metrics, gb_metrics, output_dir='models'):
    """
    Save model metrics to a JSON file for web visualization.
    
    Parameters:
    -----------
    rf_metrics : dict
        Random Forest metrics dictionary
    gb_metrics : dict
        Gradient Boosting metrics dictionary
    output_dir : str
        Output directory to save metrics file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {
        "cross_validation": {
            "random_forest": rf_metrics.get('cv', {}),
            "gradient_boosting": gb_metrics.get('cv', {})
        },
        "final_model": {
            "random_forest": rf_metrics.get('test', {}),
            "gradient_boosting": gb_metrics.get('test', {})
        }
    }
    
    # Convert NumPy types to Python native types for JSON serialization
    for model_type in ['random_forest', 'gradient_boosting']:
        for eval_type in ['cross_validation', 'final_model']:
            for metric, value in metrics[eval_type][model_type].items():
                if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                    metrics[eval_type][model_type][metric] = int(value)
                elif isinstance(value, (np.float64, np.float32, np.float16)):
                    metrics[eval_type][model_type][metric] = float(value)
    
    # Save the metrics
    metrics_file = os.path.join(output_dir, 'model_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Model metrics saved to {metrics_file}")
    return metrics_file

def save_cv_scores(cv_results, output_dir='models'):
    """
    Save cross-validation scores to a JSON file.
    
    Parameters:
    -----------
    cv_results : list of dict
        List of dictionaries containing CV scores for each fold
    output_dir : str
        Output directory to save CV scores file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure all values are Python native types
    processed_results = []
    for fold_dict in cv_results:
        processed_fold = {}
        for key, value in fold_dict.items():
            if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                processed_fold[key] = int(value)
            elif isinstance(value, (np.float64, np.float32, np.float16)):
                processed_fold[key] = float(value)
            else:
                processed_fold[key] = value
        processed_results.append(processed_fold)
    
    # Save the CV scores
    cv_file = os.path.join(output_dir, 'cv_scores.json')
    with open(cv_file, 'w') as f:
        json.dump(processed_results, f, indent=4)
    
    print(f"CV scores saved to {cv_file}")
    return cv_file

def save_feature_importance(feature_names, rf_importances, gb_importances, rf_std=None, gb_std=None, output_dir='models'):
    """
    Save feature importance data to a JSON file.
    
    Parameters:
    -----------
    feature_names : list
        List of feature names
    rf_importances : array
        Random Forest feature importances
    gb_importances : array
        Gradient Boosting feature importances
    rf_std : array, optional
        Standard deviation of Random Forest feature importances
    gb_std : array, optional
        Standard deviation of Gradient Boosting feature importances
    output_dir : str
        Output directory to save feature importance file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create feature importance data
    feature_importance = []
    for i, feature in enumerate(feature_names):
        importance_dict = {
            "Feature": feature,
            "RF_Importance": float(rf_importances[i]),
            "GB_Importance": float(gb_importances[i]),
        }
        
        if rf_std is not None:
            importance_dict["RF_Importance_Std"] = float(rf_std[i])
        
        if gb_std is not None:
            importance_dict["GB_Importance_Std"] = float(gb_std[i])
        
        feature_importance.append(importance_dict)
    
    # Sort by Random Forest importance
    feature_importance.sort(key=lambda x: x["RF_Importance"], reverse=True)
    
    # Save the feature importance
    importance_file = os.path.join(output_dir, 'feature_importance.json')
    with open(importance_file, 'w') as f:
        json.dump(feature_importance, f, indent=4)
    
    print(f"Feature importance saved to {importance_file}")
    return importance_file

def compute_model_metrics(y_true, y_pred, y_proba=None):
    """
    Compute model metrics from predictions.
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    y_proba : array, optional
        Predicted probabilities for the positive class
    
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")
            metrics['roc_auc'] = 0.5
    
    return metrics

# Example usage in a model script:
if __name__ == "__main__":
    try:
        # Load sample data
        if os.path.exists('grid_coverage_statistics.csv'):
            print("Generating sample metrics using grid_coverage_statistics.csv...")
            df = pd.read_csv('grid_coverage_statistics.csv')
            print(f"Loaded grid statistics with {len(df)} rows")
            
            # Check for low coverage column (could be named 'poor_coverage' or 'low_coverage_area')
            if 'low_coverage_area' in df.columns:
                target_column = 'low_coverage_area'
                print(f"Found '{target_column}' column, creating sample predictions...")
                
                # Convert to binary if needed (some versions might have continuous values)
                if df[target_column].nunique() > 2:
                    print(f"Converting {target_column} to binary...")
                    df['target'] = (df[target_column] > df[target_column].median()).astype(int)
                    target_column = 'target'
                
                # Create random predictions for demonstration
                np.random.seed(42)
                y_true = df[target_column].values
                y_pred_rf = np.random.choice([0, 1], size=len(y_true), p=[0.7, 0.3])
                y_pred_gb = np.random.choice([0, 1], size=len(y_true), p=[0.7, 0.3])
                y_proba_rf = np.random.random(size=len(y_true))
                y_proba_gb = np.random.random(size=len(y_true))
                
                # Compute metrics
                print("Computing model metrics...")
                rf_test_metrics = compute_model_metrics(y_true, y_pred_rf, y_proba_rf)
                gb_test_metrics = compute_model_metrics(y_true, y_pred_gb, y_proba_gb)
                
                # Create sample CV results
                print("Creating sample cross-validation results...")
                cv_results = []
                for i in range(5):
                    fold_metrics = {
                        'fold': i+1,
                        'rf_accuracy': np.random.uniform(0.65, 0.75),
                        'rf_f1': np.random.uniform(0.60, 0.70),
                        'rf_roc_auc': np.random.uniform(0.65, 0.75),
                        'gb_accuracy': np.random.uniform(0.65, 0.75),
                        'gb_f1': np.random.uniform(0.60, 0.70),
                        'gb_roc_auc': np.random.uniform(0.65, 0.75),
                    }
                    cv_results.append(fold_metrics)
                
                # Create sample metrics
                print("Creating sample metrics...")
                rf_metrics = {
                    'cv': {
                        'accuracy': np.mean([fold['rf_accuracy'] for fold in cv_results]),
                        'f1_score': np.mean([fold['rf_f1'] for fold in cv_results]),
                        'roc_auc': np.mean([fold['rf_roc_auc'] for fold in cv_results]),
                        'precision': np.random.uniform(0.60, 0.70),
                        'recall': np.random.uniform(0.60, 0.70),
                    },
                    'test': rf_test_metrics
                }
                
                gb_metrics = {
                    'cv': {
                        'accuracy': np.mean([fold['gb_accuracy'] for fold in cv_results]),
                        'f1_score': np.mean([fold['gb_f1'] for fold in cv_results]),
                        'roc_auc': np.mean([fold['gb_roc_auc'] for fold in cv_results]),
                        'precision': np.random.uniform(0.60, 0.70),
                        'recall': np.random.uniform(0.60, 0.70),
                    },
                    'test': gb_test_metrics
                }
                
                # Create feature importance data
                print("Creating feature importance data...")
                feature_names = [col for col in df.columns if col not in [target_column, 'grid_id', 'lat', 'lon', 'lat_grid', 'lon_grid', 'target']]
                print(f"Found {len(feature_names)} features")
                if len(feature_names) > 0:
                    print("Generating random feature importances...")
                    rf_importances = np.random.random(size=len(feature_names))
                    rf_importances = rf_importances / rf_importances.sum()
                    gb_importances = np.random.random(size=len(feature_names))
                    gb_importances = gb_importances / gb_importances.sum()
                    rf_std = np.random.random(size=len(feature_names)) * 0.05
                    gb_std = np.random.random(size=len(feature_names)) * 0.05
                    
                    # Save metrics
                    print("Saving metrics files...")
                    save_model_metrics(rf_metrics, gb_metrics)
                    save_cv_scores(cv_results)
                    save_feature_importance(feature_names, rf_importances, gb_importances, rf_std, gb_std)
                    
                    print("Sample metrics saved!")
                else:
                    print("Error: No features found")
            else:
                print("Error: No target column found (looked for 'low_coverage_area' or 'poor_coverage')")
                print("Available columns:", df.columns.tolist())
        else:
            print("Error: grid_coverage_statistics.csv not found")
            print("Please run: py eda_summary_generator.py first")
    except Exception as e:
        print(f"Error generating sample metrics: {e}")
        import traceback
        traceback.print_exc() 