import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsRegressor
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

class UnifiedWifiCoverageModel:
    """
    A unified WiFi coverage prediction model that combines all processing steps:
    - Data loading and cleaning
    - Feature engineering
    - Anomaly detection
    - Model training
    - Visualization generation
    
    This avoids the need for a multi-step pipeline and provides direct access to results.
    """
    
    def __init__(self, 
                rssi_threshold=-75,
                output_dir='output',
                plots_dir='plots',
                models_dir='models'):
        """Initialize the unified model"""
        self.rssi_threshold = rssi_threshold
        self.output_dir = output_dir
        self.plots_dir = plots_dir
        self.models_dir = models_dir
        
        # Create necessary directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize model components
        self.scaler = StandardScaler()
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.knn_spatial = KNeighborsRegressor(n_neighbors=5)
        self.anomaly_model = IsolationForest(contamination=0.05, random_state=42)
        self.anomaly_scaler = StandardScaler()
        
        # Store metrics
        self.metrics = {}
        
    def load_and_clean_data(self, wifi_path, location_path, gps_path=None, nrows=None):
        """Load and clean the WiFi and location data"""
        print("Loading and cleaning data...")
        
        # Load data
        if nrows:
            wifi_df = pd.read_csv(wifi_path, nrows=nrows)
            location_df = pd.read_csv(location_path, nrows=nrows)
        else:
            wifi_df = pd.read_csv(wifi_path)
            location_df = pd.read_csv(location_path)
            
        # Load GPS data if provided
        if gps_path:
            if nrows:
                gps_df = pd.read_csv(gps_path, nrows=nrows)
            else:
                gps_df = pd.read_csv(gps_path)
        else:
            gps_df = None
            
        # Clean WiFi data
        # Handle hidden SSIDs - replace with "__hidden__" instead of a specific network name
        if 'ssid' in wifi_df.columns:
            wifi_df['ssid'] = wifi_df['ssid'].replace('', '__hidden__')
        
        # Convert timestamps to datetime properly
        if 'timestamp' in wifi_df.columns:
            wifi_df['timestamp'] = pd.to_datetime(wifi_df['timestamp'], unit='ms')
        
        # Convert RSSI to numeric
        if 'rssi' in wifi_df.columns:
            wifi_df['rssi'] = pd.to_numeric(wifi_df['rssi'], errors='coerce')
            
            # Apply Hampel filter to remove RSSI spikes
            def hampel_filter(x, k=5, threshold=3):
                # Hampel filter to remove outliers
                # k = window size, threshold = number of standard deviations
                rolling_median = x.rolling(window=k, center=True).median()
                rolling_std = x.rolling(window=k, center=True).std()
                diff = (x - rolling_median).abs()
                outlier_idx = diff > (threshold * rolling_std)
                return x.mask(outlier_idx, rolling_median)
            
            # Apply Hampel filter to RSSI values grouped by BSSID
            wifi_df['rssi'] = wifi_df.groupby('bssid')['rssi'].transform(hampel_filter)
        
        # Clean location data
        # Convert timestamps to datetime properly
        if 'timestamp' in location_df.columns:
            location_df['timestamp'] = pd.to_datetime(location_df['timestamp'], unit='ms')
        
        # Merge WiFi and location data with proper timestamp handling
        print("Merging WiFi and location data...")
        try:
            # Ensure both dataframes are sorted by timestamp
            wifi_df = wifi_df.sort_values('timestamp')
            location_df = location_df.sort_values('timestamp')
            
            # Use merge_asof with proper pd.Timedelta tolerance
            merged_df = pd.merge_asof(
                wifi_df,
                location_df,
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta(seconds=5)  # Proper tolerance as Timedelta
            )
            
            # Verify merge quality
            join_rate = merged_df['latitude'].notna().mean()
            print(f"Join rate: {join_rate:.2%} of records have location data")
            
            if join_rate < 0.90:
                print("WARNING: GPS join is sparse (<90% match rate)")
            
            assert join_rate > 0.50, "GPS join too sparse (<50% match rate)"
            
        except Exception as e:
            print(f"Error in merge_asof: {e}. Using regular merge...")
            # Fall back to regular merge but with proper timestamp comparison
            # First ensure we have timestamp_ms for joining
            if 'timestamp_ms' not in wifi_df.columns:
                wifi_df['timestamp_ms'] = wifi_df['timestamp'].astype(int) // 10**6
            
            if 'timestamp_ms' not in location_df.columns:
                location_df['timestamp_ms'] = location_df['timestamp'].astype(int) // 10**6
            
            # Merge on timestamp_ms
            merged_df = pd.merge(
                wifi_df,
                location_df,
                on='timestamp_ms',
                how='inner'
            )
            
            join_rate = len(merged_df) / len(wifi_df)
            print(f"Fallback join rate: {join_rate:.2%}")
        
        # Sort data for time-series operations
        merged_df = merged_df.sort_values(['bssid', 'timestamp'])
        
        # Add calculated columns
        print("Adding calculated columns...")
        
        # Calculate RSSI change between consecutive measurements
        merged_df['rssi_change'] = merged_df.groupby('bssid')['rssi'].diff().fillna(0)
        
        # Calculate rolling standard deviation of RSSI
        window_size = 5
        merged_df['rssi_rolling_std'] = merged_df.groupby('bssid')['rssi'].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).std()
        ).fillna(0)
        
        # Save cleaned data
        merged_df.to_csv(f"{self.output_dir}/merged_wifi_location.csv", index=False)
        
        # Create EDA summary statistics and save to a file
        print("Generating EDA summary statistics...")
        eda_summary = {
            'total_wifi_records': len(wifi_df),
            'total_location_records': len(location_df),
            'total_merged_records': len(merged_df),
            'unique_bssids': merged_df['bssid'].nunique(),
            'join_rate': join_rate,
            'rssi_mean': merged_df['rssi'].mean(),
            'rssi_std': merged_df['rssi'].std(),
            'rssi_min': merged_df['rssi'].min(),
            'rssi_max': merged_df['rssi'].max(),
            'hidden_ssids_count': wifi_df[wifi_df['ssid'] == '__hidden__'].shape[0] if 'ssid' in wifi_df.columns else 0,
            'hidden_ssids_percentage': (wifi_df[wifi_df['ssid'] == '__hidden__'].shape[0] / len(wifi_df)) if 'ssid' in wifi_df.columns else 0,
        }
        
        # Save EDA summary as JSON
        with open(f"{self.output_dir}/eda_summary.json", 'w') as f:
            json.dump(eda_summary, f, indent=4)
        
        # Also save as CSV for easy viewing
        pd.DataFrame([eda_summary]).to_csv(f"{self.output_dir}/eda_summary.csv", index=False)
        
        self.wifi_df = wifi_df
        self.location_df = location_df
        self.gps_df = gps_df
        self.merged_df = merged_df
        
        print(f"Data loaded and cleaned: {len(merged_df)} records")
        return merged_df
    
    def calculate_grid_statistics(self):
        """Calculate grid statistics from merged data"""
        print("Calculating grid statistics...")
        
        # Define grid cell size (in degrees)
        lat_grid_size = 0.0002  # Approximately 25 meters
        lon_grid_size = 0.0002  # Approximately 25 meters
        
        # Create grid cells
        self.merged_df['lat_grid'] = (self.merged_df['latitude'] // lat_grid_size) * lat_grid_size
        self.merged_df['lon_grid'] = (self.merged_df['longitude'] // lon_grid_size) * lon_grid_size
        
        # Create grid_id for use in group-based cross validation (combine lat and lon grid)
        self.merged_df['grid_id'] = self.merged_df['lat_grid'].astype(str) + '_' + self.merged_df['lon_grid'].astype(str)
        
        # Extract date for temporal validation
        if 'timestamp' in self.merged_df.columns:
            self.merged_df['date'] = self.merged_df['timestamp'].dt.date
        
        # Aggregate statistics by grid cell and BSSID
        grid_stats = self.merged_df.groupby(['lat_grid', 'lon_grid', 'bssid']).agg({
            'rssi': ['mean', 'std', 'min', 'max', 'count'],
            'rssi_change': ['mean', 'std'],
            'rssi_rolling_std': ['mean'],
            'grid_id': 'first'  # Keep grid_id for group CV
        }).reset_index()
        
        # Flatten column names
        grid_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in grid_stats.columns]
        
        # Define low coverage areas
        grid_stats['low_coverage_area'] = (grid_stats['rssi_mean'] < self.rssi_threshold).astype(int)
        
        # Add hourly variation if timestamp data available
        if 'timestamp' in self.merged_df.columns:
            # Calculate hourly statistics
            hourly_stats = self.merged_df.assign(
                hour = self.merged_df['timestamp'].dt.hour
            ).groupby(['lat_grid', 'lon_grid', 'bssid', 'hour']).agg({
                'rssi': ['mean', 'std']
            }).reset_index()
            
            # Rename columns
            hourly_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in hourly_stats.columns]
            
            # Calculate hourly variation per grid/bssid
            hourly_var = hourly_stats.groupby(['lat_grid', 'lon_grid', 'bssid']).agg({
                'rssi_mean': ['std', 'max', 'min']
            }).reset_index()
            
            # Rename columns
            hourly_var.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in hourly_var.columns]
            
            # Rename the aggregated columns more clearly
            hourly_var = hourly_var.rename(columns={
                'rssi_mean_std': 'hourly_variation',
                'rssi_mean_max': 'hourly_max',
                'rssi_mean_min': 'hourly_min'
            })
            
            # Merge hourly variation into grid_stats
            grid_stats = pd.merge(
                grid_stats,
                hourly_var[['lat_grid', 'lon_grid', 'bssid', 'hourly_variation', 'hourly_max', 'hourly_min']],
                on=['lat_grid', 'lon_grid', 'bssid'],
                how='left'
            )
            
            # Fill missing values
            grid_stats['hourly_variation'] = grid_stats['hourly_variation'].fillna(0)
            
        # Save grid statistics
        grid_stats.to_csv(f"{self.output_dir}/grid_coverage_statistics.csv", index=False)
        
        self.grid_stats = grid_stats
        print(f"Grid statistics calculated: {len(grid_stats)} grid cells")
        
        # Create a summary of coverage
        low_coverage_count = grid_stats['low_coverage_area'].sum()
        total_grids = len(grid_stats)
        low_coverage_pct = low_coverage_count/total_grids*100
        print(f"Low coverage areas: {low_coverage_count} out of {total_grids} grid cells ({low_coverage_pct:.2f}%)")
        
        # Add grid stats to EDA summary
        grid_summary = {
            'total_grid_cells': total_grids,
            'low_coverage_cells': low_coverage_count,
            'low_coverage_percentage': low_coverage_pct,
            'average_rssi_per_grid': grid_stats['rssi_mean'].mean(),
            'grid_cell_size_meters': 25,  # Approximate size in meters
        }
        
        # Update EDA summary with grid stats
        try:
            with open(f"{self.output_dir}/eda_summary.json", 'r') as f:
                eda_summary = json.load(f)
            
            eda_summary.update(grid_summary)
            
            with open(f"{self.output_dir}/eda_summary.json", 'w') as f:
                json.dump(eda_summary, f, indent=4)
                
            # Update CSV as well
            pd.DataFrame([eda_summary]).to_csv(f"{self.output_dir}/eda_summary.csv", index=False)
        except Exception as e:
            print(f"Warning: Could not update EDA summary with grid stats: {e}")
        
        return grid_stats
    
    def detect_anomalies(self):
        """Detect signal anomalies in the WiFi data"""
        print("Detecting signal anomalies...")
        
        # Prepare data for anomaly detection
        anomaly_features = ['rssi', 'rssi_change', 'rssi_rolling_std']
        X_anomaly = self.merged_df[anomaly_features].copy()
        
        # Scale the data
        X_anomaly_scaled = self.anomaly_scaler.fit_transform(X_anomaly)
        
        # Fit the model and predict
        self.merged_df['is_anomaly'] = self.anomaly_model.fit_predict(X_anomaly_scaled)
        
        # Convert predictions to binary (1 for anomalies, 0 for normal)
        self.merged_df['is_anomaly'] = (self.merged_df['is_anomaly'] == -1).astype(int)
        
        # Count anomalies
        anomaly_count = self.merged_df['is_anomaly'].sum()
        print(f"Detected {anomaly_count} anomalies out of {len(self.merged_df)} measurements ({anomaly_count/len(self.merged_df)*100:.2f}%)")
        
        # Aggregate anomaly information to grid level
        grid_anomaly_stats = self.merged_df.groupby(['lat_grid', 'lon_grid', 'bssid']).agg({
            'is_anomaly': ['mean', 'sum', 'count']
        }).reset_index()
        
        # Flatten column names
        grid_anomaly_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in grid_anomaly_stats.columns]
        
        # Merge anomaly information into grid_stats
        self.grid_stats = pd.merge(
            self.grid_stats,
            grid_anomaly_stats,
            on=['lat_grid', 'lon_grid', 'bssid'],
            how='left'
        )
        
        # Fill missing values for grids without anomaly information
        self.grid_stats = self.grid_stats.fillna({
            'is_anomaly_mean': 0,
            'is_anomaly_sum': 0
        })
        
        # Create an anomaly density feature (anomalies per measurement)
        self.grid_stats['anomaly_density'] = self.grid_stats['is_anomaly_sum'] / self.grid_stats['rssi_count']
        
        # Save models
        with open(f"{self.models_dir}/anomaly_detector.pkl", 'wb') as f:
            pickle.dump(self.anomaly_model, f)
            
        with open(f"{self.models_dir}/anomaly_scaler.pkl", 'wb') as f:
            pickle.dump(self.anomaly_scaler, f)
            
        return self.grid_stats
    
    def train_models(self):
        """Train prediction models for coverage problems"""
        print("Training coverage prediction models...")
        
        # Define feature columns
        feature_columns = [
            'rssi_mean', 'rssi_std', 'rssi_min', 'rssi_max', 'rssi_count',
            'rssi_change_mean', 'rssi_change_std', 'rssi_rolling_std_mean'
        ]
        
        # Add hourly variation if available
        if 'hourly_variation' in self.grid_stats.columns:
            feature_columns.append('hourly_variation')
        
        # Add anomaly density if available
        if 'anomaly_density' in self.grid_stats.columns:
            feature_columns.append('anomaly_density')
        
        # Prepare X and y
        X = self.grid_stats[feature_columns]
        y = self.grid_stats['low_coverage_area']
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        # Get grid_id for GroupKFold
        groups = self.grid_stats['grid_id']
        
        # Use GroupKFold to avoid data leakage
        print("Using GroupKFold cross-validation to prevent spatial data leakage")
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=5)
        fold_scores = []
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame(index=feature_columns)
        
        # Train with group-based cross-validation
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42+fold)
            rf.fit(X_train_scaled, y_train)
            rf_predictions = rf.predict(X_test_scaled)
            
            # Train Gradient Boosting
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42+fold)
            gb.fit(X_train_scaled, y_train)
            gb_predictions = gb.predict(X_test_scaled)
            
            # Calculate metrics
            rf_accuracy = accuracy_score(y_test, rf_predictions)
            gb_accuracy = accuracy_score(y_test, gb_predictions)
            
            # Store metrics for this fold
            fold_scores.append({
                'fold': fold + 1,
                'rf_accuracy': rf_accuracy,
                'rf_precision': precision_score(y_test, rf_predictions, zero_division=0),
                'rf_recall': recall_score(y_test, rf_predictions, zero_division=0),
                'rf_f1': f1_score(y_test, rf_predictions, zero_division=0),
                'gb_accuracy': gb_accuracy,
                'gb_precision': precision_score(y_test, gb_predictions, zero_division=0),
                'gb_recall': recall_score(y_test, gb_predictions, zero_division=0),
                'gb_f1': f1_score(y_test, gb_predictions, zero_division=0),
            })
            
            # Store feature importance
            feature_importance_df[f'rf_fold_{fold+1}'] = rf.feature_importances_
            feature_importance_df[f'gb_fold_{fold+1}'] = gb.feature_importances_
            
            print(f"Fold {fold+1}: RF Accuracy = {rf_accuracy:.4f}, GB Accuracy = {gb_accuracy:.4f}")
        
        # Calculate average metrics across folds
        fold_scores_df = pd.DataFrame(fold_scores)
        avg_metrics = fold_scores_df.mean().to_dict()
        print("\nAverage Cross-Validation Metrics:")
        print(f"RF Accuracy: {avg_metrics['rf_accuracy']:.4f}")
        print(f"RF Precision: {avg_metrics['rf_precision']:.4f}")
        print(f"RF Recall: {avg_metrics['rf_recall']:.4f}")
        print(f"RF F1 Score: {avg_metrics['rf_f1']:.4f}")
        print(f"GB Accuracy: {avg_metrics['gb_accuracy']:.4f}")
        print(f"GB Precision: {avg_metrics['gb_precision']:.4f}")
        print(f"GB Recall: {avg_metrics['gb_recall']:.4f}")
        print(f"GB F1 Score: {avg_metrics['gb_f1']:.4f}")
        
        # Save fold scores
        fold_scores_df.to_csv(f"{self.output_dir}/cv_fold_scores.csv", index=False)
        
        # Calculate average feature importance and add to the dataframe
        feature_importance_df['rf_importance_mean'] = feature_importance_df.filter(like='rf_fold').mean(axis=1)
        feature_importance_df['gb_importance_mean'] = feature_importance_df.filter(like='gb_fold').mean(axis=1)
        feature_importance_df['rf_importance_std'] = feature_importance_df.filter(like='rf_fold').std(axis=1)
        feature_importance_df['gb_importance_std'] = feature_importance_df.filter(like='gb_fold').std(axis=1)
        
        # Save feature importance
        feature_importance_df.to_csv(f"{self.output_dir}/feature_importance.csv")
        
        # Now train the final models on the entire dataset
        print("\nTraining final models on entire dataset...")
        
        # Train final RF model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest model
        self.rf_model.fit(X_scaled, y)
        rf_predictions = self.rf_model.predict(X_scaled)
        rf_probs = self.rf_model.predict_proba(X_scaled)[:, 1]
        
        # Train Gradient Boosting model
        self.gb_model.fit(X_scaled, y)
        gb_predictions = self.gb_model.predict(X_scaled)
        gb_probs = self.gb_model.predict_proba(X_scaled)[:, 1]
        
        # Train KNN Spatial model for interpolation
        X_spatial = self.grid_stats[['lat_grid', 'lon_grid']].values
        y_spatial = self.grid_stats['low_coverage_area'].values
        self.knn_spatial.fit(X_spatial, y_spatial)
        
        # Store final model metrics (will be similar to training metrics since no hold-out set)
        self.metrics = {
            'cross_validation': {
                'random_forest': {
                    'accuracy': float(avg_metrics['rf_accuracy']),
                    'precision': float(avg_metrics['rf_precision']),
                    'recall': float(avg_metrics['rf_recall']),
                    'f1_score': float(avg_metrics['rf_f1'])
                },
                'gradient_boosting': {
                    'accuracy': float(avg_metrics['gb_accuracy']),
                    'precision': float(avg_metrics['gb_precision']),
                    'recall': float(avg_metrics['gb_recall']),
                    'f1_score': float(avg_metrics['gb_f1'])
                }
            },
            'final_model': {
                'random_forest': {
                    'accuracy': float(accuracy_score(y, rf_predictions)),
                    'precision': float(precision_score(y, rf_predictions, zero_division=0)),
                    'recall': float(recall_score(y, rf_predictions, zero_division=0)),
                    'f1_score': float(f1_score(y, rf_predictions, zero_division=0))
                },
                'gradient_boosting': {
                    'accuracy': float(accuracy_score(y, gb_predictions)),
                    'precision': float(precision_score(y, gb_predictions, zero_division=0)),
                    'recall': float(recall_score(y, gb_predictions, zero_division=0)),
                    'f1_score': float(f1_score(y, gb_predictions, zero_division=0))
                }
            }
        }
        
        # Save models
        with open(f"{self.models_dir}/rf_coverage_model.pkl", 'wb') as f:
            pickle.dump(self.rf_model, f)
            
        with open(f"{self.models_dir}/gb_coverage_model.pkl", 'wb') as f:
            pickle.dump(self.gb_model, f)
            
        with open(f"{self.models_dir}/knn_spatial_model.pkl", 'wb') as f:
            pickle.dump(self.knn_spatial, f)
            
        with open(f"{self.models_dir}/feature_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
            
        # Save feature list
        with open(f"{self.models_dir}/feature_list.txt", 'w') as f:
            f.write('\n'.join(feature_columns))
            
        # Save metrics
        with open(f"{self.models_dir}/model_metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Calculate permutation feature importance for model interpretability
        from sklearn.inspection import permutation_importance
        
        # Calculate permutation importance for Random Forest
        rf_perm_importance = permutation_importance(
            self.rf_model, X_scaled, y, n_repeats=10, random_state=42
        )
        
        # Calculate permutation importance for Gradient Boosting
        gb_perm_importance = permutation_importance(
            self.gb_model, X_scaled, y, n_repeats=10, random_state=42
        )
        
        # Create DataFrame for permutation importance
        perm_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'RF_Importance': rf_perm_importance.importances_mean,
            'RF_Importance_Std': rf_perm_importance.importances_std,
            'GB_Importance': gb_perm_importance.importances_mean,
            'GB_Importance_Std': gb_perm_importance.importances_std
        })
        
        # Sort by importance
        perm_importance_df = perm_importance_df.sort_values('RF_Importance', ascending=False)
        
        # Save permutation importance
        perm_importance_df.to_csv(f"{self.output_dir}/permutation_importance.csv", index=False)
        
        # Create plots
        self._create_model_plots(X, y, feature_columns, rf_predictions, gb_predictions, rf_probs, gb_probs)
        
        return self.metrics
    
    def _create_model_plots(self, X, y, feature_columns, rf_predictions, gb_predictions, rf_probs, gb_probs):
        """Create model evaluation plots"""
        # Confusion matrix plots
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y, rf_predictions), annot=True, fmt='d', cmap='Blues')
        plt.title('Random Forest Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f"{self.plots_dir}/rf_confusion_matrix.png")
        plt.close()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y, gb_predictions), annot=True, fmt='d', cmap='Blues')
        plt.title('Gradient Boosting Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f"{self.plots_dir}/gb_confusion_matrix.png")
        plt.close()
        
        # Feature importance plot for Random Forest
        plt.figure(figsize=(10, 6))
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': self.rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/rf_feature_importance.png")
        plt.close()
        
        # ROC Curve
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=(8, 6))
        # Random Forest ROC
        fpr_rf, tpr_rf, _ = roc_curve(y, rf_probs)
        roc_auc_rf = auc(fpr_rf, tpr_rf)
        plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
        
        # Gradient Boosting ROC
        fpr_gb, tpr_gb, _ = roc_curve(y, gb_probs)
        roc_auc_gb = auc(fpr_gb, tpr_gb)
        plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = {roc_auc_gb:.3f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(f"{self.plots_dir}/roc_curve.png")
        plt.close()
        
        # Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        plt.figure(figsize=(8, 6))
        # Random Forest PR curve
        precision_rf, recall_rf, _ = precision_recall_curve(y, rf_probs)
        avg_precision_rf = average_precision_score(y, rf_probs)
        plt.plot(recall_rf, precision_rf, label=f'Random Forest (AP = {avg_precision_rf:.3f})')
        
        # Gradient Boosting PR curve
        precision_gb, recall_gb, _ = precision_recall_curve(y, gb_probs)
        avg_precision_gb = average_precision_score(y, gb_probs)
        plt.plot(recall_gb, precision_gb, label=f'Gradient Boosting (AP = {avg_precision_gb:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.savefig(f"{self.plots_dir}/precision_recall_curve.png")
        plt.close()
        
        # Calibration Curve
        from sklearn.calibration import calibration_curve
        
        plt.figure(figsize=(8, 6))
        # Random Forest calibration
        prob_true_rf, prob_pred_rf = calibration_curve(y, rf_probs, n_bins=10)
        plt.plot(prob_pred_rf, prob_true_rf, marker='o', label='Random Forest')
        
        # Gradient Boosting calibration
        prob_true_gb, prob_pred_gb = calibration_curve(y, gb_probs, n_bins=10)
        plt.plot(prob_pred_gb, prob_true_gb, marker='o', label='Gradient Boosting')
        
        # Plot diagonal (perfect calibration)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend(loc='lower right')
        plt.savefig(f"{self.plots_dir}/calibration_curve.png")
        plt.close()
    
    def generate_coverage_map(self):
        """Generate coverage map visualization"""
        print("Generating coverage map visualizations...")
        
        # Make predictions on grid data
        feature_columns = []
        with open(f"{self.models_dir}/feature_list.txt", 'r') as f:
            feature_columns = f.read().strip().split('\n')
            
        # Prepare features
        X = self.grid_stats[feature_columns].copy()
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions with both models
        self.grid_stats['rf_prediction'] = self.rf_model.predict(X_scaled)
        self.grid_stats['rf_prob'] = self.rf_model.predict_proba(X_scaled)[:, 1]
        
        self.grid_stats['gb_prediction'] = self.gb_model.predict(X_scaled)
        self.grid_stats['gb_prob'] = self.gb_model.predict_proba(X_scaled)[:, 1]
        
        # Create ensemble prediction (average of probabilities)
        self.grid_stats['ensemble_prob'] = (self.grid_stats['rf_prob'] + self.grid_stats['gb_prob']) / 2
        self.grid_stats['ensemble_prediction'] = (self.grid_stats['ensemble_prob'] > 0.5).astype(int)
        
        # Save prediction results
        self.save_prediction_results()
        
        # Use KNN spatial model for smooth interpolation
        # Extract coordinates from grid data
        min_lat, max_lat = self.grid_stats['lat_grid'].min(), self.grid_stats['lat_grid'].max()
        min_lon, max_lon = self.grid_stats['lon_grid'].min(), self.grid_stats['lon_grid'].max()
        
        lat_step = (max_lat - min_lat) / 100
        lon_step = (max_lon - min_lon) / 100
        
        lat_grid = np.arange(min_lat, max_lat, lat_step)
        lon_grid = np.arange(min_lon, max_lon, lon_step)
        
        lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
        prediction_points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
        
        # Use the KNN model for spatial interpolation
        grid_predictions = self.knn_spatial.predict(prediction_points)
        grid_predictions = grid_predictions.reshape(lon_mesh.shape)
        
        # Create the main coverage map
        plt.figure(figsize=(12, 10))
        
        # Plot interpolated coverage
        coverage_map = plt.contourf(lon_mesh, lat_mesh, grid_predictions, levels=50, cmap='RdYlGn_r', alpha=0.7)
        plt.colorbar(coverage_map, label='Low Coverage Probability')
        
        # Plot predicted grid cells
        low_coverage_points = self.grid_stats[self.grid_stats['ensemble_prediction'] == 1]
        plt.scatter(
            low_coverage_points['lon_grid'], 
            low_coverage_points['lat_grid'],
            c='red',
            marker='x',
            s=30,
            label='Predicted Low Coverage'
        )
        
        # Plot all measured points
        plt.scatter(
            self.merged_df['longitude'],
            self.merged_df['latitude'],
            c='blue',
            alpha=0.1,
            s=2
        )
        
        plt.title('Predicted Low Coverage Areas Map')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.plots_dir}/coverage_prediction_map.png")
        plt.close()
        
        # Create a dedicated anomaly map if we have anomaly data
        if 'anomaly_density' in self.grid_stats.columns:
            plt.figure(figsize=(12, 10))
            
            # Create a continuous colormap of anomaly density
            anomaly_map = plt.scatter(
                self.grid_stats['lon_grid'],
                self.grid_stats['lat_grid'],
                c=self.grid_stats['anomaly_density'],
                cmap='plasma',
                alpha=0.7,
                s=50,
                edgecolors='k'
            )
            plt.colorbar(anomaly_map, label='Anomaly Density')
            
            # Overlay low coverage contours
            contour = plt.contour(
                lon_mesh, 
                lat_mesh, 
                grid_predictions, 
                levels=[0.5], 
                colors='red', 
                linewidths=2
            )
            plt.clabel(contour, inline=True, fontsize=10, fmt='Low Coverage')
            
            plt.title('Signal Anomalies and Coverage Problems')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{self.plots_dir}/anomaly_coverage_map.png")
            plt.close()
    
    def save_prediction_results(self):
        """Save prediction results to CSV"""
        # Extract main columns for output
        result_columns = [
            'lat_grid', 'lon_grid', 'bssid', 
            'rssi_mean', 'rssi_std', 'rssi_count',
            'rf_prediction', 'rf_prob',
            'gb_prediction', 'gb_prob',
            'ensemble_prediction', 'ensemble_prob'
        ]
        
        # Add anomaly columns if available
        if 'anomaly_density' in self.grid_stats.columns:
            result_columns.append('anomaly_density')
        if 'is_anomaly_sum' in self.grid_stats.columns:
            result_columns.append('is_anomaly_sum')
        
        # Save results
        output_path = f"{self.output_dir}/coverage_predictions.csv"
        self.grid_stats[result_columns].to_csv(output_path, index=False)
        print(f"Saved prediction results to {output_path}")
    
    def run_full_analysis(self, wifi_path, location_path, gps_path=None, nrows=None):
        """Run the complete analysis pipeline in one step"""
        print("\n=== Starting Unified WiFi Coverage Analysis ===\n")
        
        # 1. Load and clean data
        self.load_and_clean_data(wifi_path, location_path, gps_path, nrows)
        
        # 2. Calculate grid statistics
        self.calculate_grid_statistics()
        
        # 3. Detect anomalies
        self.detect_anomalies()
        
        # 4. Train models
        self.train_models()
        
        # 5. Generate visualizations
        self.generate_coverage_map()
        
        print("\n=== Analysis Complete ===")
        print(f"Results saved to {self.output_dir}")
        print(f"Visualizations saved to {self.plots_dir}")
        print(f"Models saved to {self.models_dir}")
        
        return {
            'grid_stats': self.grid_stats,
            'metrics': self.metrics,
            'merged_df': self.merged_df
        }
    
    def load_from_existing_data(self):
        """Load from existing processed data files"""
        print("Loading from existing data files...")
        
        # Check if required files exist
        if not os.path.exists(f"{self.output_dir}/merged_wifi_location.csv"):
            print("Error: Merged data file not found. Please run full analysis first.")
            return False
            
        if not os.path.exists(f"{self.output_dir}/grid_coverage_statistics.csv"):
            print("Error: Grid statistics file not found. Please run full analysis first.")
            return False
            
        # Load merged data
        self.merged_df = pd.read_csv(f"{self.output_dir}/merged_wifi_location.csv")
        
        # Load grid statistics
        self.grid_stats = pd.read_csv(f"{self.output_dir}/grid_coverage_statistics.csv")
        
        # Load models if they exist
        try:
            with open(f"{self.models_dir}/rf_coverage_model.pkl", 'rb') as f:
                self.rf_model = pickle.load(f)
                
            with open(f"{self.models_dir}/gb_coverage_model.pkl", 'rb') as f:
                self.gb_model = pickle.load(f)
                
            with open(f"{self.models_dir}/knn_spatial_model.pkl", 'rb') as f:
                self.knn_spatial = pickle.load(f)
                
            with open(f"{self.models_dir}/feature_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
                
            # Load metrics if available
            if os.path.exists(f"{self.models_dir}/model_metrics.json"):
                with open(f"{self.models_dir}/model_metrics.json", 'r') as f:
                    self.metrics = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load some model files: {e}")
            
        print("Loaded existing data and models successfully")
        return True

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified WiFi coverage prediction model')
    parser.add_argument('--wifi', type=str, default='Hips_WiFi.csv', help='Path to WiFi data CSV')
    parser.add_argument('--location', type=str, default='Hips_Location.csv', help='Path to location data CSV')
    parser.add_argument('--gps', type=str, default=None, help='Path to GPS data CSV')
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to load (for testing)')
    parser.add_argument('--threshold', type=float, default=-75, help='RSSI threshold for low coverage')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--plots', type=str, default='plots', help='Plots directory')
    parser.add_argument('--models', type=str, default='models', help='Models directory')
    parser.add_argument('--load-existing', action='store_true', help='Load from existing data files')
    parser.add_argument('--skip-anomaly', action='store_true', help='Skip anomaly detection')
    
    args = parser.parse_args()
    
    # Initialize the model
    model = UnifiedWifiCoverageModel(
        rssi_threshold=args.threshold,
        output_dir=args.output,
        plots_dir=args.plots,
        models_dir=args.models
    )
    
    if args.load_existing:
        if model.load_from_existing_data():
            # Generate coverage map from existing data
            model.generate_coverage_map()
    else:
        # Run full analysis
        model.run_full_analysis(
            wifi_path=args.wifi,
            location_path=args.location,
            gps_path=args.gps,
            nrows=args.nrows
        ) 