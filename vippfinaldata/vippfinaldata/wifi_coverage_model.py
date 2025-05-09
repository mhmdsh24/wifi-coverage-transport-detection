import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsRegressor
import pickle
import os
import json

class CoveragePredictionModel:
    """
    WiFi Coverage Prediction Model
    
    This model predicts areas with low WiFi coverage based on signal statistics
    and includes proper spatial cross-validation to prevent data leakage.
    """
    
    def __init__(self, output_dir='output', models_dir='models', plots_dir='plots'):
        """
        Initialize the coverage prediction model
        
        Parameters:
        -----------
        output_dir : str
            Directory to save output files
        models_dir : str
            Directory to save trained models
        plots_dir : str
            Directory to save plots
        """
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.plots_dir = plots_dir
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Initialize model pipelines
        self.rf_pipeline = None
        self.gb_pipeline = None
        self.knn_spatial = None
        
        # Store metrics
        self.metrics = {}
        self.feature_columns = []
        
    def prepare_data(self, grid_stats):
        """
        Prepare data for model training
        
        Parameters:
        -----------
        grid_stats : DataFrame
            Grid statistics with feature columns
            
        Returns:
        --------
        X : DataFrame
            Features for model training
        y : Series
            Target variable (low_coverage_area)
        groups : Series
            Groups for cross-validation (grid_id)
        dates : Series
            Dates for temporal validation
        """
        print("Preparing data for model training...")
        
        # Define core feature columns, avoiding all direct RSSI values (which leak into target)
        # Since low_coverage_area is directly derived from rssi_mean < threshold,
        # and rssi_min/rssi_max are very highly correlated with rssi_mean
        self.feature_columns = [
            'rssi_std',
            'rssi_count',
            'rssi_change_mean', 
            'rssi_change_std', 
            'rssi_rolling_std_mean'
        ]
        
        # Add hourly variation if available
        if 'hourly_variation' in grid_stats.columns:
            self.feature_columns.append('hourly_variation')
            
        # Add anomaly density if available
        if 'anomaly_density' in grid_stats.columns:
            self.feature_columns.append('anomaly_density')
            if 'is_anomaly_mean' in grid_stats.columns:
                self.feature_columns.append('is_anomaly_mean')
        
        # Print warning about data leakage
        leakage_cols = ['rssi_mean', 'rssi_min', 'rssi_max']
        for col in leakage_cols:
            if col in self.feature_columns:
                print(f"WARNING: Removing '{col}' from features to prevent data leakage")
                self.feature_columns.remove(col)
            
        print("Using only derived features to prevent data leakage from RSSI → target")
        
        # Prepare X and y
        X = grid_stats[self.feature_columns].copy()
        y = grid_stats['low_coverage_area']
        
        # Add BSSID frequency information (how many different BSSIDs are in each grid)
        if 'bssid' in grid_stats.columns and 'lat_grid' in grid_stats.columns and 'lon_grid' in grid_stats.columns:
            print("Adding BSSID density features...")
            grid_bssid_counts = grid_stats.groupby(['lat_grid', 'lon_grid'])['bssid'].nunique().reset_index()
            grid_bssid_counts.rename(columns={'bssid': 'bssid_count'}, inplace=True)
            
            # Merge back to grid_stats
            grid_stats = pd.merge(
                grid_stats,
                grid_bssid_counts,
                on=['lat_grid', 'lon_grid'],
                how='left'
            )
            
            # Add to features
            if 'bssid_count' in grid_stats.columns:
                X['bssid_count'] = grid_stats['bssid_count']
                self.feature_columns.append('bssid_count')
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        # Ensure grid_id column exists for GroupKFold
        if 'grid_id' not in grid_stats.columns:
            print("Creating grid_id column for spatial cross-validation...")
            grid_stats['grid_id'] = grid_stats['lat_grid'].astype(str) + '_' + grid_stats['lon_grid'].astype(str)
        
        # Get grid_id for GroupKFold to prevent spatial data leakage
        groups = grid_stats['grid_id']
        
        # Check for date column (could be date or date_first)
        date_col = None
        if 'date' in grid_stats.columns:
            date_col = 'date'
        elif 'date_first' in grid_stats.columns:
            date_col = 'date_first'
            
        # Get dates for temporal validation if available
        dates = grid_stats[date_col] if date_col else None
        
        # Print column info for debugging
        print(f"Feature columns: {self.feature_columns}")
        print(f"Total features: {len(self.feature_columns)}")
        print(f"X shape: {X.shape}, y shape: {y.shape}, groups shape: {groups.shape}")
        
        return X, y, groups, dates
    
    def build_pipelines(self):
        """
        Build model pipelines with preprocessing and classifiers
        """
        print("Building model pipelines...")
        
        # Preprocessor with StandardScaler
        preprocessor = ColumnTransformer([
            ('scaler', StandardScaler(), self.feature_columns)
        ])
        
        # Random Forest pipeline
        self.rf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        # Gradient Boosting pipeline
        self.gb_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ))
        ])
        
        # KNN for spatial interpolation
        self.knn_spatial = KNeighborsRegressor(n_neighbors=5)
    
    def evaluate_with_spatial_cv(self, X, y, groups, n_splits=5):
        """
        Evaluate models with spatial cross-validation
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : Series
            Target
        groups : Series
            Spatial groups for GroupKFold
        n_splits : int
            Number of cross-validation splits
            
        Returns:
        --------
        cv_results : dict
            Cross-validation results
        """
        print(f"Evaluating models with {n_splits}-fold spatial cross-validation...")
        print("Using GroupKFold to prevent spatial data leakage")
        
        # Initialize GroupKFold for spatial cross-validation
        gkf = GroupKFold(n_splits=n_splits)
        
        # Initialize results
        fold_scores = []
        rf_probas = np.zeros_like(y, dtype=float)
        gb_probas = np.zeros_like(y, dtype=float)
        
        # Feature importance dataframe
        feature_imp_df = pd.DataFrame(index=self.feature_columns)
        
        # Cross-validation loop
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            print(f"\nFold {fold+1}/{n_splits}")
            print(f"Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
            
            # Ensure no grid_id appears in both train and test
            train_grids = set(groups.iloc[train_idx])
            test_grids = set(groups.iloc[test_idx])
            overlap = train_grids.intersection(test_grids)
            if overlap:
                print(f"WARNING: {len(overlap)} grid cells appear in both train and test!")
                
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train Random Forest
            self.rf_pipeline.fit(X_train, y_train)
            rf_predictions = self.rf_pipeline.predict(X_test)
            rf_proba = self.rf_pipeline.predict_proba(X_test)[:, 1]
            
            # Store probabilities for ROC calculation
            rf_probas[test_idx] = rf_proba
            
            # Train Gradient Boosting
            self.gb_pipeline.fit(X_train, y_train)
            gb_predictions = self.gb_pipeline.predict(X_test)
            gb_proba = self.gb_pipeline.predict_proba(X_test)[:, 1]
            
            # Store probabilities for ROC calculation
            gb_probas[test_idx] = gb_proba
            
            # Calculate metrics for this fold
            rf_accuracy = accuracy_score(y_test, rf_predictions)
            rf_precision = precision_score(y_test, rf_predictions, zero_division=0)
            rf_recall = recall_score(y_test, rf_predictions, zero_division=0)
            rf_f1 = f1_score(y_test, rf_predictions, zero_division=0)
            rf_roc_auc = roc_auc_score(y_test, rf_proba)
            
            gb_accuracy = accuracy_score(y_test, gb_predictions)
            gb_precision = precision_score(y_test, gb_predictions, zero_division=0)
            gb_recall = recall_score(y_test, gb_predictions, zero_division=0)
            gb_f1 = f1_score(y_test, gb_predictions, zero_division=0)
            gb_roc_auc = roc_auc_score(y_test, gb_proba)
            
            # Extract Random Forest feature importance
            rf_feature_imp = self.rf_pipeline.named_steps['classifier'].feature_importances_
            feature_imp_df[f'rf_fold_{fold+1}'] = rf_feature_imp
            
            # Store fold metrics
            fold_scores.append({
                'fold': fold + 1,
                'rf_accuracy': rf_accuracy,
                'rf_precision': rf_precision,
                'rf_recall': rf_recall,
                'rf_f1': rf_f1,
                'rf_roc_auc': rf_roc_auc,
                'gb_accuracy': gb_accuracy,
                'gb_precision': gb_precision,
                'gb_recall': gb_recall,
                'gb_f1': gb_f1,
                'gb_roc_auc': gb_roc_auc
            })
            
            print(f"RF - Accuracy: {rf_accuracy:.4f}, ROC-AUC: {rf_roc_auc:.4f}, F1: {rf_f1:.4f}")
            print(f"GB - Accuracy: {gb_accuracy:.4f}, ROC-AUC: {gb_roc_auc:.4f}, F1: {gb_f1:.4f}")
        
        # Calculate average feature importance
        feature_imp_df['rf_importance_mean'] = feature_imp_df.filter(like='rf_fold').mean(axis=1)
        feature_imp_df['rf_importance_std'] = feature_imp_df.filter(like='rf_fold').std(axis=1)
        
        # Save fold scores
        fold_scores_df = pd.DataFrame(fold_scores)
        fold_scores_df.to_csv(f"{self.output_dir}/cv_fold_scores.csv", index=False)
        
        # Save feature importance
        feature_imp_df.to_csv(f"{self.output_dir}/feature_importance.csv")
        
        # Calculate mean scores
        mean_scores = fold_scores_df.mean().to_dict()
        
        # Print average scores
        print("\nAverage Cross-validation Metrics:")
        print(f"RF - Accuracy: {mean_scores['rf_accuracy']:.4f}, ROC-AUC: {mean_scores['rf_roc_auc']:.4f}")
        print(f"GB - Accuracy: {mean_scores['gb_accuracy']:.4f}, ROC-AUC: {mean_scores['gb_roc_auc']:.4f}")
        
        return {
            'fold_scores': fold_scores_df,
            'feature_importance': feature_imp_df,
            'mean_scores': mean_scores,
            'rf_probas': rf_probas,
            'gb_probas': gb_probas
        }
    
    def train_final_models(self, X, y, X_spatial=None, y_spatial=None):
        """
        Train final models on the entire dataset
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : Series
            Target
        X_spatial : ndarray, optional
            Spatial coordinates for KNN
        y_spatial : ndarray, optional
            Target for KNN spatial model
        """
        print("Training final models on entire dataset...")
        
        # Train Random Forest
        self.rf_pipeline.fit(X, y)
        rf_predictions = self.rf_pipeline.predict(X)
        rf_probs = self.rf_pipeline.predict_proba(X)[:, 1]
        
        # Train Gradient Boosting
        self.gb_pipeline.fit(X, y)
        gb_predictions = self.gb_pipeline.predict(X)
        gb_probs = self.gb_pipeline.predict_proba(X)[:, 1]
        
        # Train KNN spatial model if coordinates provided
        if X_spatial is not None and y_spatial is not None:
            self.knn_spatial.fit(X_spatial, y_spatial)
        
        # Calculate final metrics
        rf_metrics = {
            'accuracy': float(accuracy_score(y, rf_predictions)),
            'precision': float(precision_score(y, rf_predictions, zero_division=0)),
            'recall': float(recall_score(y, rf_predictions, zero_division=0)),
            'f1_score': float(f1_score(y, rf_predictions, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, rf_probs))
        }
        
        gb_metrics = {
            'accuracy': float(accuracy_score(y, gb_predictions)),
            'precision': float(precision_score(y, gb_predictions, zero_division=0)),
            'recall': float(recall_score(y, gb_predictions, zero_division=0)),
            'f1_score': float(f1_score(y, gb_predictions, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, gb_probs))
        }
        
        # Save final metrics
        final_metrics = {
            'random_forest': rf_metrics,
            'gradient_boosting': gb_metrics
        }
        
        return final_metrics
    
    def calculate_permutation_importance(self, X, y):
        """
        Calculate permutation feature importance
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : Series
            Target
        
        Returns:
        --------
        perm_importance_df : DataFrame
            Permutation importance results
        """
        print("Calculating permutation feature importance...")
        
        # Calculate permutation importance for Random Forest
        rf_perm_importance = permutation_importance(
            self.rf_pipeline, X, y, n_repeats=10, random_state=42
        )
        
        # Calculate permutation importance for Gradient Boosting
        gb_perm_importance = permutation_importance(
            self.gb_pipeline, X, y, n_repeats=10, random_state=42
        )
        
        # Create DataFrame
        perm_importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'RF_Importance': rf_perm_importance.importances_mean,
            'RF_Importance_Std': rf_perm_importance.importances_std,
            'GB_Importance': gb_perm_importance.importances_mean,
            'GB_Importance_Std': gb_perm_importance.importances_std
        })
        
        # Sort by importance
        perm_importance_df = perm_importance_df.sort_values('RF_Importance', ascending=False)
        
        # Save to file
        perm_importance_df.to_csv(f"{self.output_dir}/permutation_importance.csv", index=False)
        
        return perm_importance_df
    
    def save_models(self):
        """Save trained models"""
        print("Saving trained models...")
        
        # Save Random Forest pipeline
        with open(f"{self.models_dir}/rf_pipeline.pkl", 'wb') as f:
            pickle.dump(self.rf_pipeline, f)
        
        # Save Gradient Boosting pipeline
        with open(f"{self.models_dir}/gb_pipeline.pkl", 'wb') as f:
            pickle.dump(self.gb_pipeline, f)
        
        # Save KNN spatial model
        if self.knn_spatial:
            with open(f"{self.models_dir}/knn_spatial.pkl", 'wb') as f:
                pickle.dump(self.knn_spatial, f)
        
        # Save feature list
        with open(f"{self.models_dir}/feature_list.txt", 'w') as f:
            f.write('\n'.join(self.feature_columns))
        
        # Save metrics
        with open(f"{self.models_dir}/model_metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def load_models(self):
        """Load trained models"""
        print("Loading trained models...")
        
        # Load Random Forest pipeline
        with open(f"{self.models_dir}/rf_pipeline.pkl", 'rb') as f:
            self.rf_pipeline = pickle.load(f)
        
        # Load Gradient Boosting pipeline
        with open(f"{self.models_dir}/gb_pipeline.pkl", 'rb') as f:
            self.gb_pipeline = pickle.load(f)
        
        # Load KNN spatial model
        try:
            with open(f"{self.models_dir}/knn_spatial.pkl", 'rb') as f:
                self.knn_spatial = pickle.load(f)
        except:
            print("KNN spatial model not found.")
        
        # Load feature list
        with open(f"{self.models_dir}/feature_list.txt", 'r') as f:
            self.feature_columns = f.read().strip().split('\n')
        
        # Load metrics
        with open(f"{self.models_dir}/model_metrics.json", 'r') as f:
            self.metrics = json.load(f)
    
    def create_plots(self, X, y, cv_results, preds=None):
        """
        Create evaluation plots
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : Series
            Target
        cv_results : dict
            Cross-validation results
        preds : dict, optional
            Predictions for grid data
        """
        print("Creating evaluation plots...")
        
        # 1. Confusion Matrix plots
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        rf_predictions = self.rf_pipeline.predict(X)
        sns.heatmap(confusion_matrix(y, rf_predictions), annot=True, fmt='d', cmap='Blues')
        plt.title('Random Forest Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.subplot(1, 2, 2)
        gb_predictions = self.gb_pipeline.predict(X)
        sns.heatmap(confusion_matrix(y, gb_predictions), annot=True, fmt='d', cmap='Blues')
        plt.title('Gradient Boosting Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/confusion_matrices.png")
        plt.close()
        
        # 2. Feature Importance
        plt.figure(figsize=(12, 8))
        feature_importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.rf_pipeline.named_steps['classifier'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/rf_feature_importance.png")
        plt.close()
        
        # 3. ROC Curve
        plt.figure(figsize=(10, 8))
        
        # Random Forest ROC
        fpr_rf, tpr_rf, _ = roc_curve(y, cv_results['rf_probas'])
        roc_auc_rf = auc(fpr_rf, tpr_rf)
        plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
        
        # Gradient Boosting ROC
        fpr_gb, tpr_gb, _ = roc_curve(y, cv_results['gb_probas'])
        roc_auc_gb = auc(fpr_gb, tpr_gb)
        plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = {roc_auc_gb:.3f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve - Cross-Validation')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.plots_dir}/roc_curve.png")
        plt.close()
        
        # 4. Precision-Recall Curve
        plt.figure(figsize=(10, 8))
        
        # Random Forest PR curve
        precision_rf, recall_rf, _ = precision_recall_curve(y, cv_results['rf_probas'])
        pr_auc_rf = auc(recall_rf, precision_rf)
        plt.plot(recall_rf, precision_rf, label=f'Random Forest (AUC = {pr_auc_rf:.3f})')
        
        # Gradient Boosting PR curve
        precision_gb, recall_gb, _ = precision_recall_curve(y, cv_results['gb_probas'])
        pr_auc_gb = auc(recall_gb, precision_gb)
        plt.plot(recall_gb, precision_gb, label=f'Gradient Boosting (AUC = {pr_auc_gb:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Cross-Validation')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.plots_dir}/precision_recall_curve.png")
        plt.close()
        
        # 5. Permutation Importance
        if os.path.exists(f"{self.output_dir}/permutation_importance.csv"):
            perm_imp = pd.read_csv(f"{self.output_dir}/permutation_importance.csv")
            plt.figure(figsize=(12, 8))
            
            # Plot Random Forest permutation importance
            sns.barplot(x='RF_Importance', y='Feature', data=perm_imp)
            plt.title('Random Forest Permutation Importance')
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/rf_permutation_importance.png")
            plt.close()
    
    def make_predictions(self, grid_stats):
        """
        Make predictions on grid data
        
        Parameters:
        -----------
        grid_stats : DataFrame
            Grid statistics with feature columns
            
        Returns:
        --------
        grid_stats : DataFrame
            Grid statistics with predictions added
        """
        print("Making predictions on grid data...")
        
        # Add BSSID frequency information if not already present
        if ('bssid_count' in self.feature_columns and 
            'bssid_count' not in grid_stats.columns and 
            'bssid' in grid_stats.columns and 
            'lat_grid' in grid_stats.columns and 
            'lon_grid' in grid_stats.columns):
            
            print("Adding BSSID density features for prediction...")
            grid_bssid_counts = grid_stats.groupby(['lat_grid', 'lon_grid'])['bssid'].nunique().reset_index()
            grid_bssid_counts.rename(columns={'bssid': 'bssid_count'}, inplace=True)
            
            # Merge back to grid_stats
            grid_stats = pd.merge(
                grid_stats,
                grid_bssid_counts,
                on=['lat_grid', 'lon_grid'],
                how='left'
            )
        
        # Check if all features are available
        missing_features = [f for f in self.feature_columns if f not in grid_stats.columns]
        if missing_features:
            print(f"WARNING: Missing features for prediction: {missing_features}")
            print("Creating missing features with default value 0")
            for feature in missing_features:
                grid_stats[feature] = 0
                
        # Prepare features
        X = grid_stats[self.feature_columns].copy()
        X = X.fillna(0)
        
        # Make predictions
        grid_stats['rf_prediction'] = self.rf_pipeline.predict(X)
        grid_stats['rf_prob'] = self.rf_pipeline.predict_proba(X)[:, 1]
        
        grid_stats['gb_prediction'] = self.gb_pipeline.predict(X)
        grid_stats['gb_prob'] = self.gb_pipeline.predict_proba(X)[:, 1]
        
        # Create ensemble prediction (average of probabilities)
        grid_stats['ensemble_prob'] = (grid_stats['rf_prob'] + grid_stats['gb_prob']) / 2
        grid_stats['ensemble_prediction'] = (grid_stats['ensemble_prob'] > 0.5).astype(int)
        
        # Save predictions
        grid_stats.to_csv(f"{self.output_dir}/coverage_predictions.csv", index=False)
        
        return grid_stats
    
    def generate_coverage_map(self, grid_stats):
        """
        Generate coverage map visualization
        
        Parameters:
        -----------
        grid_stats : DataFrame
            Grid statistics with predictions
        """
        print("Generating coverage map...")
        
        # Extract coordinates
        min_lat = grid_stats['lat_grid'].min()
        max_lat = grid_stats['lat_grid'].max()
        min_lon = grid_stats['lon_grid'].min()
        max_lon = grid_stats['lon_grid'].max()
        
        # Create grid for visualization
        lat_step = (max_lat - min_lat) / 100
        lon_step = (max_lon - min_lon) / 100
        
        lat_grid = np.arange(min_lat, max_lat, lat_step)
        lon_grid = np.arange(min_lon, max_lon, lon_step)
        
        lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
        prediction_points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
        
        # Use KNN spatial model for interpolation
        if self.knn_spatial is not None:
            # Prepare data for KNN
            X_spatial = grid_stats[['lat_grid', 'lon_grid']].values
            y_spatial = grid_stats['ensemble_prob'].values
            
            # Fit KNN if not already fitted
            if not hasattr(self.knn_spatial, 'n_features_in_'):
                self.knn_spatial.fit(X_spatial, y_spatial)
            
            # Make predictions
            grid_predictions = self.knn_spatial.predict(prediction_points)
            grid_predictions = grid_predictions.reshape(lon_mesh.shape)
            
            # Create coverage map
            plt.figure(figsize=(12, 10))
            
            # Plot interpolated coverage
            coverage_map = plt.contourf(
                lon_mesh, lat_mesh, grid_predictions, 
                levels=50, cmap='RdYlGn_r', alpha=0.7
            )
            plt.colorbar(coverage_map, label='Low Coverage Probability')
            
            # Plot predicted grid cells
            low_coverage_points = grid_stats[grid_stats['ensemble_prediction'] == 1]
            plt.scatter(
                low_coverage_points['lon_grid'],
                low_coverage_points['lat_grid'],
                c='red', marker='x', s=30,
                label='Predicted Low Coverage'
            )
            
            # Add details
            plt.title('Predicted Low Coverage Areas Map')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{self.plots_dir}/coverage_prediction_map.png")
            plt.close()
            
            # Create anomaly map if we have anomaly data
            if 'anomaly_density' in grid_stats.columns:
                plt.figure(figsize=(12, 10))
                
                # Create a continuous colormap of anomaly density
                anomaly_map = plt.scatter(
                    grid_stats['lon_grid'],
                    grid_stats['lat_grid'],
                    c=grid_stats['anomaly_density'],
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
        
        # Plot RSSI heatmap
        plt.figure(figsize=(12, 10))
        rssi_map = plt.scatter(
            grid_stats['lon_grid'],
            grid_stats['lat_grid'],
            c=grid_stats['rssi_mean'],
            cmap='viridis',
            alpha=0.7,
            s=50,
            edgecolors='k'
        )
        plt.colorbar(rssi_map, label='RSSI (dBm)')
        plt.title('Average RSSI by Grid Cell')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.plots_dir}/rssi_heatmap.png")
        plt.close()
    
    def run_full_pipeline(self, grid_stats):
        """
        Run the full model pipeline
        
        Parameters:
        -----------
        grid_stats : DataFrame
            Grid statistics with features
            
        Returns:
        --------
        grid_stats : DataFrame
            Grid statistics with predictions added
        """
        print("Running full coverage prediction pipeline...")
        
        # 1. Prepare data
        X, y, groups, dates = self.prepare_data(grid_stats)
        
        # 2. Build model pipelines
        self.build_pipelines()
        
        # 3. Evaluate with spatial cross-validation
        cv_results = self.evaluate_with_spatial_cv(X, y, groups)
        
        # 4. Train final models
        X_spatial = grid_stats[['lat_grid', 'lon_grid']].values
        y_spatial = y.values
        final_metrics = self.train_final_models(X, y, X_spatial, y_spatial)
        
        # 5. Calculate permutation importance
        perm_importance = self.calculate_permutation_importance(X, y)
        
        # 6. Store metrics
        self.metrics = {
            'cross_validation': cv_results['mean_scores'],
            'final_model': final_metrics
        }
        
        # 7. Save models
        self.save_models()
        
        # 8. Make predictions
        grid_stats = self.make_predictions(grid_stats)
        
        # 9. Create plots
        self.create_plots(X, y, cv_results, grid_stats)
        
        # 10. Generate coverage map
        self.generate_coverage_map(grid_stats)
        
        return grid_stats 