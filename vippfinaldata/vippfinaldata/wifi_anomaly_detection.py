import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import os

class SignalAnomalyDetector:
    """
    Signal anomaly detection using Isolation Forest
    
    Designed to work within a scikit-learn pipeline to prevent data leakage
    """
    
    def __init__(self, contamination=0.05, random_state=42, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create pipeline with scaling and Isolation Forest
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('isolation_forest', IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=100,
                max_samples='auto'
            ))
        ])
        
    def fit(self, X, y=None):
        """
        Fit the anomaly detection model
        
        Parameters:
        -----------
        X : DataFrame
            Features for anomaly detection, typically:
            - rssi: Signal strength
            - rssi_change: Change between consecutive measurements
            - rssi_rolling_std: Rolling standard deviation
            
        Returns:
        --------
        self : object
            Returns self
        """
        print("Fitting anomaly detection model...")
        
        # Make sure we have the required features
        required_features = ['rssi', 'rssi_change', 'rssi_rolling_std']
        missing_features = [f for f in required_features if f not in X.columns]
        
        if missing_features:
            raise ValueError(f"Missing required features for anomaly detection: {missing_features}")
            
        # Select only the features we need
        X_anomaly = X[required_features].copy()
        
        # Fit the pipeline
        self.pipeline.fit(X_anomaly)
        
        return self
    
    def predict(self, X):
        """
        Predict anomalies
        
        Parameters:
        -----------
        X : DataFrame
            Features for anomaly detection
            
        Returns:
        --------
        is_anomaly : array
            1 for anomalies, 0 for normal
        """
        # Select features
        required_features = ['rssi', 'rssi_change', 'rssi_rolling_std']
        X_anomaly = X[required_features].copy()
        
        # Get predictions: -1 for anomalies, 1 for normal
        predictions = self.pipeline.predict(X_anomaly)
        
        # Convert to 1 for anomalies, 0 for normal
        is_anomaly = (predictions == -1).astype(int)
        
        anomaly_count = is_anomaly.sum()
        print(f"Detected {anomaly_count} anomalies out of {len(X)} measurements ({anomaly_count/len(X)*100:.2f}%)")
        
        return is_anomaly
    
    def fit_predict(self, X, y=None):
        """
        Fit the model and predict anomalies
        
        Parameters:
        -----------
        X : DataFrame
            Features for anomaly detection
            
        Returns:
        --------
        is_anomaly : array
            1 for anomalies, 0 for normal
        """
        self.fit(X)
        return self.predict(X)
    
    def save_model(self, filename='anomaly_detector.pkl'):
        """Save the anomaly detection model"""
        path = os.path.join(self.output_dir, filename)
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"Anomaly detection model saved to {path}")
    
    def load_model(self, filename='anomaly_detector.pkl'):
        """Load the anomaly detection model"""
        path = os.path.join(self.output_dir, filename)
        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)
        print(f"Anomaly detection model loaded from {path}")
        return self

def detect_anomalies(merged_df, contamination=0.05, output_dir='output'):
    """
    Detect signal anomalies in WiFi data
    
    Parameters:
    -----------
    merged_df : DataFrame
        Merged WiFi and location data with engineered features
    contamination : float
        Expected proportion of anomalies in the data
    output_dir : str
        Directory to save model
    
    Returns:
    --------
    merged_df : DataFrame
        Input DataFrame with 'is_anomaly' column added
    detector : SignalAnomalyDetector
        Fitted anomaly detector
    """
    print("Detecting signal anomalies...")
    
    # Create detector
    detector = SignalAnomalyDetector(
        contamination=contamination,
        output_dir=output_dir
    )
    
    # Fit and predict
    merged_df['is_anomaly'] = detector.fit_predict(merged_df)
    
    # Save model
    detector.save_model()
    
    return merged_df, detector

def add_anomaly_features_to_grid(grid_stats, merged_df):
    """
    Add anomaly statistics to grid statistics
    
    Parameters:
    -----------
    grid_stats : DataFrame
        Grid statistics DataFrame
    merged_df : DataFrame
        Merged data with anomaly predictions
    
    Returns:
    --------
    grid_stats : DataFrame
        Grid statistics with anomaly features
    """
    print("Adding anomaly features to grid statistics...")
    
    # Aggregate anomaly information to grid level
    grid_anomaly = merged_df.groupby(['lat_grid', 'lon_grid', 'bssid']).agg({
        'is_anomaly': ['mean', 'sum', 'count']
    }).reset_index()
    
    # Flatten column names
    grid_anomaly.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in grid_anomaly.columns]
    
    # Merge anomaly information into grid_stats
    grid_stats = pd.merge(
        grid_stats,
        grid_anomaly,
        on=['lat_grid', 'lon_grid', 'bssid'],
        how='left'
    )
    
    # Fill missing values
    grid_stats = grid_stats.fillna({
        'is_anomaly_mean': 0,
        'is_anomaly_sum': 0,
        'is_anomaly_count': 0
    })
    
    # Create anomaly density feature (anomalies per measurement)
    grid_stats['anomaly_density'] = grid_stats['is_anomaly_sum'] / grid_stats['rssi_count']
    
    return grid_stats 