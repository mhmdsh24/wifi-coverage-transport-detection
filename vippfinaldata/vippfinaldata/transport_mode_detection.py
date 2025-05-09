import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from math import radians, sin, cos, sqrt, atan2
import warnings

class TransportModeDetector:
    """
    Transport Mode Detection based on GPS and WiFi metadata
    
    Classifies transport modes into:
    - Still: Not moving or very slow movement
    - Walk: Walking pace movement
    - Bike: Cycling pace movement
    - Vehicle: In a car, bus, train, etc.
    
    Uses only GPS metadata (speed, heading changes) and WiFi scan properties
    without needing inertial sensors, as demonstrated in the SHL dataset challenges.
    
    References:
    [1] Sussex-Huawei Locomotion Dataset (https://www.shl-dataset.org/dataset/)
    [2] University of Sussex Multimodal Locomotion Analytics (https://www.sussex.ac.uk/strc/research/wearable/locomotion-transportation)
    [3] SHL Challenge 2021 - ACM Digital Library (https://dl.acm.org/doi/10.1145/3460418.3479373)
    [4] Application of machine learning to predict transport modes from GPS data (https://pmc.ncbi.nlm.nih.gov/articles/PMC9667683/)
    """
    
    def __init__(self, output_dir='output', models_dir='models', plots_dir='plots'):
        """
        Initialize the transport mode detector
        
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
        os.makedirs(f"{plots_dir}/transport_modes", exist_ok=True)
        
        # Initialize model pipeline
        self.pipeline = None
        
        # Mode thresholds (m/s) based on SHL dataset papers
        self.mode_thresholds = {
            'still': 0.5,  # Below 0.5 m/s is considered still
            'walk': 2.5,   # 0.5-2.5 m/s is walking pace (slight adjustment from 3.0)
            'bike': 6.0,   # 2.5-6.0 m/s is cycling pace
            'vehicle': float('inf')  # Above 6.0 m/s is vehicle
        }
        
        # Mapping from numeric code to string mode
        self.mode_mapping = {
            0: 'still',
            1: 'walk',
            2: 'bike',
            3: 'vehicle'
        }
        
        # Inverse mapping from string mode to numeric code
        self.inverse_mode_mapping = {v: k for k, v in self.mode_mapping.items()}
    
    def prepare_features(self, gps_df, wifi_df, merged_df=None):
        """
        Prepare features for transport mode detection
        
        Parameters:
        -----------
        gps_df : DataFrame
            GPS data with timestamps, coordinates, and speed
        wifi_df : DataFrame
            WiFi data with timestamps and RSSI
        merged_df : DataFrame, optional
            Already merged data with both GPS and WiFi information
            
        Returns:
        --------
        X : DataFrame
            Features for transport mode detection
        y : Series
            Target variable (transport mode codes)
        merged_data : DataFrame
            Complete merged data with features and target
        """
        print("Preparing features for transport mode detection...")
        
        if merged_df is None:
            # Check required GPS columns
            required_gps_cols = ['timestamp_ms', 'latitude_deg', 'longitude_deg', 'speed_mps']
            if not all(col in gps_df.columns for col in required_gps_cols):
                raise ValueError(f"GPS data missing required columns: {required_gps_cols}")
            
            # Check required WiFi columns
            required_wifi_cols = ['timestamp_ms', 'bssid', 'rssi']
            if not all(col in wifi_df.columns for col in required_wifi_cols):
                raise ValueError(f"WiFi data missing required columns: {required_wifi_cols}")
                
            # Prepare data - ensure timestamps are sorted
            gps_df = gps_df.sort_values('timestamp_ms')
            wifi_df = wifi_df.sort_values('timestamp_ms')
            
            # Convert timestamps to datetime for easier manipulation
            if 'timestamp_dt' not in gps_df.columns:
                gps_df['timestamp_dt'] = pd.to_datetime(gps_df['timestamp_ms'], unit='ms')
            
            if 'timestamp_dt' not in wifi_df.columns:
                wifi_df['timestamp_dt'] = pd.to_datetime(wifi_df['timestamp_ms'], unit='ms')
            
            # Create time windows for aggregation (5-second windows)
            window_size_ms = 5000
            gps_df['time_window'] = (gps_df['timestamp_ms'] // window_size_ms) * window_size_ms
            wifi_df['time_window'] = (wifi_df['timestamp_ms'] // window_size_ms) * window_size_ms
            
            # Aggregate GPS data by time window
            gps_agg = gps_df.groupby('time_window').agg({
                'timestamp_ms': 'first',
                'latitude_deg': 'mean',
                'longitude_deg': 'mean',
                'speed_mps': 'mean',
                'timestamp_dt': 'first'
            }).reset_index()
            
            # Calculate heading changes
            gps_agg['prev_lat'] = gps_agg['latitude_deg'].shift(1)
            gps_agg['prev_lon'] = gps_agg['longitude_deg'].shift(1)
            
            # Calculate bearing (heading)
            def calculate_bearing(lat1, lon1, lat2, lon2):
                lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                dlon = lon2 - lon1
                y = sin(dlon) * cos(lat2)
                x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
                bearing = np.arctan2(y, x)
                # Convert to degrees
                bearing = np.degrees(bearing)
                # Normalize to 0-360
                bearing = (bearing + 360) % 360
                return bearing
            
            gps_agg['bearing'] = gps_agg.apply(
                lambda row: calculate_bearing(
                    row['prev_lat'], row['prev_lon'], row['latitude_deg'], row['longitude_deg']
                ) if pd.notna(row['prev_lat']) else 0,
                axis=1
            )
            
            # Calculate bearing change (heading change)
            gps_agg['prev_bearing'] = gps_agg['bearing'].shift(1)
            gps_agg['bearing_change'] = (gps_agg['bearing'] - gps_agg['prev_bearing']).abs()
            # Adjust for circular nature of bearings
            gps_agg['bearing_change'] = gps_agg['bearing_change'].apply(
                lambda x: min(x, 360-x) if pd.notna(x) else 0
            )
            
            # Calculate acceleration (SHL feature)
            gps_agg['prev_speed'] = gps_agg['speed_mps'].shift(1)
            gps_agg['prev_timestamp'] = gps_agg['timestamp_ms'].shift(1)
            gps_agg['time_diff'] = (gps_agg['timestamp_ms'] - gps_agg['prev_timestamp']) / 1000  # in seconds
            gps_agg['acceleration'] = gps_agg.apply(
                lambda row: (row['speed_mps'] - row['prev_speed']) / row['time_diff'] 
                if pd.notna(row['prev_speed']) and row['time_diff'] > 0 else 0,
                axis=1
            )
            
            # Count WiFi scans by time window
            wifi_counts = wifi_df.groupby(['time_window', 'bssid']).size().reset_index(name='scan_count')
            wifi_agg = wifi_counts.groupby('time_window').agg({
                'scan_count': 'sum',
                'bssid': 'nunique'
            }).reset_index()
            wifi_agg.rename(columns={'bssid': 'unique_bssids'}, inplace=True)
            
            # Calculate WiFi scanning rate (SHL feature)
            if 'timestamp_ms' in wifi_df.columns:
                # Group by time window and calculate time differences between scans
                wifi_df = wifi_df.sort_values(['time_window', 'timestamp_ms'])
                wifi_df['prev_timestamp'] = wifi_df.groupby('time_window')['timestamp_ms'].shift(1)
                wifi_df['scan_interval'] = (wifi_df['timestamp_ms'] - wifi_df['prev_timestamp']) / 1000  # in seconds
                
                # Aggregate scan intervals by time window
                scan_intervals = wifi_df.groupby('time_window').agg({
                    'scan_interval': ['mean', 'std', 'max', 'min']
                }).reset_index()
                
                # Flatten the column names
                scan_intervals.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in scan_intervals.columns]
                
                # Merge with wifi_agg
                wifi_agg = pd.merge(
                    wifi_agg,
                    scan_intervals,
                    on='time_window',
                    how='left'
                )
            
            # Merge GPS and WiFi data
            merged_data = pd.merge(
                gps_agg,
                wifi_agg,
                on='time_window',
                how='left'
            )
            
            # Fill missing WiFi data
            merged_data['scan_count'] = merged_data['scan_count'].fillna(0)
            merged_data['unique_bssids'] = merged_data['unique_bssids'].fillna(0)
            if 'scan_interval_mean' in merged_data.columns:
                merged_data['scan_interval_mean'] = merged_data['scan_interval_mean'].fillna(0)
                merged_data['scan_interval_std'] = merged_data['scan_interval_std'].fillna(0)
            
        else:
            # Use the provided merged DataFrame
            merged_data = merged_df.copy()
            
            # Ensure required columns exist
            if 'speed_mps' not in merged_data.columns and 'speed' in merged_data.columns:
                merged_data['speed_mps'] = merged_data['speed']
            
            if 'speed_mps' not in merged_data.columns:
                raise ValueError("Merged data missing 'speed_mps' column")
            
            if 'timestamp_ms' not in merged_data.columns:
                raise ValueError("Merged data missing 'timestamp_ms' column")
                
            # Create time windows for aggregation if needed
            if 'time_window' not in merged_data.columns:
                window_size_ms = 5000
                merged_data['time_window'] = (merged_data['timestamp_ms'] // window_size_ms) * window_size_ms
            
            # Calculate scan density if not already present
            if 'scan_count' not in merged_data.columns and 'bssid' in merged_data.columns:
                scan_counts = merged_data.groupby(['time_window', 'bssid']).size().reset_index(name='scan_count')
                scan_agg = scan_counts.groupby('time_window').agg({
                    'scan_count': 'sum',
                    'bssid': 'nunique'
                }).reset_index()
                scan_agg.rename(columns={'bssid': 'unique_bssids'}, inplace=True)
                
                # Merge back to main data
                merged_data = pd.merge(
                    merged_data.drop(columns=['scan_count', 'unique_bssids'], errors='ignore'),
                    scan_agg,
                    on='time_window',
                    how='left'
                )
                
            # Calculate bearing changes if not already present
            if 'bearing_change' not in merged_data.columns and 'latitude_deg' in merged_data.columns and 'longitude_deg' in merged_data.columns:
                # Sort by timestamp
                merged_data = merged_data.sort_values('timestamp_ms')
                
                # Calculate previous positions
                merged_data['prev_lat'] = merged_data['latitude_deg'].shift(1)
                merged_data['prev_lon'] = merged_data['longitude_deg'].shift(1)
                
                # Calculate bearing
                def calculate_bearing(lat1, lon1, lat2, lon2):
                    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                    dlon = lon2 - lon1
                    y = sin(dlon) * cos(lat2)
                    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
                    bearing = np.arctan2(y, x)
                    # Convert to degrees
                    bearing = np.degrees(bearing)
                    # Normalize to 0-360
                    bearing = (bearing + 360) % 360
                    return bearing
                
                merged_data['bearing'] = merged_data.apply(
                    lambda row: calculate_bearing(
                        row['prev_lat'], row['prev_lon'], row['latitude_deg'], row['longitude_deg']
                    ) if pd.notna(row['prev_lat']) else 0,
                    axis=1
                )
                
                # Calculate bearing change
                merged_data['prev_bearing'] = merged_data['bearing'].shift(1)
                merged_data['bearing_change'] = (merged_data['bearing'] - merged_data['prev_bearing']).abs()
                # Adjust for circular nature of bearings
                merged_data['bearing_change'] = merged_data['bearing_change'].apply(
                    lambda x: min(x, 360-x) if pd.notna(x) else 0
                )
                
            # Calculate acceleration if not present
            if 'acceleration' not in merged_data.columns and 'speed_mps' in merged_data.columns:
                # Sort by timestamp
                merged_data = merged_data.sort_values('timestamp_ms')
                
                # Calculate previous speed and timestamp
                merged_data['prev_speed'] = merged_data['speed_mps'].shift(1)
                merged_data['prev_timestamp'] = merged_data['timestamp_ms'].shift(1)
                merged_data['time_diff'] = (merged_data['timestamp_ms'] - merged_data['prev_timestamp']) / 1000  # in seconds
                
                # Calculate acceleration
                merged_data['acceleration'] = merged_data.apply(
                    lambda row: (row['speed_mps'] - row['prev_speed']) / row['time_diff'] 
                    if pd.notna(row['prev_speed']) and row['time_diff'] > 0 else 0,
                    axis=1
                )
        
        # Extract temporal features for mode prediction
        if 'timestamp_dt' in merged_data.columns:
            merged_data['hour'] = merged_data['timestamp_dt'].dt.hour
            merged_data['day_of_week'] = merged_data['timestamp_dt'].dt.dayofweek
            
            # Create cyclic encoding of hour (SHL feature)
            merged_data['hour_sin'] = np.sin(2 * np.pi * merged_data['hour'] / 24)
            merged_data['hour_cos'] = np.cos(2 * np.pi * merged_data['hour'] / 24)
            
            # Create cyclic encoding of day of week (SHL feature)
            merged_data['day_sin'] = np.sin(2 * np.pi * merged_data['day_of_week'] / 7)
            merged_data['day_cos'] = np.cos(2 * np.pi * merged_data['day_of_week'] / 7)
        
        # Create trajectory ID for GroupKFold validation (SHL approach)
        merged_data['trajectory_id'] = (
            (merged_data['timestamp_ms'] // (60 * 60 * 1000))  # Group by hour
        ).astype(str)
        
        # Create feature set - aligned with SHL dataset features
        feature_columns = [
            'speed_mps',              # Primary separator of modes
            'bearing_change',         # Heading changes are different by mode
            'scan_count',             # WiFi scanning behavior varies by mode
            'unique_bssids'           # Different density of APs in different modes
        ]
        
        # Add acceleration features if available
        if 'acceleration' in merged_data.columns:
            feature_columns.append('acceleration')
            
        # Add WiFi scan interval features if available
        if 'scan_interval_mean' in merged_data.columns:
            feature_columns.append('scan_interval_mean')
            feature_columns.append('scan_interval_std')
        
        # Add temporal features if available
        if 'hour_sin' in merged_data.columns:
            feature_columns.extend(['hour_sin', 'hour_cos', 'day_sin', 'day_cos'])
        
        # Create features dataframe
        X = merged_data[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Create rule-based labels based on speed (SHL approach)
        merged_data['transport_mode'] = pd.cut(
            merged_data['speed_mps'],
            bins=[0, self.mode_thresholds['still'], self.mode_thresholds['walk'], 
                  self.mode_thresholds['bike'], float('inf')],
            labels=['still', 'walk', 'bike', 'vehicle']
        )
        
        # Convert to numeric for ML
        merged_data['transport_mode_code'] = merged_data['transport_mode'].map(self.inverse_mode_mapping)
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        return X, merged_data['transport_mode_code'], merged_data

    def build_pipeline(self):
        """
        Build the model pipeline with preprocessing and classifier
        Based on successful SHL challenge approaches
        """
        print("Building transport mode detection pipeline...")
        
        # Preprocessor with StandardScaler
        preprocessor = StandardScaler()
        
        # Gradient Boosting classifier - higher n_estimators for better stability
        classifier = GradientBoostingClassifier(
            n_estimators=200,  # Increased from 100
            learning_rate=0.05,  # Decreased for better generalization
            max_depth=6,  # Slightly higher complexity
            subsample=0.8,  # Subsampling for robustness
            random_state=42
        )
        
        # Alternative is to use RandomForest
        # classifier = RandomForestClassifier(
        #     n_estimators=200,
        #     max_depth=15,
        #     min_samples_split=5,
        #     min_samples_leaf=2,
        #     random_state=42
        # )
        
        # Pipeline with preprocessing and classifier
        # Calibration to provide reliable probabilities
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', CalibratedClassifierCV(classifier, method='sigmoid', cv=5))
        ])
    
    def train_model(self, X, y, groups=None):
        """
        Train the transport mode detection model
        
        Parameters:
        -----------
        X : DataFrame
            Features for model training
        y : Series
            Target variable (transport_mode_code)
        groups : Series, optional
            Groups for cross-validation (trajectory_id)
            
        Returns:
        --------
        self : object
            Returns self
        """
        print("Training transport mode detection model...")
        
        # Build the pipeline if not already built
        if self.pipeline is None:
            self.build_pipeline()
        
        # Train the model
        self.pipeline.fit(X, y)
        
        return self
    
    def evaluate_model(self, X, y, groups=None, n_splits=5):
        """
        Evaluate the model with cross-validation
        
        Parameters:
        -----------
        X : DataFrame
            Features for evaluation
        y : Series
            Target variable (transport_mode_code)
        groups : Series, optional
            Groups for cross-validation (trajectory_id)
        n_splits : int
            Number of cross-validation splits
            
        Returns:
        --------
        results : dict
            Cross-validation results
        """
        print(f"Evaluating model with {n_splits}-fold cross-validation...")
        
        # Initialize results
        all_predictions = []
        all_true_values = []
        
        # Initialize GroupKFold
        if groups is not None:
            cv = GroupKFold(n_splits=n_splits)
            splits = cv.split(X, y, groups=groups)
        else:
            # Use regular KFold if no groups provided
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = cv.split(X, y)
        
        # Loop through cross-validation splits
        for train_idx, test_idx in splits:
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            if self.pipeline is None:
                self.build_pipeline()
            self.pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.pipeline.predict(X_test)
            
            # Store results
            all_predictions.extend(y_pred)
            all_true_values.extend(y_test)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_true_values = np.array(all_true_values)
        
        # Generate classification report
        report = classification_report(all_true_values, all_predictions, output_dict=True)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_true_values, all_predictions)
        
        # Calculate macro F1 score
        macro_f1 = f1_score(all_true_values, all_predictions, average='macro')
        
        # Store results
        results = {
            'classification_report': report,
            'confusion_matrix': cm,
            'macro_f1': macro_f1
        }
        
        # Print summary
        print(f"Macro F1 Score: {macro_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_true_values, all_predictions))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.mode_mapping.values()),
                   yticklabels=list(self.mode_mapping.values()))
        plt.title('Transport Mode Detection Confusion Matrix')
        plt.xlabel('Predicted Mode')
        plt.ylabel('True Mode')
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/transport_mode_confusion_matrix.png")
        plt.close()
        
        return results
    
    def predict(self, X):
        """
        Predict transport modes
        
        Parameters:
        -----------
        X : DataFrame
            Features for prediction
            
        Returns:
        --------
        predictions : Series
            Predicted transport modes (numeric codes)
        probabilities : DataFrame
            Prediction probabilities for each class
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Make predictions
        predictions = self.pipeline.predict(X)
        
        # Get probabilities
        probabilities = self.pipeline.predict_proba(X)
        
        # Convert probabilities to DataFrame
        prob_df = pd.DataFrame(
            probabilities,
            columns=[f'prob_{self.mode_mapping[i]}' for i in range(len(self.mode_mapping))]
        )
        
        return predictions, prob_df
    
    def save_model(self, filename='transport_mode_detector.pkl'):
        """
        Save the model to disk
        
        Parameters:
        -----------
        filename : str
            Filename to save the model
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Create full path
        path = os.path.join(self.models_dir, filename)
        
        # Save model
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, filename='transport_mode_detector.pkl'):
        """
        Load the model from disk
        
        Parameters:
        -----------
        filename : str
            Filename to load the model from
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Create full path
        path = os.path.join(self.models_dir, filename)
        
        # Load model
        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)
        
        print(f"Model loaded from {path}")
        return self
    
    def create_plots(self, X, y, merged_data):
        """
        Create plots for transport mode analysis
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : Series
            True transport modes
        merged_data : DataFrame
            Complete data with all columns
        """
        print("Creating transport mode analysis plots...")
        
        # Create plots directory
        os.makedirs(f"{self.plots_dir}/transport_modes", exist_ok=True)
        
        # Add string mode for plotting
        merged_data['transport_mode'] = merged_data['transport_mode_code'].map(self.mode_mapping)
        
        # 1. Speed distribution by transport mode
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='transport_mode', y='speed_mps', data=merged_data)
        plt.title('Speed Distribution by Transport Mode')
        plt.xlabel('Transport Mode')
        plt.ylabel('Speed (m/s)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/transport_modes/speed_by_mode.png")
        plt.close()
        
        # 2. Bearing change distribution by transport mode
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='transport_mode', y='bearing_change', data=merged_data)
        plt.title('Heading Change Distribution by Transport Mode')
        plt.xlabel('Transport Mode')
        plt.ylabel('Heading Change (degrees)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/transport_modes/heading_by_mode.png")
        plt.close()
        
        # 3. WiFi scan count distribution by transport mode
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='transport_mode', y='scan_count', data=merged_data)
        plt.title('WiFi Scan Count by Transport Mode')
        plt.xlabel('Transport Mode')
        plt.ylabel('WiFi Scan Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/transport_modes/wifi_scans_by_mode.png")
        plt.close()
        
        # 4. Hourly distribution of transport modes
        if 'hour' in merged_data.columns:
            hourly_modes = merged_data.groupby(['hour', 'transport_mode']).size().unstack(fill_value=0)
            hourly_modes_pct = hourly_modes.div(hourly_modes.sum(axis=1), axis=0) * 100
            
            plt.figure(figsize=(15, 7))
            hourly_modes_pct.plot(kind='bar', stacked=True, cmap='viridis')
            plt.title('Hourly Distribution of Transport Modes')
            plt.xlabel('Hour of Day')
            plt.ylabel('Percentage')
            plt.legend(title='Transport Mode')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/transport_modes/hourly_mode_distribution.png")
            plt.close()
        
        # 5. Spatial distribution of transport modes
        if all(col in merged_data.columns for col in ['latitude_deg', 'longitude_deg']):
            plt.figure(figsize=(12, 10))
            
            # Plot separately for each mode
            modes = merged_data['transport_mode'].unique()
            cmap = plt.cm.get_cmap('viridis', len(modes))
            
            for i, mode in enumerate(modes):
                mode_data = merged_data[merged_data['transport_mode'] == mode]
                plt.scatter(
                    mode_data['longitude_deg'],
                    mode_data['latitude_deg'],
                    s=30,
                    alpha=0.5,
                    c=[cmap(i)],
                    label=mode
                )
            
            plt.title('Spatial Distribution of Transport Modes')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/transport_modes/spatial_mode_distribution.png")
            plt.close()

def detect_transport_modes(gps_df, wifi_df, merged_df=None, mode_thresholds=None, output_dir='output', models_dir='models', plots_dir='plots'):
    """
    Detect transport modes from GPS and WiFi data
    
    Parameters:
    -----------
    gps_df : DataFrame
        GPS data
    wifi_df : DataFrame
        WiFi data
    merged_df : DataFrame, optional
        Already merged data with both GPS and WiFi information
    mode_thresholds : dict, optional
        Thresholds for transport mode classification
    output_dir : str, default='output'
        Directory to save output files
    models_dir : str, default='models'
        Directory to save trained models
    plots_dir : str, default='plots'
        Directory to save plots
        
    Returns:
    --------
    merged_data : DataFrame
        Data with transport mode predictions
    detector : TransportModeDetector
        Trained detector model
    """
    print("Detecting transport modes...")
    
    # Create detector
    detector = TransportModeDetector(
        output_dir=output_dir,
        models_dir=models_dir,
        plots_dir=plots_dir
    )
    
    # Update thresholds if provided
    if mode_thresholds is not None:
        detector.mode_thresholds.update(mode_thresholds)
    
    # Prepare features
    X, y, merged_data = detector.prepare_features(gps_df, wifi_df, merged_df)
    
    # Build and train model
    detector.build_pipeline()
    detector.train_model(X, y)
    
    # Evaluate model
    results = detector.evaluate_model(X, y, groups=merged_data.get('trajectory_id'))
    
    # Create plots
    detector.create_plots(X, y, merged_data)
    
    # Save model
    detector.save_model()
    
    # Make predictions (to ensure we have probabilities in the output)
    predictions, probabilities = detector.predict(X)
    
    # Add predictions to merged data
    merged_data['predicted_mode_code'] = predictions
    merged_data['predicted_mode'] = merged_data['predicted_mode_code'].map(detector.mode_mapping)
    
    # Add probability columns
    for col in probabilities.columns:
        merged_data[col] = probabilities[col].values
    
    return merged_data, detector

if __name__ == "__main__":
    # Test the module with sample data
    try:
        print("Loading GPS data...")
        gps_df = pd.read_csv('cleaned_gps_data.csv')
        
        print("Loading WiFi data...")
        wifi_df = pd.read_csv('cleaned_wifi_data.csv')
        
        print("Loading merged data...")
        try:
            merged_df = pd.read_csv('merged_wifi_location.csv')
            print(f"Using merged data with {len(merged_df)} rows")
            
            merged_data, detector = detect_transport_modes(None, None, merged_df=merged_df)
        except FileNotFoundError:
            print("Merged data not found, using separate GPS and WiFi files")
            merged_data, detector = detect_transport_modes(gps_df, wifi_df)
        
        # Save results to CSV
        merged_data.to_csv('transport_modes.csv', index=False)
        print("Transport modes saved to transport_modes.csv")
        
    except Exception as e:
        print(f"Error during transport mode detection: {e}") 