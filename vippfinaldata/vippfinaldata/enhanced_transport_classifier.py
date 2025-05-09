import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class EnhancedTransportClassifier:
    """
    Enhanced Transport Mode Classification using 1D CNN approach inspired by SHL dataset
    
    Classifies transport modes into:
    - still: Stationary
    - walk: Walking
    - run: Running
    - bike: Cycling
    - car: Private vehicle
    - bus: Public bus
    - train: Train transport
    - subway: Subway/metro transport
    
    Uses GPS, WiFi, and motion data to predict transportation mode with higher granularity
    than the basic classifier.
    
    References:
    [1] Sussex-Huawei Locomotion Dataset (https://www.shl-dataset.org/)
    [2] SHL Challenge 2018 (http://www.shl-dataset.org/activity-recognition-challenge/)
    """
    
    def __init__(self, output_dir='output', models_dir='models', window_size=5):
        """
        Initialize the enhanced transport classifier
        
        Parameters:
        -----------
        output_dir : str
            Directory to save output files
        models_dir : str
            Directory to save trained models
        window_size : int
            Size of the sliding window in seconds
        """
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.window_size = window_size
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Define mode mappings
        self.mode_mapping = {
            0: 'still',
            1: 'walk',
            2: 'run',
            3: 'bike',
            4: 'car',
            5: 'bus', 
            6: 'train',
            7: 'subway'
        }
        
        # Inverse mapping
        self.inverse_mode_mapping = {v: k for k, v in self.mode_mapping.items()}
        
        # Dynamic threshold mapping by mode
        self.threshold_mapping = {
            'still': -75,  # stationary
            'walk': -75,   # walking
            'run': -78,    # running 
            'bike': -80,   # cycling
            'car': -83,    # in car
            'bus': -83,    # in bus
            'train': -87,  # in train
            'subway': -87  # in subway
        }
        
        # Group mappings for basic modes
        self.mode_groups = {
            'still': ['still'],
            'walk': ['walk', 'run'],
            'bike': ['bike'],
            'vehicle': ['car', 'bus', 'train', 'subway']
        }
        
        # Initialize model
        self.model = None
        self.scaler = StandardScaler()
    
    def build_cnn_model(self, input_shape, num_classes=8):
        """
        Build a 1D CNN model for transport mode classification
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data for the model
        num_classes : int
            Number of transport mode classes
            
        Returns:
        --------
        model : keras.Model
            Built CNN model
        """
        inputs = Input(shape=input_shape)
        
        # First convolutional block
        x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        # Second convolutional block
        x = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        # Third convolutional block
        x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        
        # Fully connected layers
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_features(self, gps_df, wifi_df, motion_df=None):
        """
        Prepare features for enhanced transport classification
        
        Parameters:
        -----------
        gps_df : DataFrame
            GPS data with timestamps, coordinates, and speed
        wifi_df : DataFrame
            WiFi data with timestamps and RSSI values
        motion_df : DataFrame, optional
            Motion sensor data if available
            
        Returns:
        --------
        X : numpy.ndarray
            Features for transport mode classification
        y : numpy.ndarray, optional
            Target labels if available
        windows_df : DataFrame
            Metadata for each window
        """
        print("Preparing features for enhanced transport classification...")
        
        # Check required GPS columns
        required_gps_cols = ['timestamp_ms', 'latitude_deg', 'longitude_deg', 'speed_mps']
        if not all(col in gps_df.columns for col in required_gps_cols):
            raise ValueError(f"GPS data missing required columns: {required_gps_cols}")
            
        # Sort data by timestamp
        gps_df = gps_df.sort_values('timestamp_ms').copy()
        wifi_df = wifi_df.sort_values('timestamp_ms').copy()
        
        # Convert timestamps to datetime
        if 'timestamp_dt' not in gps_df.columns:
            gps_df['timestamp_dt'] = pd.to_datetime(gps_df['timestamp_ms'], unit='ms')
        
        if 'timestamp_dt' not in wifi_df.columns:
            wifi_df['timestamp_dt'] = pd.to_datetime(wifi_df['timestamp_ms'], unit='ms')
        
        # Create sliding windows
        window_size_ms = self.window_size * 1000
        window_step_ms = window_size_ms // 2  # 50% overlap
        
        # Get start and end timestamps
        min_ts = min(gps_df['timestamp_ms'].min(), wifi_df['timestamp_ms'].min())
        max_ts = max(gps_df['timestamp_ms'].max(), wifi_df['timestamp_ms'].max())
        
        # Generate window boundaries
        window_starts = np.arange(min_ts, max_ts, window_step_ms)
        
        # Initialize arrays for features
        windows = []
        
        # Process each window
        for start_ts in window_starts:
            end_ts = start_ts + window_size_ms
            
            # Get GPS data in this window
            gps_window = gps_df[(gps_df['timestamp_ms'] >= start_ts) & 
                               (gps_df['timestamp_ms'] < end_ts)]
            
            # Get WiFi data in this window
            wifi_window = wifi_df[(wifi_df['timestamp_ms'] >= start_ts) & 
                                (wifi_df['timestamp_ms'] < end_ts)]
            
            # Skip if not enough data
            if len(gps_window) < 3 or len(wifi_window) < 3:
                continue
                
            # Extract GPS features
            speed_mean = gps_window['speed_mps'].mean()
            speed_std = gps_window['speed_mps'].std()
            speed_max = gps_window['speed_mps'].max()
            
            # Calculate acceleration
            gps_window = gps_window.sort_values('timestamp_ms')
            gps_window['speed_diff'] = gps_window['speed_mps'].diff()
            gps_window['time_diff'] = gps_window['timestamp_ms'].diff() / 1000  # in seconds
            gps_window['acceleration'] = gps_window['speed_diff'] / gps_window['time_diff']
            
            acc_mean = gps_window['acceleration'].mean()
            acc_std = gps_window['acceleration'].std()
            acc_max = gps_window['acceleration'].abs().max()
            
            # Calculate bearing changes (heading changes)
            bearing_change_mean = 0
            bearing_change_std = 0
            if 'bearing_change' in gps_window.columns:
                bearing_change_mean = gps_window['bearing_change'].mean()
                bearing_change_std = gps_window['bearing_change'].std()
            
            # Extract WiFi features
            wifi_count = len(wifi_window)
            bssid_count = wifi_window['bssid'].nunique()
            rssi_mean = wifi_window['rssi'].mean()
            rssi_std = wifi_window['rssi'].std()
            rssi_min = wifi_window['rssi'].min()
            
            # Calculate WiFi scan rate and intervals
            wifi_window = wifi_window.sort_values('timestamp_ms')
            wifi_window['scan_interval'] = wifi_window['timestamp_ms'].diff() / 1000  # in seconds
            
            scan_interval_mean = wifi_window['scan_interval'].dropna().mean() if len(wifi_window) > 1 else 0
            scan_interval_std = wifi_window['scan_interval'].dropna().std() if len(wifi_window) > 1 else 0
            
            # Store window features
            window_features = {
                'start_ts': start_ts,
                'end_ts': end_ts,
                'speed_mean': speed_mean,
                'speed_std': speed_std,
                'speed_max': speed_max,
                'acc_mean': acc_mean,
                'acc_std': acc_std, 
                'acc_max': acc_max,
                'bearing_change_mean': bearing_change_mean,
                'bearing_change_std': bearing_change_std,
                'wifi_count': wifi_count,
                'bssid_count': bssid_count,
                'rssi_mean': rssi_mean,
                'rssi_std': rssi_std,
                'rssi_min': rssi_min,
                'scan_interval_mean': scan_interval_mean,
                'scan_interval_std': scan_interval_std
            }
            
            # Add motion features if available
            if motion_df is not None:
                motion_window = motion_df[(motion_df['timestamp_ms'] >= start_ts) & 
                                        (motion_df['timestamp_ms'] < end_ts)]
                
                if len(motion_window) > 0:
                    # Add motion features (accelerometer, gyroscope, etc.)
                    for axis in ['x', 'y', 'z']:
                        if f'acc_{axis}' in motion_window.columns:
                            window_features[f'acc_{axis}_mean'] = motion_window[f'acc_{axis}'].mean()
                            window_features[f'acc_{axis}_std'] = motion_window[f'acc_{axis}'].std()
                            window_features[f'acc_{axis}_max'] = motion_window[f'acc_{axis}'].abs().max()
                        
                        if f'gyro_{axis}' in motion_window.columns:
                            window_features[f'gyro_{axis}_mean'] = motion_window[f'gyro_{axis}'].mean()
                            window_features[f'gyro_{axis}_std'] = motion_window[f'gyro_{axis}'].std()
                            window_features[f'gyro_{axis}_max'] = motion_window[f'gyro_{axis}'].abs().max()
            
            windows.append(window_features)
        
        # Create DataFrame from windows
        windows_df = pd.DataFrame(windows)
        
        # Extract features for model
        feature_cols = [col for col in windows_df.columns 
                      if col not in ['start_ts', 'end_ts', 'transport_mode', 'transport_mode_code']]
        
        X = windows_df[feature_cols].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Reshape for CNN if needed (samples, timesteps, features)
        # For now, we'll treat each window as a single sample
        X = X.reshape(X.shape[0], 1, X.shape[1])
        
        # Return y if transport_mode is available
        y = None
        if 'transport_mode' in windows_df.columns:
            y = windows_df['transport_mode'].map(self.inverse_mode_mapping).values
        
        return X, y, windows_df
    
    def train_model(self, X, y):
        """
        Train the enhanced transport classifier
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
            Target labels
            
        Returns:
        --------
        history : History
            Training history
        """
        print("Training enhanced transport classifier...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build model
        input_shape = (X.shape[1], X.shape[2])
        self.model = self.build_cnn_model(input_shape, num_classes=len(self.mode_mapping))
        
        # Setup callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.models_dir, 'enhanced_transport_model_best.h5'),
            monitor='val_accuracy',
            save_best_only=True
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """
        Predict transport modes
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
            
        Returns:
        --------
        predictions : numpy.ndarray
            Predicted transport mode codes
        probabilities : numpy.ndarray
            Predicted probabilities for each mode
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Train or load a model first.")
        
        # Make predictions
        prob_array = self.model.predict(X)
        predictions = np.argmax(prob_array, axis=1)
        
        return predictions, prob_array
    
    def save_model(self, filename='enhanced_transport_classifier.h5'):
        """
        Save the trained model
        
        Parameters:
        -----------
        filename : str
            Filename to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_path = os.path.join(self.models_dir, filename)
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, 'enhanced_transport_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, filename='enhanced_transport_classifier.h5'):
        """
        Load a trained model
        
        Parameters:
        -----------
        filename : str
            Filename of the saved model
        """
        model_path = os.path.join(self.models_dir, filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = load_model(model_path)
        
        # Load scaler
        scaler_path = os.path.join(self.models_dir, 'enhanced_transport_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        print(f"Model loaded from {model_path}")
    
    def get_threshold_for_mode(self, mode):
        """
        Get the appropriate RSSI threshold for a given transport mode
        
        Parameters:
        -----------
        mode : str
            Transport mode
            
        Returns:
        --------
        threshold : float
            RSSI threshold for the given mode
        """
        # If mode is in our mapping, return the threshold
        if mode in self.threshold_mapping:
            return self.threshold_mapping[mode]
        
        # For backward compatibility with basic modes
        if mode == 'vehicle':
            return -83  # Use car/bus threshold for general vehicle
        
        # Default threshold
        return -75  # Default to still/walk threshold
    
    def plot_confusion_matrix(self, y_true, y_pred, filename='enhanced_transport_confusion.png'):
        """
        Plot confusion matrix for model evaluation
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True labels
        y_pred : numpy.ndarray
            Predicted labels
        filename : str
            Filename to save the plot
        """
        from sklearn.metrics import confusion_matrix
        
        # Get class names
        class_names = [self.mode_mapping[i] for i in range(len(self.mode_mapping))]
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot confusion matrix
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Enhanced Transport Mode Confusion Matrix')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

# Function to classify transport modes using pre-trained model or training a new one
def classify_enhanced_transport_modes(gps_df, wifi_df, motion_df=None, 
                                    output_dir='output', models_dir='models',
                                    load_existing=True):
    """
    Classify transport modes with enhanced granularity
    
    Parameters:
    -----------
    gps_df : DataFrame
        GPS data
    wifi_df : DataFrame
        WiFi data
    motion_df : DataFrame, optional
        Motion sensor data if available
    output_dir : str
        Directory to save outputs
    models_dir : str
        Directory to load/save models
    load_existing : bool
        Whether to load an existing model if available
        
    Returns:
    --------
    windows_df : DataFrame
        Data windows with transport mode predictions
    classifier : EnhancedTransportClassifier
        Trained classifier
    """
    print("Classifying enhanced transport modes...")
    
    # Initialize classifier
    classifier = EnhancedTransportClassifier(
        output_dir=output_dir,
        models_dir=models_dir
    )
    
    # Try to load existing model if requested
    model_loaded = False
    if load_existing:
        try:
            classifier.load_model()
            model_loaded = True
            print("Loaded existing enhanced transport model")
        except FileNotFoundError:
            print("No existing enhanced transport model found. Will train a new one.")
    
    # Prepare features
    X, y, windows_df = classifier.prepare_features(gps_df, wifi_df, motion_df)
    
    # If no model or not loading existing, train a new one
    if not model_loaded:
        # For demonstration purposes, we'll simulate having labeled data
        # In a real application, you would need labeled SHL data to train the model
        
        # Simulate labels based on speed as a simple heuristic
        # In a real application, use actual labeled SHL dataset
        print("Note: Using speed-based heuristics to simulate labels for demo purposes.")
        print("In a real application, use labeled SHL dataset to train the model.")
        
        windows_df['simulated_mode'] = 'still'  # Default
        windows_df.loc[windows_df['speed_mean'] > 0.5, 'simulated_mode'] = 'walk'
        windows_df.loc[windows_df['speed_mean'] > 2.0, 'simulated_mode'] = 'run'
        windows_df.loc[windows_df['speed_mean'] > 2.5, 'simulated_mode'] = 'bike'
        windows_df.loc[windows_df['speed_mean'] > 6.0, 'simulated_mode'] = 'car'
        
        # Add random bus/train/subway for demonstration
        # In a real application, use actual SHL labels
        np.random.seed(42)
        vehicle_mask = windows_df['simulated_mode'] == 'car'
        vehicle_indices = windows_df[vehicle_mask].index
        
        # Randomly assign 30% of vehicles to bus
        bus_indices = np.random.choice(vehicle_indices, 
                                      size=int(0.3 * len(vehicle_indices)), 
                                      replace=False)
        windows_df.loc[bus_indices, 'simulated_mode'] = 'bus'
        
        # Randomly assign 20% of vehicles to train
        remaining_indices = list(set(vehicle_indices) - set(bus_indices))
        train_indices = np.random.choice(remaining_indices,
                                        size=int(0.2 * len(vehicle_indices)),
                                        replace=False)
        windows_df.loc[train_indices, 'simulated_mode'] = 'train'
        
        # Randomly assign 10% of vehicles to subway
        remaining_indices = list(set(remaining_indices) - set(train_indices))
        subway_indices = np.random.choice(remaining_indices,
                                         size=int(0.1 * len(vehicle_indices)),
                                         replace=False)
        windows_df.loc[subway_indices, 'simulated_mode'] = 'subway'
        
        # Map modes to codes
        windows_df['transport_mode_code'] = windows_df['simulated_mode'].map(classifier.inverse_mode_mapping)
        
        # Get numeric labels
        y = windows_df['transport_mode_code'].values
        
        # Train model
        history = classifier.train_model(X, y)
        
        # Save model
        classifier.save_model()
    
    # Make predictions
    predictions, probabilities = classifier.predict(X)
    
    # Add predictions to windows DataFrame
    windows_df['predicted_mode_code'] = predictions
    windows_df['predicted_mode'] = windows_df['predicted_mode_code'].map(classifier.mode_mapping)
    
    # Add probabilities
    for i, mode in classifier.mode_mapping.items():
        windows_df[f'prob_{mode}'] = probabilities[:, i]
    
    # Add appropriate threshold for each window
    windows_df['rssi_threshold'] = windows_df['predicted_mode'].map(classifier.threshold_mapping)
    
    # Save results
    windows_df.to_csv(os.path.join(output_dir, 'enhanced_transport_modes.csv'), index=False)
    
    return windows_df, classifier 