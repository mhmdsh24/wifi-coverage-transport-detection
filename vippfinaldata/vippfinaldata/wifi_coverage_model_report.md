# WiFi Coverage Prediction Model - Final Report

## Project Overview
This project implements a robust WiFi coverage prediction model using the Sussex‑Huawei Locomotion (SHL) dataset. The model predicts areas with low WiFi signal coverage based on signal strength (RSSI) and other derived features, without leaking information from the target variable.

## Key Improvements

### 1. Data Engineering Improvements
- **Fixed data integration**: Implemented proper timestamp synchronization between WiFi and location data with adaptive tolerance (300 seconds)
- **Time window selection**: Added support for filtering data by time window to ensure maximum overlap between datasets
- **Hidden SSID handling**: Replaced hidden SSIDs with "__hidden__" placeholder instead of global constants
- **Hampel filter outlier detection**: Applied robust outlier filtering that preserves spatial variance

### 2. Anomaly Detection
- **Isolation Forest model**: Implemented dedicated anomaly detection using Isolation Forest
- **Feature integration**: Integrated anomaly detection results into coverage prediction as additional features
- **Spatial analysis**: Added anomaly density metrics to grid-level statistics

### 3. Modeling Improvements
- **Strict validation**: Implemented spatial GroupKFold cross-validation to eliminate data leakage
- **Feature selection**: Carefully excluded direct RSSI values (mean, min, max) to prevent target leakage
- **Ensemble approach**: Combined Random Forest and Gradient Boosting models for better predictions
- **Permutation importance**: Used permutation methods to correctly identify important features

### 4. Visualization
- **Coverage maps**: Generated interpolated coverage maps and identified poor coverage areas
- **Anomaly visualization**: Created maps showing relationship between anomalies and coverage issues
- **Feature analysis**: Visualized the relationship between input features and prediction probabilities

## Final Model Performance

### Cross-validation Results
- **Random Forest**: 70.4% accuracy, 68.5% ROC-AUC
- **Gradient Boosting**: 71.4% accuracy, 69.1% ROC-AUC

These metrics represent realistic performance without data leakage, compared to the original misleading 100% accuracy.

### Most Important Features
1. **BSSID count**: Number of unique WiFi networks in each grid cell (80.3% importance)
2. **RSSI count**: Number of measurements per grid/BSSID (13.9% importance)
3. **RSSI standard deviation**: Variability of signal strength (1.1% importance)
4. **Anomaly metrics**: Anomaly rate and density (combined ~2% importance)

## Recommendations for Further Improvement

1. **Context enrichment**: Add user movement mode (still/walk/vehicle) from SHL dataset
2. **Multi-radio integration**: Include cellular signal data for a comprehensive coverage model
3. **Urban morphology features**: Add building height & land-use from OpenStreetMap
4. **Temporal modeling**: Implement time-series forecasting for coverage prediction
5. **Online deployment**: Create a real-time ETL pipeline for continuous prediction updates

## Conclusion
The improved model provides reliable predictions of WiFi coverage areas without information leakage. The use of proper spatial cross-validation and careful feature selection produces realistic accuracy metrics. The model successfully identifies areas of poor WiFi coverage and can be used for network planning and optimization. 