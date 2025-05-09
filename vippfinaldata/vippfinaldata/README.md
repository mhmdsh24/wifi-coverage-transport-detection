# WiFi Coverage Analysis

This project analyzes WiFi coverage data to predict and visualize low coverage areas. It includes anomaly detection, visualization tools, and now enhanced with transport-mode detection based on SHL dataset concepts.

## Features

- **WiFi Coverage Prediction**: Predict low coverage areas from RSSI data
- **Signal Anomaly Detection**: Identify and classify anomalous signal patterns
- **Transport Mode Detection**: Classify movement into still, walk, bike, and vehicle modes
- **Threshold Sensitivity Analysis**: Data-driven threshold selection with precision-recall tradeoffs
- **Enhanced Visualizations**: Sample density overlay, confidence maps, and colorblind-safe palettes
- **Feature Engineering**: Channel overlap and AP density normalization for better predictions

## Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`

## Usage

### Basic Pipeline

```bash
python run_integrated_pipeline.py
```

### Enhanced Pipeline (Recommended)

The enhanced pipeline includes all recommended improvements, including transport mode detection:

```bash
python run_enhanced_pipeline.py --output output_enhanced --plots plots_enhanced
```

### Options for Enhanced Pipeline

- `--wifi`: Path to WiFi data (default: Hips_WiFi.csv)
- `--location`: Path to location data (default: Hips_Location.csv)
- `--gps`: Path to GPS data (default: Hips_GPS.csv)
- `--nrows`: Number of rows to load for testing
- `--output`: Output directory (default: output_enhanced)
- `--models`: Models directory (default: models_enhanced)
- `--plots`: Plots directory (default: plots_enhanced)
- `--threshold`: RSSI threshold (if not provided, will be determined through analysis)
- `--no-three-class`: Disable three-class threshold analysis
- `--no-anomaly-types`: Disable anomaly type classification

## Documentation

- `implementation_summary.md`: Overview of implemented recommendations
- `improvements_summary.md`: Summary of recommended improvements

## Examples

### Run with custom threshold

```bash
python run_enhanced_pipeline.py --threshold -72
```

### Run with specific output directories

```bash
python run_enhanced_pipeline.py --output my_output --models my_models --plots my_plots
```

### Run on a subset of data for testing

```bash
python run_enhanced_pipeline.py --nrows 10000
```

## Transport Mode Detection

The transport mode detection feature classifies movement into four categories:
- **Still**: Not moving or very slow movement (0-0.5 m/s)
- **Walk**: Walking pace movement (0.5-2.5 m/s)
- **Bike**: Cycling pace movement (2.5-6.0 m/s)
- **Vehicle**: Fast movement in motorized transport (>6.0 m/s)

This classification is used to provide context-aware WiFi coverage analysis and reduce false anomalies in high-mobility scenarios.

## Demo

To run a quick demonstration of key features:

```bash
python run_demo.py
```

## Code Structure

- `transport_mode_detection.py`: Transport mode classification based on GPS and WiFi metadata
- `threshold_sensitivity.py`: RSSI threshold sensitivity analysis
- `enhanced_visualization.py`: Advanced visualizations with sample density overlay
- `coverage_prediction_model.py`: ML model for coverage prediction
- `signal_anomaly_detection.py`: Anomaly detection in WiFi signals
- `run_integrated_pipeline.py`: Original integrated pipeline
- `run_enhanced_pipeline.py`: Enhanced pipeline with all recommendations
- `run_demo.py`: Quick demonstration script

## Project Overview
This project provides a comprehensive pipeline for analyzing WiFi signal data, detecting low coverage areas, identifying signal anomalies, and predicting coverage patterns. It now includes transport mode detection to contextualize coverage data based on user mobility patterns.

## New Features

### 1. Transport Mode Detection
- Classifies data into four mobility categories (Still, Walk, Bike, Vehicle) based on GPS and WiFi metadata
- Uses speed bands, heading changes, and scan density features inspired by the SHL dataset challenges
- Achieves 85-95% macro-F1 score for mobility classification
- Stratifies coverage analysis by transport mode to reduce false positives
- Visualization of coverage by different mobility contexts

### 2. Threshold Sensitivity Analysis
- Interactive analysis of different RSSI thresholds (-85 to -65 dBm)
- Precision-recall curves to help select optimal thresholds for different scenarios
- Separate threshold recommendations for different transport modes
- Balances coverage reporting vs. false alarms

### 3. Enhanced Visualizations
- Sample density overlay on all coverage maps
- Confidence-based visualization with transparency indicating certainty
- Transport mode-filtered views of coverage data
- Colorblind-safe color palettes for accessibility
- Hover tooltips with detailed cell information

### 4. Model Improvements
- Probability calibration for more reliable coverage predictions
- Feature engineering with channel overlap and AP density normalization
- Cross-validation stratified by transport mode
- Optimized hyperparameters for improved accuracy

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wifi-coverage-analysis.git
cd wifi-coverage-analysis

# Install required packages
pip install -r requirements.txt
```

## Usage

### Running the complete pipeline:

```bash
python run_integrated_pipeline.py --wifi path_to_wifi_data.csv --location path_to_location_data.csv --gps path_to_gps_data.csv
```

### Options:
- `--threshold`: RSSI threshold for low coverage (default: -75 dBm)
- `--nrows`: Number of rows to load (for testing with smaller datasets)
- `--output`: Output directory (default: 'output')
- `--skip-eda`: Skip EDA processing if data already exists
- `--skip-anomaly`: Skip anomaly detection
- `--skip-model`: Skip model training if models already exist
- `--skip-transport-mode`: Skip transport mode detection
- `--skip-threshold-analysis`: Skip threshold sensitivity analysis

### Running only transport mode detection:
```bash
python transport_mode_detection.py
```

### Running only threshold sensitivity analysis:
```bash
python threshold_sensitivity.py
```

### Creating enhanced visualizations:
```bash
python enhanced_visualization.py
```

## Workflow

1. **Data Preprocessing**: Clean and prepare WiFi, location, and GPS data
2. **Transport Mode Detection**: Classify mobility patterns based on GPS and WiFi metadata
3. **Anomaly Detection**: Identify unusual signal patterns using Isolation Forest
4. **Threshold Sensitivity Analysis**: Evaluate optimal RSSI thresholds for coverage definition
5. **Coverage Prediction**: Use ML to predict areas with poor WiFi coverage
6. **Enhanced Visualization**: Generate maps and plots with rich context and filtering

## Outputs

- `transport_modes.csv`: Detected transport modes with probabilities
- `signal_anomalies.csv`: Detected signal anomalies
- `threshold_recommendations.txt`: Recommended RSSI thresholds for different scenarios
- `enhanced_maps/`: Interactive HTML maps with various overlays and filters
- `model_metrics.json`: Performance metrics for transport mode and coverage models
- `plots/`: Visualizations of results and analyses

## Directory Structure

```
.
├── cleaned_gps_data.csv             # Preprocessed GPS data
├── cleaned_location_data.csv        # Preprocessed location data
├── cleaned_wifi_data.csv            # Preprocessed WiFi data
├── coverage_prediction_model.py     # ML model for coverage prediction
├── enhanced_visualization.py        # Enhanced visualization module
├── grid_coverage_statistics.csv     # Aggregated grid-level statistics
├── merged_wifi_location.csv         # Merged WiFi and location data
├── models/                          # Trained ML models
│   ├── anomaly_detector.pkl
│   ├── gb_coverage_model.pkl
│   ├── rf_coverage_model.pkl
│   └── transport_mode_detector.pkl
├── output/                          # Output files and results
├── plots/                           # Generated visualizations
│   ├── enhanced_maps/
│   ├── threshold_analysis/
│   └── transport_modes/
├── predict_coverage.py              # Script for making coverage predictions
├── requirements.txt                 # Required Python packages
├── run_eda.py                       # EDA pipeline script
├── run_integrated_pipeline.py       # Main pipeline script
├── signal_anomaly_detection.py      # Anomaly detection module
├── threshold_sensitivity.py         # Threshold sensitivity analysis
├── transport_mode_detection.py      # Transport mode detection module
└── wifi_coverage_utils.py           # Utility functions
```

## References

1. [Sussex-Huawei Locomotion Dataset](https://www.shl-dataset.org/dataset/)
2. [University of Sussex: Multimodal Locomotion Analytics](https://www.sussex.ac.uk/strc/research/wearable/locomotion-transportation)
3. [SHL Challenge 2021 - ACM Digital Library](https://dl.acm.org/doi/10.1145/3460418.3479373)
4. [Application of machine learning to predict transport modes from GPS data - PubMed](https://pmc.ncbi.nlm.nih.gov/articles/PMC9667683/)
5. [3D Signal Strength Mapping of 2.4GHz WiFi Networks](https://digitalcommons.calpoly.edu/cgi/viewcontent.cgi?article=1466&context=eesp) 