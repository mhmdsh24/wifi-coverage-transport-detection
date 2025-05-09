# Cellular/WiFi Coverage Analysis System

A comprehensive system for analyzing and predicting WiFi and cellular coverage using machine learning techniques. This project includes mobility-aware threshold adaptation and human-flow mapping for improved accuracy.

## Features

- **WiFi Coverage Prediction**: Analyze and predict WiFi signal coverage across geographical areas
- **Mobility-Aware Thresholds**: Dynamically adjust RSSI thresholds based on transport modes
- **Human-Flow Mapping**: Use crowdsourced mobility data to improve AP planning
- **Anomaly Detection**: Identify signal anomalies and distinguish from mobility-induced variations
- **Interactive Visualizations**: View coverage maps, signal predictions, and analyses

## Project Structure

- `django_app/`: Web application built with Django
- `vippfinaldata/`: Main application code
  - `Sussex-Huawei/`: Data processing utilities
  - Core ML modules:
    - `wifi_coverage_model.py`: WiFi coverage prediction
    - `transport_mode_detection.py`: Transport mode classification
    - `signal_anomaly_detection.py`: Signal anomaly detection
    - `unified_model.py`: Combined models for analysis
    - `enhanced_transport_classifier.py`: Advanced transport classification
    - `threshold_sensitivity.py`: Dynamic threshold adaptation
  - Pipeline runners:
    - `run_enhanced_pipeline.py`: Enhanced analysis pipeline
    - `run_integrated_pipeline.py`: Integrated analysis pipeline
    - `run_wifi_pipeline.py`: WiFi-specific pipeline
    - `run_demo.py`: Demo pipeline

## Setup Instructions

### Prerequisites

- Python 3.8+
- Django 3.2+
- Scikit-learn, Pandas, NumPy, Matplotlib

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/wifi-coverage-analysis.git
   cd wifi-coverage-analysis
   ```

2. Install the required dependencies:
   ```bash
   pip install -r vippfinaldata/requirements.txt
   ```

3. Download the required datasets (not included in the repository due to size):
   - Place the following files in the `vippfinaldata` directory:
     - Hips_WiFi.csv
     - Hips_Location.csv
     - Hips_GPS.csv
     - Hips_Motion.csv

### Running the Application

#### Using Docker

1. Make sure Docker and Docker Compose are installed
2. Run:
   ```bash
   docker-compose up --build
   ```
3. Access the web interface at `http://localhost:8000`

#### Manual Execution

1. Run the web application:
   ```bash
   cd vippfinaldata
   python run_webapp.py
   ```

2. Run the analysis pipeline:
   ```bash
   python run_enhanced_pipeline.py
   ```

## Usage

- **Home**: Overview of the system
- **Run Pipeline**: Execute data analysis and model building
- **View Results**: Explore analysis results and model performance
- **Coverage Maps**: View geographical coverage predictions
- **Anomaly Detection**: Identify signal anomalies
- **Mobility-Aware Thresholds**: Learn about dynamic threshold adaptation

## Data Requirements

This project requires several large datasets that are not included in the Git repository. You'll need to obtain these separately and place them in the appropriate directory:

- `Hips_WiFi.csv`: WiFi signal data
- `Hips_Location.csv`: Location data
- `Hips_GPS.csv`: GPS tracking data
- `Hips_Motion.csv`: Motion sensor data

## Developers

- Rami Eid
- Omar Kaaki
- Mohamad Chmaitily 