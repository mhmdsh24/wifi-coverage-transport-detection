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

3. Download the required datasets:
   - Access the datasets via Google Drive: [WiFi Coverage Analysis Datasets](https://drive.google.com/drive/folders/1vQKn8IW6T640FIpk4j7a_UAGNGtwFIMI?usp=sharing)
   - The Google Drive folder contains two subfolders:
     - `raw data files/`: Contains the original data files you need to run the pipeline from scratch
     - `preprossed data files/`: Contains preprocessed data if you want to skip the initial data processing steps
   - For full pipeline execution, download the following files from the `raw data files/` folder and place them in the `vippfinaldata/vippfinaldata` directory:
     - Hips_WiFi.csv
     - Hips_Location.csv
     - Hips_GPS.csv
     - Hips_Motion.csv
   - For quicker testing or to skip preprocessing, you can use files from the `preprossed data files/` folder instead
   - For detailed dataset information, see [DATA.md](DATA.md)

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

This project requires several large datasets that are not included in the Git repository. These datasets are available via Google Drive:

- **Access the datasets**: [WiFi Coverage Analysis Datasets](https://drive.google.com/drive/folders/1vQKn8IW6T640FIpk4j7a_UAGNGtwFIMI?usp=sharing)

### Google Drive Folder Structure

The Google Drive folder contains:
1. **raw data files/** - Original datasets:
   - `Hips_WiFi.csv`: WiFi signal data
   - `Hips_Location.csv`: Location data
   - `Hips_GPS.csv`: GPS tracking data
   - `Hips_Motion.csv`: Motion sensor data

2. **preprossed data files/** - Pre-processed datasets to skip initial steps:
   - `cleaned_wifi_data.csv`: Cleaned WiFi data
   - `cleaned_location_data.csv`: Cleaned location data
   - `cleaned_gps_data.csv`: Cleaned GPS data
   - `merged_wifi_location.csv`: Merged WiFi and location data
   - `grid_coverage_statistics.csv`: Coverage statistics
   - `signal_anomalies.csv`: Detected anomalies

See [DATA.md](DATA.md) for complete details on data requirements and usage.

## Developers

- Rami Eid
- Omar Kaaki
- Mohamad Chmaitily 