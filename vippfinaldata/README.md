# Cellular/WiFi Coverage Analysis System

A comprehensive system for analyzing and predicting WiFi and cellular coverage using machine learning techniques. This project includes mobility-aware threshold adaptation and human-flow mapping for improved accuracy.

## Features

- **WiFi Coverage Prediction**: Analyze and predict WiFi signal coverage across geographical areas
- **Mobility-Aware Thresholds**: Dynamically adjust RSSI thresholds based on transport modes
- **Human-Flow Mapping**: Use crowdsourced mobility data to improve AP planning
- **Anomaly Detection**: Identify signal anomalies and distinguish from mobility-induced variations
- **Interactive Visualizations**: View coverage maps, signal predictions, and analyses

## Docker Setup

### Prerequisites

- Docker
- Docker Compose

### Building and Running with Docker

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd vippfinaldata
   ```

2. **Build and start the containers**:
   ```bash
   docker-compose up --build
   ```

3. **Access the application**:
   Open your browser and navigate to `http://localhost:8000`

4. **Stop the containers**:
   ```bash
   docker-compose down
   ```

### Docker Volume Mounts

- `./vippfinaldata:/app` - Application code
- `./data:/app/data` - Input data
- `./plots:/app/plots` - Generated plots
- `./models:/app/models` - Trained models
- `./output:/app/output` - Analysis output

## Manual Setup (without Docker)

1. **Install required packages**:
   ```bash
   pip install -r vippfinaldata/requirements.txt
   ```

2. **Run the server**:
   ```bash
   cd vippfinaldata
   python manage.py runserver
   ```

## Usage

- **Home**: Overview of the system
- **Run Pipeline**: Execute data analysis and model building
- **View Results**: Explore analysis results and model performance
- **EDA Steps**: View data exploration process
- **Model Details**: Examine model structure and performance
- **Coverage Maps**: View geographical coverage predictions
- **Anomaly Detection**: Identify signal anomalies
- **Mobility-Aware Thresholds**: Learn about dynamic threshold adaptation
- **Human-Flow Mapping**: Explore crowd-sourced AP planning

## Developers

- Rami Eid
- Omar Kaaki
- Mohamad Chmaitily 