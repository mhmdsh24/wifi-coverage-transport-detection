# Coverage Visualization Web Application

A Django web application for visualizing cellular/WiFi coverage data analysis and model predictions.

## Overview

This web application provides a user-friendly interface to explore and visualize the results of:

1. Exploratory Data Analysis (EDA) of cellular/WiFi coverage data
2. Machine learning model evaluation and performance metrics
3. Interactive coverage maps and prediction capabilities

## Features

- **EDA Visualization**: View all exploratory data analysis outputs for WiFi, location, and GPS data
- **Model Evaluation Dashboard**: Examine performance metrics and visualizations of the trained models
- **Interactive Coverage Map**: Explore predicted coverage areas on an interactive map
- **Coverage Prediction**: Make real-time predictions for specific geographic coordinates

## Installation

1. Ensure you have Python 3.8+ installed
2. Install required packages:

```
pip install django folium django-crispy-forms pandas matplotlib seaborn plotly scikit-learn
```

3. Set up the Django application:

```
python manage.py migrate
python manage.py collectstatic
```

## Running the Application

For convenience, a script is provided to set up and run the application:

```
python run_webapp.py
```

This will:
- Create necessary directories if they don't exist
- Generate model metrics if needed
- Copy plot files to the static directory
- Run migrations
- Collect static files
- Start the Django development server

## Usage

Once the server is running, access the application at:

```
http://127.0.0.1:8000/
```

Navigate through the different sections:
- Home: Overview of the application and key findings
- EDA Dashboard: Access to all exploratory data analysis visualizations
- Model Evaluation: View model performance metrics and feature importance
- Coverage Map: Interactive map showing predicted coverage areas

## Project Structure

- `coverage_viz/`: Django project settings
- `visualization/`: Main Django application
  - `views.py`: View functions for rendering templates
  - `urls.py`: URL routing
  - `templates/`: HTML templates
- `models/`: Saved machine learning models and metrics
- `plots/`: Generated visualizations
- `media/`: User-uploaded files and dynamic content
- `static/`: Static files (CSS, JS, images)
- `run_webapp.py`: Script to run the web application
- `save_model_metrics.py`: Script to extract and save model metrics

## Dependencies

- Django: Web framework
- Folium: Interactive maps
- Pandas/NumPy: Data manipulation
- Matplotlib/Seaborn: Visualization
- Scikit-learn: Machine learning
- Crispy Forms: Form styling

## Notes

- This application is designed for development/visualization purposes and is not production-ready
- For production deployment, follow Django's deployment guides and security best practices 