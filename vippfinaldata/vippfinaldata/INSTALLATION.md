# WiFi Coverage Analysis Visualization Setup

This guide will help you set up the WiFi Coverage Analysis visualization system.

## Prerequisites

- Python 3.8 or higher
- Django 3.2 or higher
- Pandas, NumPy, Matplotlib, Seaborn for data processing
- Required data files (WiFi, location, and GPS data)

## Installation Steps

1. **Clone the repository**

```bash
git clone <repository-url>
cd vippfinaldata
```

2. **Install required packages**

```bash
pip install -r requirements.txt
```

3. **Collect static files**

```bash
python manage.py collectstatic
```

4. **Run database migrations**

```bash
python manage.py migrate
```

5. **Start the development server**

```bash
python manage.py runserver
```

6. **Access the visualization**

Open your browser and go to: http://127.0.0.1:8000/

## Directory Structure

- `visualization/` - Django app for the visualization system
  - `static/` - Static files (CSS, images)
    - `css/` - CSS stylesheets
    - `coverage_maps/` - Coverage visualization images
    - `anomaly_detection/` - Anomaly detection images
  - `templates/` - HTML templates
  - `views.py` - View functions
  - `urls.py` - URL routing

## Running the Pipeline

1. Navigate to the "Run Pipeline" page
2. Select your data files
3. Set the RSSI threshold (default: -75 dBm)
4. Click "Run Pipeline"
5. Wait for processing to complete
6. View results on the various visualization pages

## Troubleshooting

- If images are not displaying, ensure you have run `python manage.py collectstatic`
- Check that all required Python packages are installed
- Verify that your data files have the expected columns and formats

## Contact

For more information, contact the developer: Rami Eid 