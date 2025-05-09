#!/bin/bash

# Navigate to the project directory
cd vippfinaldata

# Apply database migrations
echo "Applying database migrations..."
python manage.py migrate

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Start the development server
echo "Starting the development server..."
python manage.py runserver 0.0.0.0:8000 