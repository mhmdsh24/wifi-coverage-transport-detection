# Navigate to the project directory
cd vippfinaldata

# Apply database migrations
Write-Output "Applying database migrations..."
python manage.py migrate

# Collect static files
Write-Output "Collecting static files..."
python manage.py collectstatic --noinput

# Start the development server
Write-Output "Starting the development server..."
python manage.py runserver 0.0.0.0:8000 