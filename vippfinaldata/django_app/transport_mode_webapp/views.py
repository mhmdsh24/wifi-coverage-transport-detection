from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import pandas as pd
import json
import uuid
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import traceback
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import transport mode detection
from transport_mode_detection import TransportModeDetector, detect_transport_modes
# Import the enhanced transport classifier
import sys
import os
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
try:
    from enhanced_transport_classifier import EnhancedTransportClassifier, classify_enhanced_transport_modes
except ImportError:
    # Define a fallback class for systems where the enhanced classifier isn't available
    class EnhancedTransportClassifier:
        def __init__(self, *args, **kwargs):
            pass
    def classify_enhanced_transport_modes(*args, **kwargs):
        return None, None
        
from .models import TransportModeSession, TransportModeResult, TransportModeStats
from .forms import UploadFileForm
import tempfile
from celery import shared_task

# Create a task for background processing
@shared_task
def process_transport_mode_detection(file_path, session_id, use_enhanced_transport=False, use_mobility_thresholds=False):
    """Process transport mode detection in background"""
    try:
        # Get the session
        session = TransportModeSession.objects.get(id=session_id)
        session.status = "processing"
        session.uses_enhanced_transport = use_enhanced_transport
        session.uses_mobility_thresholds = use_mobility_thresholds
        session.save()
        
        # Load data from the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Check if required columns exist
        required_cols = ['timestamp_ms', 'latitude_deg', 'longitude_deg', 'speed_mps']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            session.status = "failed"
            session.save()
            return f"Missing required columns: {', '.join(missing_cols)}"
        
        # Set up output directories
        output_dir = os.path.join(settings.MEDIA_ROOT, 'transport_mode', str(session_id))
        models_dir = os.path.join(settings.MEDIA_ROOT, 'transport_mode', str(session_id), 'models')
        plots_dir = os.path.join(settings.MEDIA_ROOT, 'transport_mode', str(session_id), 'plots')
        
        # Check if we should use enhanced transport or basic
        if use_enhanced_transport:
            # Use enhanced transport classifier
            try:
                # Prepare data for enhanced transport classifier
                # We don't have motion data in the web interface, so pass None
                enhanced_transport_df, classifier = classify_enhanced_transport_modes(
                    gps_df=df,
                    wifi_df=df,
                    motion_df=None,
                    output_dir=output_dir,
                    models_dir=models_dir,
                    load_existing=True  # Try to load existing model
                )
                
                # If we have enhanced results, use them
                if enhanced_transport_df is not None:
                    transport_df = enhanced_transport_df
                    mode_mapping = classifier.mode_mapping
                    
                    # Add predictions to transport dataframe
                    transport_df['predicted_mode_code'] = transport_df['predicted_mode_code']
                    transport_df['predicted_mode'] = transport_df['predicted_mode']
                    
                    # Process with mobility-aware thresholds
                    print("Using enhanced transport classification with mobility-aware thresholds")
                else:
                    # Fall back to basic detection
                    print("Enhanced transport classification failed, falling back to basic")
                    use_enhanced_transport = False
            except Exception as e:
                print(f"Error using enhanced transport classifier: {str(e)}")
                use_enhanced_transport = False
        
        # If enhanced transport failed or wasn't requested, use basic detection
        if not use_enhanced_transport:
            # Initialize the basic detector
            detector = TransportModeDetector(
                output_dir=output_dir,
                models_dir=models_dir,
                plots_dir=plots_dir
            )
            
            # Process the data
            X, y, transport_df = detector.prepare_features(None, None, merged_df=df)
            detector.build_pipeline()
            detector.train_model(X, y)
            
            # Evaluate model
            results = detector.evaluate_model(X, y, groups=transport_df.get('trajectory_id'))
            
            # Make predictions
            predictions, probabilities = detector.predict(X)
            
            # Add predictions to transport dataframe
            transport_df['predicted_mode_code'] = predictions
            transport_df['predicted_mode'] = transport_df['predicted_mode_code'].map(detector.mode_mapping)
            
            # Add probability columns
            for col in probabilities.columns:
                transport_df[col] = probabilities[col].values
            
            # Use basic mode mapping
            mode_mapping = detector.mode_mapping
        
        # Calculate confidence for all cases
        transport_df['confidence'] = transport_df.apply(
            lambda row: max(
                row.get('prob_still', 0),
                row.get('prob_walk', 0),
                row.get('prob_run', 0),  # Include enhanced modes
                row.get('prob_bike', 0),
                row.get('prob_car', 0),   # Include enhanced modes
                row.get('prob_bus', 0),   # Include enhanced modes
                row.get('prob_train', 0), # Include enhanced modes
                row.get('prob_subway', 0), # Include enhanced modes
                row.get('prob_vehicle', 0)
            ),
            axis=1
        )
        
        # If using mobility thresholds, add the threshold column
        if use_mobility_thresholds and use_enhanced_transport:
            # Use thresholds from the classifier
            threshold_mapping = {
                'still': -75,
                'walk': -75,
                'run': -78,
                'bike': -80,
                'car': -83,
                'bus': -83,
                'train': -87,
                'subway': -87,
                'vehicle': -83  # For backward compatibility
            }
            
            # Apply thresholds
            transport_df['rssi_threshold'] = transport_df['predicted_mode'].map(
                lambda mode: threshold_mapping.get(mode, -75)
            )
        else:
            # Use static threshold
            transport_df['rssi_threshold'] = -75
        
        # Create plots
        if use_enhanced_transport:
            # We already created plots in the classifier
            pass
        else:
            detector.create_plots(X, y, transport_df)
        
        # Save results to database
        for idx, row in transport_df.iterrows():
            result = TransportModeResult(
                session=session,
                timestamp=row['timestamp_ms'],
                latitude=row['latitude_deg'],
                longitude=row['longitude_deg'],
                speed_mps=row['speed_mps'],
                predicted_mode=row['predicted_mode'],
                confidence=row['confidence'],
                # Add all probability fields with defaults in case they're missing
                prob_still=row.get('prob_still', 0),
                prob_walk=row.get('prob_walk', 0),
                prob_run=row.get('prob_run', 0),
                prob_bike=row.get('prob_bike', 0),
                prob_car=row.get('prob_car', 0),
                prob_bus=row.get('prob_bus', 0),
                prob_train=row.get('prob_train', 0),
                prob_subway=row.get('prob_subway', 0),
                prob_vehicle=row.get('prob_vehicle', 0),
                rssi_threshold=row.get('rssi_threshold', -75),
                bearing_change=row.get('bearing_change', None)
            )
            result.save()
        
        # Calculate statistics - include all modes
        stats = {
            'still_count': (transport_df['predicted_mode'] == 'still').sum(),
            'walk_count': (transport_df['predicted_mode'] == 'walk').sum(),
            'run_count': (transport_df['predicted_mode'] == 'run').sum() if 'run' in transport_df['predicted_mode'].values else 0,
            'bike_count': (transport_df['predicted_mode'] == 'bike').sum(),
            'car_count': (transport_df['predicted_mode'] == 'car').sum() if 'car' in transport_df['predicted_mode'].values else 0,
            'bus_count': (transport_df['predicted_mode'] == 'bus').sum() if 'bus' in transport_df['predicted_mode'].values else 0,
            'train_count': (transport_df['predicted_mode'] == 'train').sum() if 'train' in transport_df['predicted_mode'].values else 0,
            'subway_count': (transport_df['predicted_mode'] == 'subway').sum() if 'subway' in transport_df['predicted_mode'].values else 0,
            'vehicle_count': (transport_df['predicted_mode'] == 'vehicle').sum(),
            'max_speed_mps': transport_df['speed_mps'].max(),
            'avg_speed_mps': transport_df['speed_mps'].mean(),
            'start_time': transport_df['timestamp_ms'].min(),
            'end_time': transport_df['timestamp_ms'].max(),
        }
        
        # Calculate anomaly reduction if using mobility thresholds
        if use_mobility_thresholds:
            # Calculate how many points would be anomalies with static threshold (-75)
            static_anomalies = (transport_df['rssi'] < -75).sum() if 'rssi' in transport_df.columns else 0
            
            # Calculate how many points are anomalies with dynamic thresholds
            dynamic_anomalies = (transport_df.apply(
                lambda row: row.get('rssi', 0) < row.get('rssi_threshold', -75), 
                axis=1
            )).sum() if 'rssi' in transport_df.columns else 0
            
            # Store anomaly reduction stats
            stats['anomalies_detected'] = static_anomalies
            stats['anomalies_reduced'] = max(0, static_anomalies - dynamic_anomalies)
        
        # Calculate total distance
        total_distance = 0
        last_lat, last_lon = None, None
        
        for idx, row in transport_df.sort_values('timestamp_ms').iterrows():
            if last_lat is not None and last_lon is not None:
                from math import radians, sin, cos, sqrt, atan2
                
                lat1, lon1 = radians(last_lat), radians(last_lon)
                lat2, lon2 = radians(row['latitude_deg']), radians(row['longitude_deg'])
                
                # Haversine formula
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                distance = 6371000 * c  # Earth radius in meters
                
                total_distance += distance
            
            last_lat, last_lon = row['latitude_deg'], row['longitude_deg']
            
        stats['total_distance_m'] = total_distance
        
        # Save statistics
        TransportModeStats.objects.create(
            session=session,
            **stats
        )
        
        # Mark session as completed
        session.status = "completed"
        session.save()
        
        return "Transport mode detection completed successfully"
    
    except Exception as e:
        try:
            session = TransportModeSession.objects.get(id=session_id)
            session.status = "failed"
            session.save()
        except:
            pass
        
        return f"Error: {str(e)}\n{traceback.format_exc()}"

def index(request):
    """Index view - shows upload form and list of sessions"""
    sessions = TransportModeSession.objects.all()
    form = UploadFileForm()
    
    return render(request, 'transport_mode/index.html', {
        'sessions': sessions,
        'form': form
    })

@csrf_exempt
def upload_file(request):
    """Handle file upload for transport mode detection"""
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded file
            uploaded_file = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(f"uploads/{uploaded_file.name}", uploaded_file)
            file_path = os.path.join(settings.MEDIA_ROOT, filename)
            
            # Get processing options from form
            use_enhanced_transport = request.POST.get('use_enhanced_transport', 'off') == 'on'
            use_mobility_thresholds = request.POST.get('use_mobility_thresholds', 'off') == 'on'
            
            # Create a new session
            session = TransportModeSession.objects.create(
                user=request.user if request.user.is_authenticated else None,
                file_name=uploaded_file.name,
                uses_enhanced_transport=use_enhanced_transport,
                uses_mobility_thresholds=use_mobility_thresholds
            )
            
            # Start background task with processing options
            process_transport_mode_detection.delay(
                file_path, 
                session.id,
                use_enhanced_transport=use_enhanced_transport,
                use_mobility_thresholds=use_mobility_thresholds
            )
            
            return redirect('transport_mode_session', session_id=session.id)
    else:
        form = UploadFileForm()
    
    return render(request, 'transport_mode/upload.html', {'form': form})

def session_detail(request, session_id):
    """Show details of a transport mode detection session"""
    session = get_object_or_404(TransportModeSession, id=session_id)
    
    # Get results
    results = TransportModeResult.objects.filter(session=session)
    
    # Get statistics
    try:
        stats = session.stats
    except TransportModeStats.DoesNotExist:
        stats = None
    
    # Create plots
    plots_html = {}
    
    if results.exists() and session.status == "completed":
        # Convert to dataframe
        results_df = pd.DataFrame(list(results.values()))
        
        # Create mode distribution plot
        fig_dist = px.pie(
            results_df, 
            names='predicted_mode', 
            title='Transport Mode Distribution',
            color='predicted_mode',
            color_discrete_map={
                'still': '#3366CC',
                'walk': '#33CC66',
                'bike': '#CC6633',
                'vehicle': '#CC3366'
            }
        )
        plots_html['mode_distribution'] = plot(fig_dist, output_type='div')
        
        # Create speed by mode plot
        fig_speed = px.box(
            results_df, 
            x='predicted_mode', 
            y='speed_mps',
            title='Speed Distribution by Transport Mode', 
            color='predicted_mode',
            color_discrete_map={
                'still': '#3366CC',
                'walk': '#33CC66',
                'bike': '#CC6633',
                'vehicle': '#CC3366'
            }
        )
        plots_html['speed_by_mode'] = plot(fig_speed, output_type='div')
        
        # Create map
        fig_map = px.scatter_mapbox(
            results_df,
            lat='latitude',
            lon='longitude',
            color='predicted_mode',
            color_discrete_map={
                'still': '#3366CC',
                'walk': '#33CC66',
                'bike': '#CC6633',
                'vehicle': '#CC3366'
            },
            hover_data=['speed_mps', 'confidence', 'timestamp'],
            zoom=12,
            title='Transport Modes Map'
        )
        fig_map.update_layout(mapbox_style="open-street-map")
        plots_html['map'] = plot(fig_map, output_type='div')
    
    return render(request, 'transport_mode/session_detail.html', {
        'session': session,
        'results': results[:100],  # Limit to 100 results for page load performance
        'stats': stats,
        'plots': plots_html
    })

def session_data(request, session_id):
    """Return session data as JSON for AJAX requests"""
    session = get_object_or_404(TransportModeSession, id=session_id)
    
    # Get results
    results = TransportModeResult.objects.filter(session=session).values(
        'timestamp', 'latitude', 'longitude', 'speed_mps', 
        'predicted_mode', 'confidence', 'prob_still', 'prob_walk', 
        'prob_bike', 'prob_vehicle'
    )
    
    # Return JSON response
    return JsonResponse({
        'session': {
            'id': str(session.id),
            'status': session.status,
            'created_at': session.created_at.isoformat(),
            'file_name': session.file_name
        },
        'results': list(results)
    })

def download_results(request, session_id):
    """Download session results as CSV"""
    session = get_object_or_404(TransportModeSession, id=session_id)
    
    # Get results
    results = TransportModeResult.objects.filter(session=session).values(
        'timestamp', 'latitude', 'longitude', 'speed_mps', 
        'predicted_mode', 'confidence', 'prob_still', 'prob_walk', 
        'prob_bike', 'prob_vehicle', 'bearing_change'
    )
    
    # Convert to dataframe
    df = pd.DataFrame(list(results))
    
    # Create response
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="transport_modes_{session_id}.csv"'
    
    # Write CSV
    df.to_csv(path_or_buf=response, index=False)
    
    return response 