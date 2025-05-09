from django.db import models
from django.contrib.auth.models import User
import uuid

class TransportModeSession(models.Model):
    """Model for transport mode detection session"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    file_name = models.CharField(max_length=255, blank=True, null=True)
    status = models.CharField(max_length=20, default="pending", 
                             choices=[
                                 ("pending", "Pending"),
                                 ("processing", "Processing"),
                                 ("completed", "Completed"),
                                 ("failed", "Failed")
                             ])
    uses_enhanced_transport = models.BooleanField(default=False, help_text="Whether this session uses enhanced transport classification")
    uses_mobility_thresholds = models.BooleanField(default=False, help_text="Whether this session uses mobility-aware RSSI thresholds")
    
    def __str__(self):
        return f"Session {self.id} - {self.status}"
    
    class Meta:
        ordering = ['-created_at']

class TransportModeResult(models.Model):
    """Model for transport mode detection results"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(TransportModeSession, on_delete=models.CASCADE, related_name="results")
    timestamp = models.BigIntegerField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    speed_mps = models.FloatField()
    predicted_mode = models.CharField(max_length=20)
    confidence = models.FloatField()
    prob_still = models.FloatField(default=0.0)
    prob_walk = models.FloatField(default=0.0)
    prob_bike = models.FloatField(default=0.0)
    prob_vehicle = models.FloatField(default=0.0)
    prob_run = models.FloatField(default=0.0)
    prob_car = models.FloatField(default=0.0)
    prob_bus = models.FloatField(default=0.0)
    prob_train = models.FloatField(default=0.0)
    prob_subway = models.FloatField(default=0.0)
    rssi_threshold = models.FloatField(default=-75.0, help_text="Dynamic RSSI threshold based on transport mode")
    bearing_change = models.FloatField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.predicted_mode} at {self.timestamp}"
    
    class Meta:
        ordering = ['timestamp']

class TransportModeStats(models.Model):
    """Model for aggregated transport mode statistics"""
    session = models.OneToOneField(TransportModeSession, on_delete=models.CASCADE, related_name="stats")
    still_count = models.IntegerField(default=0)
    walk_count = models.IntegerField(default=0)
    bike_count = models.IntegerField(default=0)
    vehicle_count = models.IntegerField(default=0)
    run_count = models.IntegerField(default=0)
    car_count = models.IntegerField(default=0)
    bus_count = models.IntegerField(default=0)
    train_count = models.IntegerField(default=0)
    subway_count = models.IntegerField(default=0)
    anomalies_detected = models.IntegerField(default=0)
    anomalies_reduced = models.IntegerField(default=0)
    total_distance_m = models.FloatField(default=0.0)
    max_speed_mps = models.FloatField(default=0.0)
    avg_speed_mps = models.FloatField(default=0.0)
    start_time = models.BigIntegerField(null=True, blank=True)
    end_time = models.BigIntegerField(null=True, blank=True)
    
    def __str__(self):
        return f"Stats for session {self.session.id}" 