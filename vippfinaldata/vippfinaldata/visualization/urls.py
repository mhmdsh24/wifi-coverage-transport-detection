from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('run_pipeline/', views.run_pipeline, name='run_pipeline'),
    path('pipeline_status/', views.pipeline_status_view, name='pipeline_status'),
    path('view_results/', views.view_results, name='view_results'),
    path('eda_steps/', views.eda_steps_view, name='eda_steps'),
    path('model_detail/', views.model_detail_view, name='model_detail'),
    path('coverage_map/', views.coverage_map_view, name='coverage_map'),
    path('anomaly_detection/', views.anomaly_detection_view, name='anomaly_detection'),
    path('mobility_threshold/', views.mobility_aware_threshold_view, name='mobility_threshold'),
    path('human_flow_mapping/', views.human_flow_mapping_view, name='human_flow_mapping'),
    path('test_template/', views.test_template_view, name='test_template'),
    path('test_simple/', views.test_simple_view, name='test_simple'),
    path('test_minimal/', views.test_minimal_view, name='test_minimal'),
    path('simplified_eda/', views.simplified_eda_view, name='simplified_eda'),
] 