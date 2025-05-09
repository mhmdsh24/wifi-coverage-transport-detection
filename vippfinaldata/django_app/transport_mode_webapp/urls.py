from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='transport_mode_index'),
    path('upload/', views.upload_file, name='transport_mode_upload'),
    path('session/<uuid:session_id>/', views.session_detail, name='transport_mode_session'),
    path('session/<uuid:session_id>/data/', views.session_data, name='transport_mode_session_data'),
    path('session/<uuid:session_id>/download/', views.download_results, name='transport_mode_download'),
] 