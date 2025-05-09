from django.urls import path
from . import views

urlpatterns = [
    path('fixed_eda_steps/', views.fixed_eda_steps_view, name='fixed_eda_steps'),
] 