from django.urls import path
from rest_framework.routers import DefaultRouter
from . import views

urlpatterns = [
    path('', views.medcv, name='medcv'),
    path('demographic/<int:pk>', views.demographic, name='demographic'),
    path('output/<int:pk>', views.output, name='output')
]