from django.urls import path
from . import views

urlpatterns = [
    path('', views.medcv, name='medcv'),
]