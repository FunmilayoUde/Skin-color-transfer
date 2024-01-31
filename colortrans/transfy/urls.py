from django.urls import path
from . import views

urlpatterns = [
    
    path('', views.home_page, name= 'home_page'),
    path('process/', views.process_view, name = 'process_view'),
]