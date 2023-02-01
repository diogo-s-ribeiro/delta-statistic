from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('calculate/', views.calculate, name='calculate'),
    path('calculate/result', views.result, name='result'),
    path('help/', views.help, name='help'),

    path('help/ps/', views.h_ps, name='h_ps'),
    path('help/ace/', views.h_ace, name='h_ace'),
    path('help/files/', views.h_files, name='h_files'),
    path('help/model/', views.h_model, name='h_model'),
    path('help/entropy/', views.h_entropy, name='h_entropy'),
    
    path('delta_txt', views.delta_txt, name='delta_txt'),
    path('delta_txt_metadata', views.delta_txt_metadata, name='delta_txt_metadata'),
    path('delta_csv', views.delta_csv, name='delta_csv'),
    path('delta_csv_metadata', views.delta_csv_metadata, name='delta_csv_metadata'),
]