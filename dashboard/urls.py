from django.urls import path
from . import agri_views, views

urlpatterns = [
    path('', views.home, name='home'),
    path('products/', views.products_page, name='products_page'),
    path('news/', views.news_page, name='news_page'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('dashboard/prediction/', views.prediction_page, name='prediction_page'),
    path('report/', views.report_page, name='report_page'),
    path('api/report/pdf/', views.report_pdf, name='report_pdf'),
    path('dashboard/aws/', views.aws_page, name='aws_page'),
    path('dashboard/instance/add/', views.add_instance, name='add_instance'),
    path('api/predict-users/', views.predict_users, name='predict_users'),
    path('api/scaling-recommendation/', views.scaling_recommendation, name='scaling_recommendation'),
    path('api/agri/items/', agri_views.agri_items, name='agri_items'),
    path('api/agri/chart/', agri_views.agri_chart, name='agri_chart'),
    path('api/agri/run-forecast/', agri_views.agri_run_forecast, name='agri_run_forecast'),
    path('api/agri/reconcile/', agri_views.agri_reconcile, name='agri_reconcile'),
    path('api/agri/explain/', agri_views.agri_explain, name='agri_explain'),
    path('api/agri/analysis-image/', agri_views.agri_analysis_image, name='agri_analysis_image'),
    path('api/agri/external-fetch/', agri_views.agri_external_fetch, name='agri_external_fetch'),
]
