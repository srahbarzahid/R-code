from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),

    # Category pages
    path('classification/', views.classification, name='classification'),
    path('regression/', views.regression, name='regression'),
    path('clustering/', views.clustering, name='clustering'),

    # Regression (core)
    path('regression/linear_regression/', views.linear_regression, name='linear_regression'),
    path('regression/random_forest/', views.random_forest_regression, name='random_forest_regression'),

    # Classification (core)
    path('classification/logistic_regression/', views.logistic_regression, name='logistic_regression'),
    path('classification/svm/', views.svm, name='svm'),
    path('classification/random_forest/', views.random_forest, name='random_forest'),

    # Clustering (core)
    path('clustering/kmeans/', views.kmeans, name='kmeans'),
    path('clustering/hierarchical_clustering/', views.hierarchical_clustering, name='hierarchical_clustering'),

    # Other pages / utilities
    path('samples/', views.samples, name='samples'),
    path('about/', views.about, name='about'),
    path('download-model/', views.download_model, name='download_model'),
    path('test/', views.test, name='test'),
    path('predict', views.predict, name='predict'),

    # Dataset storage + preprocessing
    path('save_file/', views.save_file, name='save_file'),
    path('get_file/', views.get_file, name='get_file'),
    path('clear_file/', views.clear_file, name='clear_file'),
    path("get_cluster_plot/", views.get_cluster_plot, name="get_cluster_plot"),
    path("get_scatter_plot/", views.get_scatter_plot, name="get_scatter_plot"),
    path('preprocessing', views.preprocessing, name='preprocessing'),
    path('preprocessing/fill-missing/', views.fill_missing_values, name='fill_missing'),
    path('preprocessing/encoding/', views.encoding, name='encoding'),
    path('preprocessing/scaling/', views.scaling, name='scaling'),
    path('download_csv/', views.download_csv, name='download_csv'),
    path('preprocessing/scaling/data_details/', views.data_details, name='data_details'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
