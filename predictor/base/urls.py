from django.urls import path
from . import views
urlpatterns=[
    path('',views.result),
    path('predict',views.result)
      ]