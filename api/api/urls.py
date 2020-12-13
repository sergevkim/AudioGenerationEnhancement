"""api URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import JSONParser
from rest_framework.status import *

import agenh
#from agenh.models.dummy_autoencoder import asd

class GeneratorView(APIView):
    parser_classes = (JSONParser,)

    def post(self, request: Request):
        asd()
        return Response({"message": "Wrong token or not provided or not enough privileges"}, HTTP_401_UNAUTHORIZED)

class EnhancerView(APIView):
    parser_classes = (JSONParser,)

    def post(self, request: Request):
        asd()
        return Response({"message": "Also wrong token or not provided or not enough privileges"}, HTTP_401_UNAUTHORIZED)


urlpatterns = [
    path('admin/', admin.site.urls),
    path('generate', GeneratorView.as_view()),
    path('enhance', EnhancerView.as_view()),
]
