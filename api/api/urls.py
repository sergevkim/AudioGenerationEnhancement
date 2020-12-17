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
from rest_framework.parsers import *
from rest_framework.status import *

import agenh
import numpy as np
import io
import torch


class PlainTextParser(BaseParser):
    """
    Plain text parser.
    """
    media_type = 'text/plain'

    def parse(self, stream, media_type=None, parser_context=None):
        """
        Simply return a string representing the body of the request.
        """
        return stream.read()


def tensor_to_bytes(tensor):
    np_array = tensor.cpu().detach().numpy()
    buffer = io.BytesIO(np_array.tobytes())
    return buffer.read()


def bytes_to_tensor(bytes):
    np_array = np.frombuffer(bytes, dtype=np.int32)
    tensor = torch.tensor(np_array)
    return tensor


# from agenh.models.dummy_autoencoder import asd


class GeneratorView(APIView):
    parser_classes = (JSONParser,)

    def post(self, request: Request):
        seed = 42
        if 'seed' in request.data:
            seed = int(request.data.get('seed'))

        wav = torch.tensor([1, 2, 3], dtype=torch.int32)  # TODO call some generator
        wav_bytes = tensor_to_bytes(wav)

        return Response(wav_bytes,
                        HTTP_403_FORBIDDEN)


class EnhancerView(APIView):
    parser_classes = (PlainTextParser,)

    def post(self, request: Request):
        wav_bytes = request.data
        wav = bytes_to_tensor(wav_bytes)

        enhanced_wav = wav  # TODO call enhance model
        wav_bytes = tensor_to_bytes(enhanced_wav)

        return Response(wav_bytes,
                        HTTP_403_FORBIDDEN)


urlpatterns = [
    path('admin/', admin.site.urls),
    path('generate', GeneratorView.as_view()),
    path('enhance', EnhancerView.as_view()),
]
