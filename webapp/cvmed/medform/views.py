from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .serializers import ImageSerializer
from .models import ImageClassification

def medcv(request):
  serializer_class = ImageSerializer
  model_query = ImageClassification.objects.all()
  template = loader.get_template('index.html')
  return HttpResponse(template.render())
