from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from .forms import *
from .models import ImageClassification

def medcv(request):
  form = ImageClassificationForm(request.POST, request.FILES)
  if form.is_valid():
    print(form.cleaned_data)
    ### Load Model and Get a Label
    ### Pass that record to the next page
    
    form.save()
    return redirect('demographics', form.pk)
  else:
    print(form.errors)
  
  return render(request, template_name='index.html')

def demographic(request, pk):
  model = ImageClassification.objects.get(pk)
  form = DemogrphicScoreForm(request.POST, request.FILES)
  if form.is_valid():
    ### Load Model choice with Label
    ### Obtain Demogrphic information
    ### Run the regression and get the output
    
    form.save()
    return redirect('output')
  
  return render(request, template_name='demographic.html')

def output(request):
  form = DemogrphicScoreForm(request.POST, request.FILES)
  
  return render(request, template_name='output.html')

