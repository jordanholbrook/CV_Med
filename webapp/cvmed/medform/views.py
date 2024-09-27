from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from .forms import *
from .models import ImageClassification
from PIL import Image
import pandas as pd
import joblib
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from model.modelfiles import predict_cost_model, predict_derma_model, predict_xray_model

def medcv(request):
  form = ImageClassificationForm(request.POST, request.FILES)
  if request.method == 'POST' and request.FILES['image']:
    if form.is_valid():
      if form.cleaned_data.get('type') == 'xray':
        instance = form.save(commit=False)
        myfile = request.FILES['image']
        preprocessed_img = predict_xray_model.preprocess_image(myfile)
        n_channels = 1
        n_classes = 2
        task = 'binary-class'
        loaded_model = predict_xray_model.Net(in_channels=n_channels, num_classes=n_classes)  # Initialize the model architecture
        loaded_model.load_state_dict(torch.load('./model/modelfiles/xray_model.pth'))  # Load the saved parameters
        loaded_model.eval()
        loaded_model.load_state_dict(torch.load('./model/modelfiles/xray_model.pth'))
        # Make prediction
        with torch.no_grad():
            output = loaded_model(preprocessed_img)  # Get model output
            _, predicted_class = torch.max(output, 1)  # Get predicted class

        # Print the predicted class
        print(f"Predicted class: {predicted_class.item()}")
        instance.label = predicted_class.item()
        instance.type = 'xray'
        instance.save()
      
      elif form.cleaned_data.get('type') == 'dermatology':
        instance = form.save(commit=False)
        myfile = request.FILES['image']
        preprocessed_img = predict_derma_model.preprocess_image(myfile)
        n_channels = 3
        n_classes = 7
        task = 'multi-class'
        loaded_model = predict_derma_model.Net(in_channels=n_channels, num_classes=n_classes)  # Initialize the model architecture
        loaded_model.load_state_dict(torch.load('./model/modelfiles/derma_model.pth'))  # Load the saved parameters
        loaded_model.eval()
        loaded_model.load_state_dict(torch.load('./model/modelfiles/derma_model.pth'))
        # Make prediction
        with torch.no_grad():
            output = loaded_model(preprocessed_img)  # Get model output
            _, predicted_class = torch.max(output, 1)  # Get predicted class

        # Print the predicted class
        print(f"Predicted class: {predicted_class.item()}")
        instance.label = predicted_class.item()
        instance.type = 'dermatology'
        instance.save()

      else:
        return render(request, template_name='index.html')
  
      return redirect('demographic', instance.pk)
  else:
    print(form.errors)
  
  return render(request, template_name='index.html')

def demographic(request, pk):
  model = ImageClassification.objects.get(id=pk)
  form = DemogrphicScoreForm(request.POST, request.FILES, instance=model)
  context = {'demographic': model}
  if request.method == 'POST':
    if form.is_valid():
      print(form.cleaned_data)
      instance2 = form.save(commit=False)
      instance2.age = form['age'].value()
      instance2.gender = form['gender'].value()
      instance2.zip = form['zip'].value()
      if model.type == 'xray':
        image_type = 'Xray'
        image_label = 'Normal Lungs' if model.label==0 else 'pneumonia'
      else:
        image_type = "Skin"
        image_label = 'melanoma'

      input_data = pd.DataFrame({
          'Age': instance2.age,
          'Sex': instance2.gender,
          'Zipcode': instance2.zip,
          'Image_Type': [image_type],
          'Diagnosis': [image_label]
      })

      loaded_model = joblib.load('./model/modelfiles/medical_cost_predictor_model_v3.pkl')
      print("Model loaded successfully.")

      # Make predictions with the loaded model
      predicted_cost = loaded_model.predict(input_data)
      print(f"Predicted medical cost: {predicted_cost[0]}")
      instance2.score = predicted_cost[0]
      print(f"instance2: {instance2}")
      instance2.save()

      return redirect('output', instance2.pk)

    else:
      print(form.errors)
  
  return render(request, template_name='demographic.html', context=context)

def output(request, pk):
  model = ImageClassification.objects.get(id=pk)
  context = {'model':model}
    
  return render(request, template_name='output.html', context=context)

