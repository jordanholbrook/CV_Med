from .models import *
from django import forms

class ImageClassificationForm(forms.ModelForm):
    class Meta:
        model = ImageClassification
        fields = ['image','type']

class DemogrphicScoreForm(forms.ModelForm):
    class Meta:
        model = DemogrphicScore
        fields = ['age', 'zip', 'gender']