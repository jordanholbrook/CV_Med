from django.db import models

# Create your models here.

GENDER_CHOICES = (
    ('Male','Male'),
    ('Female', 'Female')
)

MODEL_CHOICES = (
    ('pneumonia','pneumonia'),
    ('dermatology','dermatology')
)

class ImageClassification(models.Model):
    image = models.ImageField(upload_to='images')
    label = models.CharField(max_length=20)
    type = models.CharField(choices=MODEL_CHOICES, max_length=20)
    updated = models.DateField(auto_now=True)
    created = models.DateField(auto_now_add=True)

class DemogrphicScore(models.Model):
    image = models.ForeignKey(ImageClassification, on_delete=models.CASCADE)
    age = models.IntegerField(max_length=2)
    gender = models.CharField(choices=GENDER_CHOICES, max_length=6)
    zip = models.IntegerField(max_length=5)