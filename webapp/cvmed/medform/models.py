from django.db import models

# Create your models here.

GENDER_CHOICES = (
    ('M','M'),
    ('F', 'F')
)

MODEL_CHOICES = (
    ('xray','xray'),
    ('dermatology','dermatology')
)

class ImageClassification(models.Model):
    image = models.ImageField(upload_to='images')
    label = models.CharField(max_length=20)
    type = models.CharField(choices=MODEL_CHOICES, max_length=20)
    updated = models.DateField(auto_now=True)
    created = models.DateField(auto_now_add=True)

class DemogrphicScore(models.Model):
    model = models.ForeignKey(ImageClassification, on_delete=models.CASCADE)
    age = models.IntegerField(max_length=2)
    gender = models.CharField(choices=GENDER_CHOICES, max_length=6)
    zip = models.IntegerField(max_length=5)
    score = models.FloatField()