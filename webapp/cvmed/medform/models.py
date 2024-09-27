from django.db import models

# Create your models here.

class ImageClassification(models.Model):
    image = models.ImageField(upload_to='images')
    result = models.CharField(max_length=2, blank=True)
    updated = models.DateField(auto_now=True)
    created = models.DateField(auto_now_add=True)

    def __str__(self):
        return str(self.id)
    
    # def save(self, *args, **kwargs):
    #     return str(self.id)