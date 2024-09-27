from rest_framework import serializers
from .models import ImageClassification

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageClassification
        fields = ('id', 'image', 'result')