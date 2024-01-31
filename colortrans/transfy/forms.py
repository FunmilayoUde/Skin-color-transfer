from django import forms
from django.forms.widgets import ClearableFileInput


class ImageUploadForm(forms.Form):
    source_image = forms.ImageField(widget=ClearableFileInput())
    target_image = forms.ImageField(widget=ClearableFileInput())
