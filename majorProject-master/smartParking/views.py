
from django.shortcuts import render

def index(request):

    return render(request, 'index.html')
def contact(request):

    return render(request, 'contact.html')
def home(request):

    return render(request, 'home.html')
def map(request):

    return render(request, 'map.html')
