from django.http import HttpResponse
from django.shortcuts import render
import joblib
def home(request):
    return render(request, 'home.html')

def resultt(request):
    cls = joblib.load('final.sav')
    lis=[]
    lis.append(request.GET['Glucose'])
    lis.append(request.GET['Insulin'])
    lis.append(request.GET['BMI'])
    lis.append(request.GET['Age'])
    
    print(lis)
    
    ans= cls.predict([lis])
    return render(request, 'resultt.html',{'ans':ans})