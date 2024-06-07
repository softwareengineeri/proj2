
from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from twilio.rest import Client
from django.conf import settings
import os

# Create your views here.
@login_required(login_url='login')

def index(request):
    return render(request, 'index.html')

def SignupPage(request):
    if request.method=='POST':
        uname=request.POST.get('username')
        email=request.POST.get('email')
        pass1=request.POST.get('password1')
        pass2=request.POST.get('password2')
        phone = request.POST.get('phone')

        if pass1!=pass2:
            return HttpResponse("Your password and confrom password are not Same!!")
        else:

            my_user=User.objects.create_user(uname,email,pass1)
            my_user.phone_number = phone
            my_user.save()
            return redirect('login')
    return render (request,'signup.html')


def LoginPage(request):
    if request.method=='POST':
        username=request.POST.get('username')
        pass1=request.POST.get('pass')
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
            login(request,user)
            return redirect('index')
        else:
            return HttpResponse ("Username or Password is incorrect!!!")

    return render (request,'login.html')

def LogoutPage(request):
    logout(request)
    return redirect('login')

def detection_result(request):
    result = request.POST.get('result')
    uploaded_image_url = request.POST.get('uploaded_image_url')
    sms_status = send_defective_tire_sms(result)  # Send SMS and get status
    print("sms_status:", sms_status) 
    return render(request, 'prediction_result.html', {'result': result, 'uploaded_image_url': uploaded_image_url, 'sms_status': sms_status})

def send_defective_tire_sms(result):
    if result == 'Defective tire':
        phone_number = '+919493317267'  # Replace with the recipient's phone number
        message = "Alert: Defective tire detected. Please check your vehicle."
        if send_sms(phone_number, message):
            return "SMS sent successfully"
        else:
            return "Failed to send SMS"
    else:
        return "No SMS sent (Tire is normal)"

def send_sms(recipient_number, message_body):
    # Twilio credentials
    account_sid = 'ACa6a55c077efb4aa44d031a70212d2e1a'
    auth_token = '44bf437e6334d07af65433bb3c6a81ab'
    twilio_number = '+12056971113'

    # Initialize Twilio client
    client = Client(account_sid, auth_token)

    try:
        # Send SMS message
        message = client.messages.create(
            body=message_body,
            from_=twilio_number,
            to=recipient_number
        )
        print(f"SMS sent successfully. SID: {message.sid}")
        return True
    except Exception as e:
        print(f"Failed to send SMS: {str(e)}")
        return False


def aboutUs(request):
    return render(request, 'aboutus.html')
def Tips(request):
    return render(request, 'tips.html')

def predictImage(request):
    if request.method == 'POST' and request.FILES['filepath']:
        # Load the trained model
        model = load_model('D:\\django project\\Tiresonhighways\\models\\tire_defect_detection_model.h5')

        # Get the uploaded image
        uploaded_image = request.FILES['filepath']

        # Preprocess the image
        processed_image = preprocess_image(uploaded_image)

        # Make predictions
        prediction = model.predict(processed_image)
        #print("Prediction Probability:", prediction)
        # Interpret the prediction
        if prediction > 0.5:
            result = "Defective tire"
        else:
            result = "Normal tire"
        
        # Send SMS and get status
        sms_status = send_defective_tire_sms(result)

        uploaded_image_url = settings.MEDIA_URL + uploaded_image.name

        # Render the result on a webpage
        return render(request, 'prediction_result.html', {'uploaded_image_url': uploaded_image_url, 'result': result, 'sms_status': sms_status})

    else:
        return HttpResponse("Error: No image uploaded.")
    
def preprocess_image(image):
    try:
        # Read the image using OpenCV
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)  # Read the image in color

        # Resize the image to match the input shape expected by the model
        resized_image = cv2.resize(img, (128, 128))

        # Convert RGB image to grayscale
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Expand the dimensions to match the input shape expected by the model
        processed_image = np.expand_dims(grayscale_image, axis=0)
        processed_image = np.expand_dims(processed_image, axis=-1)  # Use axis=-1 to expand along the last dimension

        # Normalize pixel values
        processed_image = processed_image / 255.0
        
        return processed_image
    
    except Exception as e:
        # Handle any errors during image preprocessing
        print("Error preprocessing image:", e)
        return None


