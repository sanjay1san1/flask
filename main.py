from flask import Flask,render_template,request
import numpy as np 
import joblib
app = Flask(__name__) # initialize the app
@app.route('/')
def home():
    return render_template('diabatics.html') # this code will redirect to diabatics.html page 
#when the user will submit the form we will take the data from frontend and store into backend using below code 
@app.route('/predict',methods=['post'])
def predict():
    Pregnancies=int(request.form.get("Pregnancies") )  # here we are doing type casting 
    Glucose=int(request.form.get("Glucose") )
    BloodPressure=int(request.form.get("BloodPressure") )
    SkinThickness=int(request.form.get("SkinThickness") )
    Insulin=int(request.form.get("Insulin") )
    BMI=int(request.form.get("BMI") )
    DiabetesPedigreeFunction=int(request.form.get("DiabetesPedigreeFunction") )
    Age=int(request.form.get("Age") )
    # print(Pregnancies,Glucose,BloodPressure)
    #convert input into array 
    input_data=np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    # load the model 
    model = joblib.load('model_joblib.pkl')
    prediction=model.predict(input_data)
    print(f'Model prediction = {prediction}')
    if prediction == 1:
        return "Person is Diabatic patient"
    else:
        return "Person is not a Diabatic patient" 
    return "Form submitted successfully"
app.run(debug=True)