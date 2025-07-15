from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load('models/diabetes_model.joblib')
scaler = joblib.load('models/scaler.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        # Extract values from form and convert to float
        features = [
            float(request.form[field]) for field in
            ["Pregnancies","Glucose","BloodPressure","SkinThickness",
             "Insulin","BMI","DiabetesPedigreeFunction","Age"]
        ]
        # Scale and predict
        scaled = scaler.transform([features])
        probability = model.predict_proba(scaled)[0][1]
        result = f"{probability * 100:.1f}% chance of diabetes"
    return render_template('index.html', result=result)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

