from flask import Flask, request, render_template
import joblib
import pandas as pd
from model import fraud_model 

app = Flask(__name__)

model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')  # Render your HTML file

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            data = pd.read_csv(uploaded_file)
            X = data.drop('Class', axis=1)
            y = data['Class']
            X = scaler.transform(X)
            prediction = model.predict(X)
            return {'prediction': prediction.tolist()}
    return 'Invalid Request'

if __name__ == '__main__':
    app.run(debug=True)
