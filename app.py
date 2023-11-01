from flask import Flask, request, render_template
import joblib
import pandas as pd
from azureml.core import Workspace
from azureml.core.model import Model

app = Flask(__name__)

# Connect to Azure ML Workspace
ws = Workspace(subscription_id='f998d2c3-d89e-48ea-9975-322499356122', resource_group='Nedamco-Acadamy-Phase2-ML', workspace_name='MLSharedWorkspace')

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
            
            # Fetch the model from Azure ML Studio
            model_path = 'azureml://locations/eastus/workspaces/MLSharedWorkspace/models/fraud_model/versions/1'
            model = Model(ws, 'fraud_model', version=1, workspace=ws)
            
            # Fetch the scaler from Azure ML Studio
            scaler_path = 'azureml://subscriptions/f998d2c3-d89e-48ea-9975-322499356122/resourcegroups/Nedamco-Acadamy-Phase2-ML/workspaces/MLSharedWorkspace/datastores/workspaceblobstore/paths/UI/2023-11-01_091229_UTC/creditcard.csv'
            scaler = joblib.load(scaler_path)
            
            X = scaler.transform(X)
            prediction = model.predict(X)
            return {'prediction': prediction.tolist()}
    return 'Invalid Request'

if __name__ == '__main__':
    app.run(debug=True)
