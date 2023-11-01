from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
from azureml.core import Workspace
from azureml.core.model import Model

# Connect to Azure ML Workspace
ws = Workspace(subscription_id='f998d2c3-d89e-48ea-9975-322499356122', resource_group='Nedamco-Acadamy-Phase2-ML', workspace_name='MLSharedWorkspace')

# Load the dataset directly from Azure ML Studio
datastore = ws.datastores['workspaceblobstore']
data = pd.read_csv(datastore.path('UI/2023-11-01_091229_UTC/creditcard.csv'))

# Preprocess the data
X = data.drop('Class', axis=1)
y = data['Class']
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model to Azure ML Studio
model_name = 'fraud_detection_model'
model_path = 'model'
model.upload(src_dir='model', model_name=model_name)  # Upload the model to Azure ML

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
