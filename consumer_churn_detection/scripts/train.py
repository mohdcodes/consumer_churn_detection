# scripts/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess data
def preprocess_data(data):
    X = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
    X = pd.get_dummies(X, drop_first=True)  # Handle categorical variables
    y = data['Exited']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

# Save model and scaler
def save_model(model, scaler, model_path, scaler_path):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

if __name__ == "__main__":
    data = load_data('../data/Churn_Modelling.csv')
    X_train, X_val, y_train, y_val = preprocess_data(data)
    model, scaler = train_model(X_train, y_train)
    save_model(model, scaler, '../models/churn_model.pkl', '../models/scaler.pkl')
