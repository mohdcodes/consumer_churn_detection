# scripts/evaluate.py
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Evaluate model
def evaluate_model(model, scaler, X_val, y_val):
    X_val_scaled = scaler.transform(X_val)
    y_pred = model.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    return accuracy, report

if __name__ == "__main__":
    data = load_data('../data/Churn_Modelling.csv')
    X_train, X_val, y_train, y_val = train_test_split(data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1), 
                                                      data['Exited'], test_size=0.2, random_state=42)
    X_val = pd.get_dummies(X_val, drop_first=True)  # Ensure the same encoding as training set
    
    model = joblib.load('../models/churn_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    accuracy, report = evaluate_model(model, scaler, X_val, y_val)
    
    print(f'Model Accuracy: {accuracy:.2f}')
    print(f'Classification Report:\n{report}')
