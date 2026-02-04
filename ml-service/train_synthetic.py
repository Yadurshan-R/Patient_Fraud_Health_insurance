import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
import pickle
import os
from pathlib import Path

# Configuration
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic insurance claim data"""
    np.random.seed(42)
    
    # Generate features
    amounts = np.abs(np.random.normal(loc=5000, scale=3000, size=n_samples))
    ages = np.random.randint(18, 90, size=n_samples)
    genders = np.random.choice(['Male', 'Female'], size=n_samples)
    diagnoses = np.random.choice([
        'Pregnancy', 'Hypertension', 'Diabetes', 'Pneumonia',
        'Gastroenteritis', 'Cesarean Section', 'Cataract Surgery', 'Other'
    ], size=n_samples)
    stay_duration = np.random.randint(1, 15, size=n_samples)
    
    data = pd.DataFrame({
        'Amount Billed': amounts,
        'Age': ages,
        'Gender': genders,
        'Diagnosis': diagnoses,
        'StayDuration': stay_duration
    })
    
    # Generate Target (Fraud Logic)
    # High amounts + low age + 'Other' diagnosis = higher fraud probability
    # Just a simple heuristic for demonstration
    fraud_prob = (
        (data['Amount Billed'] > 8000).astype(int) * 0.3 + 
        (data['Age'] < 30).astype(int) * 0.1 +
        (data['Diagnosis'] == 'Other').astype(int) * 0.2
    )
    # Add random noise
    fraud_prob += np.random.uniform(0, 0.2, size=n_samples)
    
    data['Fraud'] = (fraud_prob > 0.5).astype(int)
    
    return data

def train_and_save():
    """Train model and save artifacts"""
    print("🔄 Generating synthetic data...")
    df = generate_synthetic_data(1000)
    
    # Preprocessing
    X = df.drop('Fraud', axis=1)
    y = df['Fraud']
    
    # One-Hot Encoding
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_names = X_encoded.columns.tolist()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Scale (though Tree models don't strictly need it, we keep it for pipeline consistency)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    print("🚀 Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"✅ Training Complete!")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    
    # Save Artifacts
    print(f"💾 Saving artifacts to {MODELS_DIR}...")
    
    with open(MODELS_DIR / 'best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    with open(MODELS_DIR / 'feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
        
    metadata = {
        'model_name': 'Gradient Boosting (Synthetic)',
        'accuracy': acc,
        'f1_score': f1,
        'n_features': len(feature_names),
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(MODELS_DIR / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
        
    print("🎉 All artifacts saved successfully.")

if __name__ == "__main__":
    train_and_save()
