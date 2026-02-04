"""
ML Model Loader for Cardano Insurance dApp
Loads the pre-trained Gradient Boosting model and handles feature engineering
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


class FraudDetectionModel:
    """Healthcare fraud detection model wrapper"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load model, scaler, and feature names"""
        try:
            # Load model
            with open(self.models_dir / 'best_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open(self.models_dir / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature names
            with open(self.models_dir / 'feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            
            # Load metadata
            with open(self.models_dir / 'metadata.pkl', 'rb') as f:
                self.metadata = pickle.load(f)
            
            print(f"✓ Model loaded: {self.metadata['model_name']}")
            print(f"  - Accuracy: {self.metadata['accuracy']:.4f}")
            print(f"  - F1 Score: {self.metadata['f1_score']:.4f}")
            print(f"  - Features: {self.metadata['n_features']}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifacts: {e}")
    
    def preprocess_input(self, amount_billed: float, age: int, gender: str, diagnosis: str) -> pd.DataFrame:
        """
        Convert API input to model-ready features with One-Hot Encoding
        
        Args:
            amount_billed: Claim amount in currency
            age: Patient age
            gender: 'Male' or 'Female'
            diagnosis: Diagnosis category
        
        Returns:
            DataFrame with all required features (13 columns after encoding)
        """
        # Create base dataframe
        data = pd.DataFrame({
            'Amount Billed': [amount_billed],
            'Age': [age],
            'Gender': [gender],
            'Diagnosis': [diagnosis],
            'StayDuration': [1]  # Default stay duration for API predictions
        })
        
        # One-Hot Encode (matching training pipeline)
        data_encoded = pd.get_dummies(data, drop_first=True)
        
        # Align with training features (critical step!)
        # Fill missing columns with 0
        for col in self.feature_names:
            if col not in data_encoded.columns:
                data_encoded[col] = 0
        
        # Reindex to match exact training feature order
        data_encoded = data_encoded[self.feature_names]
        
        # Fill any remaining NaN values
        data_encoded = data_encoded.fillna(0)
        
        return data_encoded
    
    def predict(self, amount_billed: float, age: int, gender: str, diagnosis: str, confidence_threshold: float = 0.90) -> Dict:
        """
        Predict fraud status for a claim with confidence threshold
        
        Args:
            amount_billed: Claim amount
            age: Patient age
            gender: 'Male' or 'Female'
            diagnosis: Diagnosis category
            confidence_threshold: Minimum confidence required to accept (default: 0.95 = 95%)
        
        Returns:
            {
                'prediction': 0 (Genuine) or 1 (Fraud),
                'prediction_label': 'genuine' or 'fake',
                'confidence': probability of the predicted class,
                'model_name': str
            }
        """
        try:
            # Preprocess input
            X = self.preprocess_input(amount_billed, age, gender, diagnosis)
            
            # Note: Gradient Boosting doesn't need scaling (tree-based model)
            # If model was SVC or KNN, we would use: X_scaled = self.scaler.transform(X)
            
            # Predict
            prediction = int(self.model.predict(X)[0])
            
            # Get probability if available
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)[0]
                confidence = float(proba[prediction])
            else:
                confidence = 1.0
            
            # Apply confidence threshold for genuine claims
            # If prediction is genuine (0) but confidence < threshold, reject it
            if prediction == 0 and confidence < confidence_threshold:
                prediction = 1  # Change to fraud/fake
                prediction_label = 'fake'
                rejection_reason = f'low_confidence'
            else:
                prediction_label = 'genuine' if prediction == 0 else 'fake'
                rejection_reason = None
            
            return {
                'prediction': prediction,
                'prediction_label': prediction_label,
                'confidence': confidence,
                'model_name': self.metadata['model_name'],
                'confidence_threshold': confidence_threshold,
                'rejection_reason': rejection_reason,
                'details': {
                    'amount_billed': amount_billed,
                    'age': age,
                    'gender': gender,
                    'diagnosis': diagnosis
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def get_feature_importance(self, top_n: int = 10) -> List[Dict]:
        """Get top N important features if available"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            features = [
                {'feature': name, 'importance': float(imp)}
                for name, imp in zip(self.feature_names, importances)
            ]
            return sorted(features, key=lambda x: x['importance'], reverse=True)[:top_n]
        return []


# Global model instance
_model_instance = None


def get_model() -> FraudDetectionModel:
    """Singleton pattern for model loading"""
    global _model_instance
    if _model_instance is None:
        _model_instance = FraudDetectionModel()
    return _model_instance


if __name__ == "__main__":
    # Test model loading and prediction
    model = get_model()
    
    # Test cases
    test_cases = [
        {"amount_billed": 15000, "age": 65, "gender": "Female", "diagnosis": "Pregnancy"},
        {"amount_billed": 5000, "age": 30, "gender": "Female", "diagnosis": "Hypertension"},
        {"amount_billed": 2000, "age": 45, "gender": "Male", "diagnosis": "Diabetes"},
    ]
    
    print("\n" + "="*70)
    print("TEST PREDICTIONS:")
    print("="*70)
    for i, case in enumerate(test_cases, 1):
        result = model.predict(**case)
        print(f"\nTest {i}: {case}")
        print(f"  → Prediction: {result['prediction_label'].upper()} ({result['confidence']:.2%} confidence)")
    
    # Show feature importance
    print("\n" + "="*70)
    print("TOP FRAUD INDICATORS:")
    print("="*70)
    for feat in model.get_feature_importance(top_5):
        print(f"  {feat['feature']:30} | {feat['importance']:.4f}")
