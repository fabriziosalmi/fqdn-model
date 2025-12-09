import pytest
import sys
import os
import joblib
import pandas as pd
import numpy as np

# Add project root to path so we can import api and predict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app
import api

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_model(monkeypatch):
    """Mocks the model loading in api.py to valid testing without a real .joblib file."""
    
    class MockModel:
        def __init__(self):
            self.feature_names_in_ = np.array(['feature_1', 'feature_2'])
            
        def predict(self, X):
            # Deterministic prediction based on dummy logic
            # If feature_1 > 0.5 -> 1 (Malicious), else 0 (Benign)
            return [1 if x[0] > 0.5 else 0 for x in X.values]
            
        def predict_proba(self, X):
            return np.array([[0.1, 0.9] if x[0] > 0.5 else [0.9, 0.1] for x in X.values])

    mock_model_instance = MockModel()
    mock_expected_features = ['feature_1', 'feature_2']
    
    # Patch the global variables in api.py
    monkeypatch.setattr(api, 'MODEL', mock_model_instance)
    monkeypatch.setattr(api, 'EXPECTED_FEATURES', mock_expected_features)
    monkeypatch.setattr(api, 'SCALER', None)
    monkeypatch.setattr(api, 'TRANSFORMER', None)
    monkeypatch.setattr(api, 'IMPUTER', None)
    
    return mock_model_instance
