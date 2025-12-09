import pytest
import json

def test_health_check(client, mock_model):
    """Test the health check endpoint returns 200 and healthy status."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True

def test_predict_success(client, mock_model, monkeypatch):
    """Test a successful prediction flow."""
    
    # Mock analyze_with_timeout to return compatible features
    def mock_analyze(fqdn):
        return {'feature_1': 0.8, 'feature_2': 0.2}
    
    monkeypatch.setattr('api.analyze_with_timeout', mock_analyze)
    
    payload = {'fqdn': 'malicious.example.com'}
    response = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['fqdn'] == 'malicious.example.com'
    assert data['classification'] == 'Malicious' # Based on mock logic (0.8 > 0.5)

def test_predict_missing_fqdn(client):
    """Test validating missing fqdn field."""
    response = client.post('/predict', data=json.dumps({}), content_type='application/json')
    assert response.status_code == 400
    assert "Missing 'fqdn'" in response.get_json()['error']

def test_predict_invalid_fqdn_chars(client):
    """Test validating invalid characters in FQDN."""
    payload = {'fqdn': 'invalid_char$.com'}
    response = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    assert response.status_code == 400
    assert "invalid characters" in response.get_json()['error']

def test_predict_empty_fqdn(client):
    """Test validating empty FQDN."""
    payload = {'fqdn': ''}
    response = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    assert response.status_code == 400
