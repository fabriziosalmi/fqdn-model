import pytest
import time
from predict import analyze_with_timeout, AnalysisTimeoutError
import threading

def test_analyze_timeout_success(monkeypatch):
    """Test that analyze_with_timeout returns result when function completes in time."""
    
    def mock_analyze_fast(fqdn, **kwargs):
        return {"result": "success"}
    
    monkeypatch.setattr('predict.analyze_fqdn', mock_analyze_fast)
    
    result = analyze_with_timeout("fast.com")
    assert result == {"result": "success"}

def test_analyze_timeout_failure(monkeypatch):
    """Test that analyze_with_timeout returns None when function takes too long."""
    
    # We need to lower the timeout for the test, or mock the constant
    monkeypatch.setattr('predict.ANALYSIS_TIMEOUT', 0.1)
    
    def mock_analyze_slow(fqdn, **kwargs):
        time.sleep(0.5)
        return {"result": "too_slow"}
    
    monkeypatch.setattr('predict.analyze_fqdn', mock_analyze_slow)
    
    result = analyze_with_timeout("slow.com")
    assert result is None
