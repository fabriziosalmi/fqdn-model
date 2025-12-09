from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import time
import logging
import os
import re
import settings as conf

from predict import load_models, analyze_with_timeout  # Reuse functions from predict.py

app = Flask(__name__)
log = logging.getLogger("flask_app")
MODEL_DIR = conf.MODEL_DIR
MAX_FQDN_LENGTH = conf.MAX_FQDN_LENGTH

def validate_fqdn(fqdn: str) -> tuple[bool, str]:
    """Validates if the input is a valid FQDN string."""
    if not fqdn or not isinstance(fqdn, str):
        return False, "FQDN must be a non-empty string"
    if len(fqdn) > MAX_FQDN_LENGTH:
        return False, f"FQDN too long (max {MAX_FQDN_LENGTH} chars)"
    
    # Basic FQDN pattern: alphanumeric labels separated by dots
    # This pattern allows for internationalized domains if encoded (punycode) or general usage
    # It ensures no empty labels and valid characters.
    # Note: rigorous FQDN validation is complex, this covers basic sanity/security.
    pattern = r'^([a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
    # We also allow "localhost" or single labels for internal use if needed, but for FQDN model usually we expect dots.
    # For now, let's keep it simple and lenient enough for standard domains but strict against injection chars.
    
    # A safer, simpler check for "safe characters" might be better to avoid regex complexity denial
    if not re.match(r'^[a-zA-Z0-9.-]+$', fqdn):
         return False, "FQDN contains invalid characters"
         
    return True, ""

def load_api_model():
    model, scaler, transformer, imputer = load_models(MODEL_DIR)
    if model is None:
        log.error("Model loading failed.")
        # We don't raise here to allow the app to start, but health check will fail
        return None, None, None, None, []
        
    if hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_.tolist()
    elif hasattr(model, 'n_features_in_'):
        features = [f"feature_{i}" for i in range(model.n_features_in_)]
    else:
         # Fallback or error
        features = []
    return model, scaler, transformer, imputer, features

MODEL, SCALER, TRANSFORMER, IMPUTER, EXPECTED_FEATURES = load_api_model()

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    status = "healthy" if MODEL is not None else "degraded"
    return jsonify({
        "status": status,
        "model_loaded": MODEL is not None,
        "expected_features_count": len(EXPECTED_FEATURES)
    }), 200 if MODEL is not None else 503

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    
    if MODEL is None:
        return jsonify({"error": "Model not loaded service unavailable."}), 503

    data = request.get_json()
    if not data or 'fqdn' not in data:
        return jsonify({"error": "Missing 'fqdn' in request."}), 400
    
    fqdn = data["fqdn"]
    
    valid, message = validate_fqdn(fqdn)
    if not valid:
        return jsonify({"error": message}), 400

    analysis_result = analyze_with_timeout(fqdn)
    if analysis_result is None:
        return jsonify({"error": f"Analysis failed for {fqdn}."}), 500

    # Build feature vector based on the expected features
    features = {}
    for feat in EXPECTED_FEATURES:
        features[feat] = analysis_result.get(feat, 0)
    df = pd.DataFrame([features])
    # Ensure all expected features exist
    for feat in EXPECTED_FEATURES:
        if feat not in df.columns:
            df[feat] = 0
    X = df[EXPECTED_FEATURES]
    
    # Apply preprocessing steps if available
    if IMPUTER:
        X = pd.DataFrame(IMPUTER.transform(X), columns=X.columns)
    if SCALER:
        X = pd.DataFrame(SCALER.transform(X), columns=X.columns)
    if TRANSFORMER:
        X = pd.DataFrame(TRANSFORMER.transform(X), columns=X.columns)
        
    try:
        prediction = MODEL.predict(X)[0]
        probabilities = MODEL.predict_proba(X)[0] if hasattr(MODEL, "predict_proba") else None
    except Exception as e:
        log.error(f"Error during model prediction: {e}")
        return jsonify({"error": "Prediction failed."}), 500
        
    label = "Benign" if prediction == 0 else "Malicious"
    confidence = f"{max(probabilities)*100:.2f}%" if probabilities is not None else "N/A"
    exec_time = time.time() - start_time
    
    return jsonify({
        "fqdn": fqdn,
        "classification": label,
        "confidence": confidence,
        "execution_time": f"{exec_time:.2f} s"
    })

if __name__ == "__main__":
    # SECURITY FIX: Do not hardcode debug=True in production
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode)
