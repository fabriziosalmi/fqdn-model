from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import time
import logging

from predict import load_models, analyze_with_timeout  # Reuse functions from predict.py

app = Flask(__name__)
log = logging.getLogger("flask_app")
MODEL_DIR = "models"

def load_api_model():
    model, scaler, transformer, imputer = load_models(MODEL_DIR)
    if model is None:
        log.error("Model loading failed.")
        raise Exception("Model loading failed.")
    if hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_.tolist()
    elif hasattr(model, 'n_features_in_'):
        features = [f"feature_{i}" for i in range(model.n_features_in_)]
    else:
        raise Exception("Cannot determine expected features.")
    return model, scaler, transformer, imputer, features

MODEL, SCALER, TRANSFORMER, IMPUTER, EXPECTED_FEATURES = load_api_model()

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    data = request.get_json()
    if not data or 'fqdn' not in data:
        return jsonify({"error": "Missing 'fqdn' in request."}), 400
    fqdn = data["fqdn"]
    
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
    app.run(debug=True)
