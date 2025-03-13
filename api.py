from flask import Flask, request, jsonify, render_template
import time
import joblib
import pandas as pd
from feature_engineering import extract_features

app = Flask(__name__)
# REMOVE flask_cors
# from flask_cors import CORS
# CORS(app)  #remove flask CORS if serving from flaks

# Load model on startup
try:
    model = joblib.load('fqdn_classifier_model.joblib')
    # model = joblib.load('fqdn_classifier_model_compressed.joblib') # Load compressed model instead
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

def predict_fqdn(model, fqdn):
    features = extract_features(fqdn)
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0]
    result = 'Bad (Malicious)' if prediction == 1 else 'Good (Benign)'
    confidence = probability[1] if prediction == 1 else probability[0]
    return result, confidence

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model is not loaded'}), 500
    data = request.get_json()
    if not data or 'fqdn' not in data:
        return jsonify({'error': 'Missing fqdn in request'}), 400
    fqdn = data['fqdn']
    start_time = time.time()
    result, confidence = predict_fqdn(model, fqdn)
    exec_time = time.time() - start_time
    return jsonify({
        'fqdn': fqdn,
        'classification': result,
        'confidence': f"{confidence:.2%}",
        'execution_time': f"{exec_time:.2f} s"
    })

@app.route('/')
def index():
    return render_template('index.html')  # Make sure your index.html is in a folder called 'templates'

if __name__ == '__main__':
    app.run(debug=True)