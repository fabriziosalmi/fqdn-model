import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load best model and preprocessing objects
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'gaussian_nb_best_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
TRANSFORMER_PATH = os.path.join(MODEL_DIR, 'quantile_transformer.joblib')
IMPUTER_PATH = os.path.join(MODEL_DIR, 'imputer.joblib')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
transformer = joblib.load(TRANSFORMER_PATH) if os.path.exists(TRANSFORMER_PATH) else None
imputer = joblib.load(IMPUTER_PATH) if os.path.exists(IMPUTER_PATH) else None

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON with a "data" key containing a list of feature dictionaries.
    The input features must match the training features.
    """
    json_data = request.get_json()
    if not json_data or 'data' not in json_data:
        return jsonify({'error': 'Invalid input format. Expected JSON with "data" key.'}), 400

    try:
        # Convert JSON into DataFrame
        data = pd.DataFrame(json_data['data'])
        # Apply preprocessing
        if imputer:
            data[:] = imputer.transform(data)
        if scaler:
            data[:] = scaler.transform(data)
        if transformer:
            data[:] = transformer.transform(data)
        predictions = model.predict(data)
        proba = model.predict_proba(data).tolist() if hasattr(model, "predict_proba") else None
        return jsonify({'predictions': predictions.tolist(), 'probabilities': proba})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)