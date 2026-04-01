from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Paths to the saved model and scaler
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

def load_assets():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            app.logger.error("Model or Scaler file missing!")
            return None, None
        with open(MODEL_PATH, 'rb') as m_file:
            model = pickle.load(m_file)
        with open(SCALER_PATH, 'rb') as s_file:
            scaler = pickle.load(s_file)
        return model, scaler
    except Exception as e:
        app.logger.error(f"Error loading assets: {e}")
        return None, None

model, scaler = load_assets()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model assets not found on server."}), 500

    try:
        data = request.get_json(force=True)
        if not data or 'experience' not in data:
            return jsonify({"error": "Missing 'experience' field."}), 400

        exp_value = float(data['experience'])
        input_query = np.array([[exp_value]])
        input_scaled = scaler.transform(input_query)
        prediction = model.predict(input_scaled)[0]

        return jsonify({
            "experience": exp_value,
            "predicted_salary": round(float(prediction), 2)
        }), 200

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
