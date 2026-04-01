from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Paths to the saved model and scaler
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

def load_assets():
    """Helper function to load pickle files."""
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            return None, None
        with open(MODEL_PATH, 'rb') as m_file:
            model = pickle.load(m_file)
        with open(SCALER_PATH, 'rb') as s_file:
            scaler = pickle.load(s_file)
        return model, scaler
    except Exception as e:
        print(f"Error loading assets: {e}")
        return None, None

model, scaler = load_assets()

@app.route('/', methods=['GET'])
def home():
    """Health check route."""
    return jsonify({
        "status": "online",
        "message": "Salary Prediction API is running.",
        "model_loaded": model is not null
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    if model is None or scaler is None:
        return jsonify({"error": "Model assets not found on server."}), 500

    # Get JSON data
    data = request.get_json(force=True)

    # Validation
    if not data or 'experience' not in data:
        return jsonify({"error": "Missing 'experience' field in request body."}), 400

    try:
        # Convert input to float and reshape for the scaler (2D array)
        exp_value = float(data['experience'])
        input_query = np.array([[exp_value]])

        # Preprocess using the loaded scaler
        input_scaled = scaler.transform(input_query)

        # Perform prediction
        prediction = model.predict(input_scaled)[0]

        return jsonify({
            "experience": exp_value,
            "predicted_salary": round(float(prediction), 2),
            "currency": "USD"
        }), 200

    except ValueError:
        return jsonify({"error": "'experience' must be a numeric value."}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Note: Using app.run for development. 
    # Use Gunicorn or similar for actual production environments.
    app.run(host='0.0.0.0', port=5000, debug=False)
