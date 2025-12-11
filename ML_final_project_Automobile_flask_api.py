#Flask API for Customer Segmentation

# build a Flask web application that exposes the model through an API.
# Flask is a lightweight Python web framework used to build APIs and web applications
# Flask is used here to create a customer segmentation API for predicting customer groups

# Flask API for Customer Segmentation

# Flask is a lightweight Python web framework used to build APIs and web applications
# Flask is used here to create a customer segmentation API for predicting customer groups

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Build model path relative to this file
model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")

# Check if model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load saved model
model = joblib.load(model_path)
print("Model loaded successfully!")   # ✅ Log message for confirmation

# Create Flask app
CUSTSEG = Flask(__name__)

# Define prediction route
@CUSTSEG.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'predicted_group': int(prediction[0])})

# Run server
if __name__ == '__main__':
    CUSTSEG.run(host="0.0.0.0", port=5000, debug=True)
