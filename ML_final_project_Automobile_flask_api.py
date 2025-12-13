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
import json

# Load model
model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
model = joblib.load(model_path)
print("Model loaded successfully!")

# Load saved label encoders
le = joblib.load("label_encoders.pkl")

# ✅ Load label mappings (must be BEFORE predict())
label_mappings = joblib.load("label_mappings.pkl")
segment_map = label_mappings["Segmentation"]

# Sample input for display
sample_input = {
    "Gender": "Female",
    "Ever_Married": "Yes",
    "Age": 35,
    "Graduated": "Yes",
    "Profession": "Engineer",
    "Work_Experience": 5,
    "Spending_Score": "Low",
    "Family_Size": 3,
    "Category": "Car"
}

print("\n✅ CUSTOMER SEGMENTATION API IS READY")
print("✅ POST URL:")
print("   http://127.0.0.1:5000/predict")
print("\n✅ SAMPLE INPUT JSON:")
print(json.dumps(sample_input, indent=2))
print("\n")

# Create Flask app
CUSTSEG = Flask(__name__)

# ✅ Prediction route
@CUSTSEG.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])

    # ✅ Apply label encoding to categorical columns
    for col in df.columns:
        if col in le:
            df[col] = le[col].transform(df[col])

    prediction = model.predict(df)

    segment_code = int(prediction[0])
    segment_label = segment_map[segment_code]

    return jsonify({
        "predicted_segment": segment_label,
        "segment_code": segment_code
    })

# Run server
if __name__ == '__main__':
    CUSTSEG.run(host="0.0.0.0", port=5000, debug=True)




