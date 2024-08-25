from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

import os

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")

# Define the model path
model_path = os.path.join(os.getcwd(), 'best_model.pkl')
print(f"Expected model path: {model_path}")

# Check if the file exists
if not os.path.exists(model_path):
    print(f"Model file not found at: {model_path}")
else:
    print(f"Model file found at: {model_path}")
    model = joblib.load(model_path)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return jsonify(predictions.tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)