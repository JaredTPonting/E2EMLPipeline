from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


model = joblib.load("app/best_model.pkl")

expected_columns = ['Pclass_2', 'Pclass_3', 'Sex_male', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S']


@app.route('/')
def home():
    return "Hello, Heroku!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    try:
        df = pd.DataFrame([data['features']], columns=expected_columns)
        predictions = model.predict(df)
        return jsonify(predictions.tolist())
    except Exception as e:
        return str(e), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)