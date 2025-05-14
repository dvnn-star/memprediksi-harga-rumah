from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../model')


app = Flask(__name__)
CORS(app)
CORS(app, origins=["http://127.0.0.1:5500"])

#load model dan scaler


model = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
features = joblib.load(os.path.join(MODEL_DIR, 'features.pkl'))
@app.route ('/predict', methods=['POST'])
def predict():
    try:
        data= request.get_json()
        df = pd.DataFrame([data])
        df =df[features]
        df = df.astype(float)
        df_scaled = scaler.transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=features, index=df.index)
    
        prediction = model.predict(df_scaled)
        result = np.expm1(prediction[0])
        return jsonify({'prediction': round(result, 2)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

