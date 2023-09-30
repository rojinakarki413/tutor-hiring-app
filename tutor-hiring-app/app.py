import joblib
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# model = pickle.load(open('SentimentAnalysisModel.pkl', 'rb'))
model = joblib.load('sentiment_model.pkl')
app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    # review = request.form.get('review')
    # result = model.predict(review)[0]
    new_review = [str(x) for x in request.form.values()]
    pred = model.predict(new_review)[0]
    return jsonify({'sentiment': str(pred)})


if __name__ == '__main__':
    app.run(debug=True)
