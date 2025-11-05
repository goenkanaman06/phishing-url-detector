from flask import Flask, render_template, request
import joblib
import pandas as pd
from features import extract_features

# Initialize the app
app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    features = extract_features(url)
    df = pd.DataFrame([features])
    result = model.predict(df)[0]
    message = "🚨 Phishing Website" if result == 1 else "✅ Legitimate Website"
    return render_template('index.html', result=message, url=url)

if __name__ == '__main__':
    app.run(debug=True)
