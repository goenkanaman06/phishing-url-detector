import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from features import extract_features

# Example dataset (for demo)
data = {
    'url': [
        'https://google.com',
        'https://paypal-login.com',
        'http://192.168.1.1/login',
        'https://amazon.com',
        'http://bankofamerica.verify-login.com'
    ],
    'label': [0, 1, 1, 0, 1]  # 1 = phishing, 0 = legitimate
}

df = pd.DataFrame(data)

# Extract features
feature_list = [extract_features(url) for url in df['url']]
X = pd.DataFrame(feature_list)
y = df['label']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')
print("✅ Model trained and saved as model.pkl")
