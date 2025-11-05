# app.py
import os
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from features import extract_features

app = Flask(__name__, template_folder="templates")

# Load model at startup
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Place model.pkl in project root or set MODEL_PATH.")
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts form submission (from HTML) or JSON { "url": "<url>" }.
    Returns rendered template for form requests, and JSON for API calls.
    """
    # Get URL from JSON body or form data
    data = request.get_json(silent=True) or {}
    url = data.get("url") or request.form.get("url")
    if not url:
        # For form requests show the page with an error
        if request.form:
            return render_template("index.html", result="Error: No URL provided.", url="")
        return jsonify({"error": "No URL provided."}), 400

    # Basic input length check
    if len(url) > 2000:
        return (render_template("index.html", result="Error: URL too long.", url=url)
                if request.form else
                (jsonify({"error":"URL too long."}), 400))

    try:
        # Extract features (returns a dict). Make sure features.py matches training features.
        feats = extract_features(url)
        X = pd.DataFrame([feats])

        # Predict: model.predict returns class (0 or 1). Try to get probability if available.
        pred = int(model.predict(X)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[:, 1][0])

        label = "Phishing 🚨" if pred == 1 else "Legitimate ✅"
        pretty_prob = f"{prob*100:.2f}%" if prob is not None else "N/A"

        # If request came from HTML form, render the page
        if request.form:
            return render_template("index.html", result=label, probability=pretty_prob, url=url)

        # Otherwise return JSON for API usage
        return jsonify({"prediction": pred, "label": label, "probability": prob})

    except Exception as e:
        # For safety, don't leak internal stack traces to end users in production
        err_msg = f"Internal error: {str(e)}"
        if request.form:
            return render_template("index.html", result=err_msg, url=url)
        return jsonify({"error": err_msg}), 500


if __name__ == "__main__":
    # Use Render-provided PORT if present, else default to 5000 for local dev
    port = int(os.environ.get("PORT", 5000))
    # Bind to 0.0.0.0 so Render can route to the app; debug=False for deployment
    app.run(host="0.0.0.0", port=port, debug=False)
