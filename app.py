from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "ML Model is Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Validation
    if "features" not in data:
        return jsonify({"error": '"features" key is missing'}), 400

    features = data["features"]

    if not isinstance(features, list):
        return jsonify({"error": '"features" must be a list'}), 400

    for item in features:
        if not isinstance(item, list) or len(item) != 4 or not all(isinstance(x, (int, float)) for x in item):
            return jsonify({"error": "Each input must be a list of 4 numeric values"}), 400

    X = np.array(features)
    predictions = model.predict(X).tolist()

    if hasattr(model, "predict_proba"):
        confidences = model.predict_proba(X).max(axis=1).tolist()
        return jsonify({
            "predictions": predictions,
            "confidences": confidences
        })
    else:
        return jsonify({
            "predictions": predictions
        })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
