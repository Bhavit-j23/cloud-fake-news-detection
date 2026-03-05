from flask import Flask, request, jsonify
import pickle

# Create Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return "Fake News Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["text"]
    vector = vectorizer.transform([data])
    prediction = model.predict(vector)

    if prediction[0] == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)