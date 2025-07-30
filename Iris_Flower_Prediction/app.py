from flask import Flask, request, render_template, jsonify
from model import predict_species  

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [
        float(data["sepal_length"]),
        float(data["sepal_width"]),
        float(data["petal_length"]),
        float(data["petal_width"])
    ]
    prediction = predict_species(features)
    return jsonify({"species": prediction})  # Must match what script.js expects

if __name__ == "__main__":
    app.run(debug=True)
