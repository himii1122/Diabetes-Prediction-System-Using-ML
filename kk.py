from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    value = float(request.form["feature"])
    prediction = model.predict([[value]])
    return render_template("index.html", result=f"Predicted Class: {prediction[0]}")

if __name__ == "__main__":
    app.run(debug=True)
