from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    skill1 = int(request.form["skill1"])
    skill2 = int(request.form["skill2"])
    skill3 = int(request.form["skill3"])
    skill4 = int(request.form["skill4"])
    skill5 = int(request.form["skill5"])

    prediction = model.predict([[skill1, skill2, skill3, skill4, skill5]])

    return render_template("index.html", prediction_text="Predicted Career: " + prediction[0])

if __name__ == "__main__":
    app.run()