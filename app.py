from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# Career description dictionary
career_info = {
"Data Scientist": "Works with data, machine learning, and analytics.",
"Software Engineer": "Develops software, applications, and systems.",
"UI Designer": "Designs user interfaces and improves user experience.",
"Doctor": "Works in healthcare and medical field.",
"Data Analyst": "Analyzes data to help businesses make decisions."
}

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

    features = [[skill1, skill2, skill3, skill4, skill5]]

    prediction = model.predict(features)[0]

    info = career_info.get(prediction, "")

    result = "Predicted Career: " + prediction + " — " + info

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)