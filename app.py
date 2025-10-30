from flask import Flask, request, jsonify, render_template
import pandas as pd
# import render_template
import pickle
# from flask import jsonify
# import pickle
# import jsonify
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def get_prediction():
    data = request.get_json()
    df = pd.DataFrame({
        "gestation": data["gestation"],
        "parity": data["parity"],
        "age": data["age"],
        "height": data["height"],
        "weight": data["weight"],
        "smoke": data["smoke"]
    })
    # baby_df = pd.DataFrame(baby_data)
    with open("model/model.pkl", "rb") as obj:
        model = pickle.load(obj)
    prediction = model.predict(df)[0]
    return jsonify({"prediction" : round(float(prediction),2)})

if __name__ == "__main__":
    app.run(debug=True)
