#getting answer using postman only
from flask import Flask, request, jsonify, render_template
import pandas as pd

import pickle
app = Flask(__name__)


# @app.route("/")
# def home():
#     return render_template("index.html")


@app.route("/predict", methods=["POST"])
def get_prediction():
    baby_data = request.get_json()
    baby_df = pd.DataFrame(baby_data)
    # baby_df = pd.DataFrame(baby_data)
    with open("model/model.pkl", "rb") as obj:
        model = pickle.load(obj)
    
    prediction = model.predict(baby_df)
    print(prediction)
    # pred = round(float(prediction[0]),2)
    response = {"prediction" : round(float(prediction[0]),2)}
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
# from flask import Flask, request, jsonify
# app = Flask(__name__)

# @app.route("/predict", methods=["POST"])
# def get_prediction():
#     data = request.get_json()
#     print("Received:", data)
#     return jsonify({"message": "Data received", "data": data})

# if __name__ == "__main__":
#     app.run(debug=True)
