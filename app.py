from flask import Flask, render_template, request
import pandas as pd
import joblib
import warnings
import xgboost as xgb

app = Flask(__name__)


def price_predict(df):
    column = df.columns[5:9]

    for col in column:
        df[col] = df[col].astype('int')

    df_dummied = pd.get_dummies(df)
    model = joblib.load("./cow.pkl")
    e = df_dummied.tail(1)
    a = e.drop(columns = ['価格'])
    a = xgb.DMatrix(a)
    d = model.predict(a)
    return int(d)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/cow_much", methods=["POST"])
def cow_much():
    sex = request.form["sex"]
    father = request.form["father"]
    gland = request.form["gland"]
    gege = request.form["gege"]
    got = request.form["got"]
    age = request.form["age"]
    wight = request.form["wight"]

    df = pd.DataFrame(
        {
            "性別": [sex],
            "父牛": [father],
            "母の父": [gland],
            "母の祖父": [gege],
            "母の祖祖父": [got],
            "日令": [age],
            "体重": [wight],
            "価格":1,
        }
    )

    result = price_predict(df)
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
