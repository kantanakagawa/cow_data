from flask import Flask, render_template, request
import pandas as pd
import pickle
import xgboost as xgb
import numpy as np


# def price_predict(df):
#     df_dummied = pd.get_dummies(df)
#     with open("cow.pk",mode="rb")as fp:
#         model=pickle.load(fp)
#     e = df_dummied.tail(1)
#     a = e.drop(columns = ['価格'])
#     d = model.predict(xgb.DMatrix(a))
#     df = df[:-1]
#     return int(d),df


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/cow_much", methods=["POST"])
def cow_much():
    df = pd.read_csv("cow_data.csv")
    columns = df.columns[1:9]
    df = df.iloc[0 : len(df), [1, 2, 3, 4, 5, 6, 7, 8]]
    df.columns = columns

    sex = request.form["sex"]
    father = request.form["father"]
    gland = request.form["gland"]
    gege = request.form["gege"]
    got = request.form["got"]
    age = request.form["age"]
    wight = request.form["wight"]

    df = df.append(
        {
            "性別": sex,
            "父牛": father,
            "母の父": gland,
            "母の祖父": gege,
            "母の祖祖父": got,
            "日令": int(age),
            "体重": int(wight),
            "価格": 1,
        },
        ignore_index=True,
    )

    df_dummied = pd.get_dummies(df, drop_first=True)
    with open("cow.pk", mode="rb") as fp:
        model = pickle.load(fp)
    cols_when_model_builds = model.get_booster().feature_names
    e = df_dummied.tail(1)
    a = e.drop(columns=["価格"])
    d = model.predict(xgb.DMatrix(a))
    df = df[:-1]
    result = int(d)
    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
