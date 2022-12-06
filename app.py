from flask import Flask, render_template, request
import pandas as pd
import pickle
import xgboost as xgb

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

    with open("cow.pk", mode="rb") as fp:
        model = pickle.load(fp)

    f_names = model.feature_names
    test_df = pd.DataFrame(index=[1], columns=f_names)
    test_df = test_df.fillna(0)

    sex = request.form["sex"]
    father = request.form["father"]
    gland = request.form["gland"]
    gege = request.form["gege"]
    got = request.form["got"]

    test_df[f"性別_{sex}"] = 1
    test_df[f"父牛_{father}"] = 1
    test_df[f"母の父_{gland}"] = 1
    test_df[f"母の祖父_{gege}"] = 1
    test_df[f"母の祖祖父_{got}"] = 1

    df_dummied = pd.get_dummies(test_df)
    e = df_dummied.tail(1)
    d = model.predict(xgb.DMatrix(e))
    result = int(d)
    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
