# app/routes.py
from flask import render_template, request, redirect, url_for
from app import app
import pickle
import numpy as np
import pandas as pd

# Load model dan scaler
with open("app/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("app/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Kolom fitur yang digunakan saat melatih model
model_columns = model.feature_names_in_


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global model_columns

    # Ambil data dari form
    year = int(request.form["year"])
    mileage = float(request.form["mileage"])
    engineSize = float(request.form["engineSize"])
    model_input = request.form["model"]
    transmission = request.form["transmission"]
    fuelType = request.form["fuelType"]
    brand = request.form["brand"]

    # Buat DataFrame dengan input
    data = {
        "year": [year],
        "mileage": [mileage],
        "engineSize": [engineSize],
        "model_" + model_input: [1],
        "transmission_" + transmission: [1],
        "fuelType_" + fuelType: [1],
        "brand_" + brand: [1],
    }
    df = pd.DataFrame(data)

    # Menambahkan kolom yang hilang dengan nilai 0
    missing_cols = set(model_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    # Susun ulang kolom agar sesuai dengan model
    df = df[model_columns]

    # Standarisasi input
    # X_scaled = scaler.transform(df)

    # print(X_scaled)

    # prediction = model.predict(X_scaled)[0]
    
    # Prediksi harga
    prediction = model.predict(df)[0]

    # Ubah prediksi ke dalam format uang
    prediction = round(prediction, 2)
    prediction = "£ " + str(prediction)
    return render_template("index.html", prediction=prediction)
