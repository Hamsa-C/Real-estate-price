
import os
import sqlite3
import secrets

import numpy as np
import pandas as pd

from flask import Flask, request, render_template, redirect, session
from werkzeug.security import generate_password_hash, check_password_hash

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# -----------------------------
# FLASK APP SETUP
# -----------------------------
app = Flask(__name__)

# Set FLASK_SECRET_KEY as an environment variable.
# A temporary key is generated for local testing if none is set.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)


# -----------------------------
# CREATE DATABASE
# -----------------------------
def init_db():
    conn = sqlite3.connect("users.db")
    try:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)

        conn.commit()
    finally:
        conn.close()


init_db()


# -----------------------------
# LOAD & PREPARE DATA
# -----------------------------
data = pd.read_excel("dataset_rs.xlsx")

features = [
    "Latitude",
    "Longitude",
    "Proximity_to_City_Center_km",
    "Nearby_Amenities",
    "Area_in_sqft",
    "Bedrooms",
    "Bathrooms",
    "Year_Built",
    "Garage",
    "Swimming_Pool",
    "Garden",
    "Total_Rooms",
    "Building_Condition",
    "Property_Type",
    "Median_Income",
    "Property_Tax_Rate",
    "Local_Crime_Rate",
    "Air_Quality_Index",
    "Noise_Levels",
    "Price_Per_Sqft",
    "Room_to_Area_Ratio"
]

target = "Property_Price"

X = data[features]
y = data[target]

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# -----------------------------
# DATA PREPROCESSING
# -----------------------------
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# -----------------------------
# MACHINE LEARNING MODEL
# -----------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])


# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# TRAIN MODEL
# -----------------------------
model.fit(X_train, y_train)


# -----------------------------
# MODEL PREDICTION & EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n-----------------------------")
print("MODEL EVALUATION RESULTS")
print("-----------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))
print("-----------------------------\n")


# -----------------------------
# REGISTER ROUTE
# -----------------------------
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]
        full_name = request.form["full_name"].strip()
        email = request.form["email"].strip()
        phone = request.form["phone"].strip()

        # Hash password before storing
        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect("users.db")

        try:
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO users (full_name, username, email, phone, password) "
    "VALUES (?, ?, ?, ?, ?)",
    (full_name, username, email, phone, hashed_password)
            )

            conn.commit()

        except sqlite3.IntegrityError:
            return render_template(
                "register.html",
                error="Username already exists!"
            )

        finally:
            conn.close()

        return redirect("/login")

    return render_template("register.html")


# -----------------------------
# LOGIN ROUTE
# -----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")

        try:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM users WHERE username=?",
                (username,)
            )

            user = cursor.fetchone()

        finally:
            conn.close()

        # Verify hashed password
        if user and check_password_hash(user[2], password):

            session.clear()
            session["user"] = username

            return redirect("/")

        else:
            return render_template(
                "login.html",
                error="Invalid Username or Password"
            )

    return render_template("login.html")


# -----------------------------
# LOGOUT ROUTE
# -----------------------------
@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect("/login")


# -----------------------------
# HOME & PRICE PREDICTION
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    if "user" not in session:
        return redirect("/login")

    prediction = None

    if request.method == "POST":

        try:
            input_data = {
                "Latitude": float(request.form["latitude"]),
                "Longitude": float(request.form["longitude"]),
                "Proximity_to_City_Center_km": float(request.form["proximity"]),
                "Nearby_Amenities": int(request.form["amenities"]),
                "Area_in_sqft": int(request.form["area"]),
                "Bedrooms": int(request.form["bedrooms"]),
                "Bathrooms": int(request.form["bathrooms"]),
                "Year_Built": int(request.form["year_built"]),
                "Garage": int(request.form["garage"]),
                "Swimming_Pool": int(request.form["swimming_pool"]),
                "Garden": int(request.form["garden"]),
                "Total_Rooms": int(request.form["total_rooms"]),
                "Building_Condition": request.form["building_condition"],
                "Property_Type": request.form["property_type"],
                "Median_Income": float(request.form["median_income"]),
                "Property_Tax_Rate": float(request.form["property_tax_rate"]),
                "Local_Crime_Rate": float(request.form["local_crime_rate"]),
                "Air_Quality_Index": int(request.form["air_quality_index"]),
                "Noise_Levels": int(request.form["noise_levels"]),
                "Price_Per_Sqft": float(request.form["price_per_sqft"]),
                "Room_to_Area_Ratio": float(request.form["room_to_area_ratio"])
            }

            input_df = pd.DataFrame([input_data])

            predicted_price = model.predict(input_df)[0]

            prediction = round(float(predicted_price), 2)

        except (ValueError, KeyError, TypeError):
            return render_template(
                "index.html",
                prediction=None,
                error="Please enter valid property details."
            )

    return render_template(
        "index.html",
        prediction=prediction
    )


# -----------------------------
# RUN FLASK APPLICATION
# -----------------------------
if __name__ == "__main__":

    app.run(debug=False)