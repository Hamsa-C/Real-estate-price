from flask import Flask, request, render_template, redirect, session
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

# -----------------------------
# FLASK APP SETUP
# -----------------------------
app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# -----------------------------
# RESET OR CREATE DATABASE
# -----------------------------
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# LOAD & PREPARE DATA
# -----------------------------
data = pd.read_excel('dataset_rs.xlsx')

features = [
    'Latitude', 'Longitude', 'Proximity_to_City_Center_km', 'Nearby_Amenities',
    'Area_in_sqft', 'Bedrooms', 'Bathrooms', 'Year_Built', 'Garage', 
    'Swimming_Pool', 'Garden', 'Total_Rooms', 'Building_Condition', 
    'Property_Type', 'Median_Income', 'Property_Tax_Rate', 'Local_Crime_Rate', 
    'Air_Quality_Index', 'Noise_Levels', 'Price_Per_Sqft', 'Room_to_Area_Ratio'
]
target = 'Property_Price'

X = data[features]
y = data[target]

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

model.fit(X, y)

# -----------------------------
# ROUTES
# -----------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            return redirect('/login')
        except sqlite3.IntegrityError:
            return render_template('register.html', error="Username already exists!")

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['user'] = username
            return redirect('/')
        else:
            return render_template('login.html', error="Invalid Username or Password")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

@app.route('/', methods=['GET', 'POST'])
def home():
    if 'user' not in session:
        return redirect('/login')

    prediction = None
    if request.method == 'POST':
        input_data = {
            'Latitude': float(request.form['latitude']),
            'Longitude': float(request.form['longitude']),
            'Proximity_to_City_Center_km': float(request.form['proximity']),
            'Nearby_Amenities': int(request.form['amenities']),
            'Area_in_sqft': int(request.form['area']),
            'Bedrooms': int(request.form['bedrooms']),
            'Bathrooms': int(request.form['bathrooms']),
            'Year_Built': int(request.form['year_built']),
            'Garage': int(request.form['garage']),
            'Swimming_Pool': int(request.form['swimming_pool']),
            'Garden': int(request.form['garden']),
            'Total_Rooms': int(request.form['total_rooms']),
            'Building_Condition': request.form['building_condition'],
            'Property_Type': request.form['property_type'],
            'Median_Income': float(request.form['median_income']),
            'Property_Tax_Rate': float(request.form['property_tax_rate']),
            'Local_Crime_Rate': float(request.form['local_crime_rate']),
            'Air_Quality_Index': int(request.form['air_quality_index']),
            'Noise_Levels': int(request.form['noise_levels']),
            'Price_Per_Sqft': float(request.form['price_per_sqft']),
            'Room_to_Area_Ratio': float(request.form['room_to_area_ratio'])
        }
        input_df = pd.DataFrame([input_data])
        prediction = round(model.predict(input_df)[0], 2)

    return render_template('index.html', prediction=prediction)

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
