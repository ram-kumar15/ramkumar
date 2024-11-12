from flask import Flask, jsonify, request
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from flask_mysqldb import MySQL
from datetime import datetime

app = Flask(__name__)

# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'smart_water_management'

mysql = MySQL(app)

# Training the model
def train_model():
    # Original data
    data = pd.DataFrame({
        'water_level': [100, 150, 200, 250, 300, 350, 400],
        'rainfall': [10, 20, 30, 40, 50, 60, 70],
        'flood_risk': [0, 0, 1, 1, 1, 1, 1]
    })

    # Generate synthetic data
    synthetic_data = pd.DataFrame({
        'water_level': pd.Series(pd.np.random.uniform(50, 500, 300)),
        'rainfall': pd.Series(pd.np.random.uniform(5, 100, 300))
    })
    synthetic_data['flood_risk'] = synthetic_data.apply(
        lambda row: 1 if row['water_level'] > 300 and row['rainfall'] > 50 else 0, axis=1
    )

    # Combine data
    data = pd.concat([data, synthetic_data], ignore_index=True)

    # Split data
    X = data[['water_level', 'rainfall']]
    y = data['flood_risk']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, 'flood_risk_predictor.joblib')
    return model

# Load or train the model
try:
    model = joblib.load('flood_risk_predictor.joblib')
except FileNotFoundError:
    model = train_model()

@app.route('/predictFloodRisk', methods=['POST'])
def predict_flood_risk():
    try:
        # Get data from request
        data = request.get_json()
        water_level = data.get('water_level')
        rainfall = data.get('rainfall')
        location = data.get('location', 'Unknown')

        # Validate input
        if water_level is None or rainfall is None:
            return jsonify({"error": "Missing parameters: 'water_level' and 'rainfall'"}), 400

        # Predict flood risk
        flood_risk = model.predict([[float(water_level), float(rainfall)]])[0]
        risk_label = "High" if flood_risk else "Low"

        # Save data to database
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO water_data (water_level, rainfall, location, risk_level) VALUES (%s, %s, %s, %s)",
            (water_level, rainfall, location, risk_label)
        )
        mysql.connection.commit()
        cur.close()

        # Return prediction
        return jsonify({
            "water_level": water_level,
            "rainfall": rainfall,
            "location": location,
            "flood_risk": risk_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getData', methods=['GET'])
def get_data():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM water_data ORDER BY timestamp DESC")
        rows = cur.fetchall()
        cur.close()

        data = [{
            "id": row[0],
            "water_level": row[1],
            "rainfall": row[2],
            "location": row[3],
            "risk_level": row[4],
            "timestamp": row[5].strftime("%Y-%m-%d %H:%M:%S")
        } for row in rows]

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
