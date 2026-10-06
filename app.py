from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("parking_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict")
def predict():
    time = float(request.args.get("time", 12))
    total_slots = float(request.args.get("total_slots", 100))
    occupied_slots = float(request.args.get("occupied_slots", 50))

    data = pd.DataFrame(
        [[time, total_slots, occupied_slots]],
        columns=["time", "total_slots", "occupied_slots"]
    )

    prediction = model.predict(data)[0]

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Parking Prediction</title>

        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #eaf4ff;
                text-align: center;
                padding-top: 80px;
            }}

            .result-box {{
                background: white;
                width: 400px;
                margin: auto;
                padding: 35px;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            }}

            h1 {{
                color: #1769aa;
            }}

            .slots {{
                font-size: 32px;
                font-weight: bold;
                color: #1769aa;
            }}

            a {{
                display: inline-block;
                margin-top: 20px;
                padding: 10px 20px;
                background: #1769aa;
                color: white;
                text-decoration: none;
                border-radius: 8px;
            }}
        </style>
    </head>

    <body>

        <div class="result-box">

            <h1>Parking Prediction</h1>

            <p>Predicted Available Slots</p>

            <div class="slots">{round(prediction, 2)}</div>

            <p>parking slots are available.</p>

            <a href="/">Back to Home</a>

        </div>

    </body>
    </html>
    """


if __name__ == "__main__":
    app.run(debug=True)