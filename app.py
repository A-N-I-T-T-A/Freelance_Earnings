from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Load model and encoder from the same file
model, encoder = joblib.load("model.pkl")

# Load dataset to get dropdown values
df = pd.read_csv("perfect_freelancer_subset.csv")
unique_countries = sorted(df["Country"].unique())
unique_platforms = sorted(df["Freelancer_Platform"].unique())

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None
    error = None

    if request.method == "POST":
        try:
            # Get user inputs
            experience = float(request.form["experience"])
            hours_per_week = float(request.form["hours_per_week"])
            country = request.form["country"]
            freelancer_platform = request.form["freelancer_platform"]

            # One-Hot Encode user inputs
            input_df = pd.DataFrame([[country, freelancer_platform]], columns=['Country', 'Freelancer_Platform'])
            encoded_input = encoder.transform(input_df)

            # Prepare final input array
            input_data = np.concatenate([[experience, hours_per_week], encoded_input[0]])

            # Predict price
            predicted_price = model.predict([input_data])[0]

        except Exception as e:
            error = str(e)

    return render_template("index.html", 
                           countries=unique_countries, 
                           platforms=unique_platforms, 
                           predicted_price=predicted_price, 
                           error=error)

if __name__ == "__main__":
    app.run(debug=True)
