from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the model and scalers
model = load_model('body_metrics_model.keras')

with open('body_metrics_scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

scaler_X = scalers['scaler_X']
scaler_Y = scalers['scaler_Y']

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    safe_to_exercise = None
    if request.method == 'POST':
        # Retrieve input data from the form
        age = float(request.form['age'])
        gender = int(request.form['gender'])  # 1 for Male, 0 for Female
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        bpm = float(request.form['bpm'])  # Heart rate input (separate)
        blood_o2 = float(request.form['blood_o2'])  # Blood O2 input (separate)

        # Create input array for prediction (excluding BPM and Blood O2)
        example_input = np.array([[age, gender, weight, height]])

        # Scale the input data
        example_input_scaled = scaler_X.transform(example_input)

        # Make predictions using the model
        predicted_values = model.predict(example_input_scaled)

        # Inverse transform the predicted values
        predicted_values = scaler_Y.inverse_transform(predicted_values)

        # Extract the predicted values
        bmi, fat_percentage, bmr, lbm, smm = predicted_values[0]

        # Determine if it's safe to exercise based on BPM and Blood O2
        if 60 <= bpm <= 100 and blood_o2 >= 95:
            safe_to_exercise = "Yes, it is safe to exercise."
        else:
            safe_to_exercise = "No, it is not safe to exercise."

        return render_template('index.html', bmi=bmi, fat_percentage=fat_percentage,
                               bmr=bmr, lbm=lbm, smm=smm, safe_to_exercise=safe_to_exercise,
                               age=age, gender=gender, weight=weight, height=height, bpm=bpm, blood_o2=blood_o2)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
