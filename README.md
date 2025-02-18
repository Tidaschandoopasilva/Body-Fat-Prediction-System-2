Fitness Tracker Web App - Version 2.0
This is the second version of the Fitness Tracker project, designed to provide users with advanced fitness metrics and an improved user experience. The app leverages a predictive model enhanced with age, gender, height, and weight as input parameters to estimate critical fitness metrics such as:

BMI (Body Mass Index)
Body Fat Percentage
BMR (Basal Metabolic Rate)
LBM (Lean Body Mass)
SMM (Skeletal Muscle Mass)
In addition to the core predictions, this version includes new features for gym users:

Inputs for Heart Rate (BPM) and Blood Oxygen Levels (SpO2) to check if it's safe to exercise.
A safety recommendation based on the user's vitals.
Key Features
Improved Prediction Model: Updated model for accurate fitness metric calculations.
Modern UI: A sleek interface built with Bootstrap 5 and Font Awesome Icons for a better user experience.
Clear Input/Output Controls: A Clear button to reset all inputs and results for quick re-calculations.
Exercise Safety Check: Provides a recommendation on whether the user is in a safe range for physical activity.
Responsive Design: Works seamlessly across desktops, tablets, and mobile devices.
Technologies Used
Frontend: HTML5, CSS3, Bootstrap 5, Font Awesome Icons
Backend: Flask (Python)
Model: Keras with TensorFlow backend
Scalers: StandardScaler for input/output data normalization
How to Run
Clone the repository:
bash
Copy
Edit
git clone https://github.com/yourusername/fitness-tracker-v2.git  
cd fitness-tracker-v2  
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt  
Run the Flask app:
bash
Copy
Edit
python app.py  
Open your browser and navigate to http://127.0.0.1:5000.
Future Enhancements
Integration with wearable devices for real-time data input.
Support for multiple users and fitness progress tracking.
Inclusion of additional health metrics like VO2 Max.
Contributions and feedback are welcome! 🚀
