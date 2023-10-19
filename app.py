import pandas as pd
from flask import Flask,request, url_for, redirect, render_template
import joblib
import numpy as np

app = Flask(__name__)

model= joblib.load('model/model.pkl')
class_list = ["Normal", "Suspect", "Pathological"]
input_data = {
    "baseline value":[0],
    "accelerations":[0],
    "fetal_movement":[0],
    "uterine_contractions":[0], 
    "light_decelerations":[0], 
    "severe_decelerations":[0],
    "prolongued_decelerations":[0], 
    "abnormal_short_term_variability":[0],
    "mean_value_of_short_term_variability":[0],
    "percentage_of_time_with_abnormal_long_term_variability":[0],
    "mean_value_of_long_term_variability":[0], 
    "histogram_width":[0],
    "histogram_min":[0], 
    "histogram_max":[0], 
    "histogram_number_of_peaks":[0],
    "histogram_number_of_zeroes":[0], 
    "histogram_mode":[0], 
    "histogram_mean":[0],
    "histogram_median":[0], 
    "histogram_variance":[0], 
    "histogram_tendency":[0],
}


@app.route('/')
def home_page():
    return render_template("home.html")


@app.route('/predict', methods=['POST','GET'])
def predict():

    input_data = {
    "baseline value":[0], 
    "accelerations":[0],
    "fetal_movement":[0],
    "uterine_contractions":[0], 
    "light_decelerations":[0], 
    "severe_decelerations":[0],
    "prolongued_decelerations":[0], 
    "abnormal_short_term_variability":[0],
    "mean_value_of_short_term_variability":[0],
    "percentage_of_time_with_abnormal_long_term_variability":[0],
    "mean_value_of_long_term_variability":[0], 
    "histogram_width":[0],
    "histogram_min":[0], 
    "histogram_max":[0], 
    "histogram_number_of_peaks":[0],
    "histogram_number_of_zeroes":[0], 
    "histogram_mode":[0], 
    "histogram_mean":[0],
    "histogram_median":[0], 
    "histogram_variance":[0], 
    "histogram_tendency":[0],
    }

    prediction = ""
    
    if request.method == "POST":
        input_data["baseline value"] = float(request.form['baseline_value'])
        input_data["accelerations"] = float(request.form['accelerations'])
        input_data["fetal_movement"] = float(request.form['fetal_movement'])
        input_data["uterine_contractions"] = float(request.form['uterine_contractions'])
        input_data["light_decelerations"] = float(request.form['light_decelerations'])
        input_data["severe_decelerations"] = float(request.form['severe_decelerations'])
        input_data["prolongued_decelerations"] = float(request.form['prolongued_decelerations'])
        input_data["abnormal_short_term_variability"] = float(request.form['abnormal_short_term_variability'])
        input_data["mean_value_of_short_term_variability"] = float(request.form['mean_value_of_short_term_variability'])
        input_data["percentage_of_time_with_abnormal_long_term_variability"] = float(request.form['percentage_of_time_with_abnormal_long_term_variability'])
        input_data["mean_value_of_long_term_variability"] = float(request.form['mean_value_of_long_term_variability'])
        input_data["histogram_width"] = float(request.form['histogram_width'])
        input_data["histogram_min"] = float(request.form['histogram_min'])
        input_data["histogram_max"] = float(request.form['histogram_max'])
        input_data["histogram_number_of_peaks"] = float(request.form['histogram_number_of_peaks'])
        input_data["histogram_number_of_zeroes"] = float(request.form['histogram_number_of_zeroes'])
        input_data["histogram_mode"] = float(request.form['histogram_mode'])
        input_data["histogram_mean"] = float(request.form['histogram_mean'])
        input_data["histogram_median"] = float(request.form['histogram_median'])
        input_data["histogram_variance"] = float(request.form['histogram_variance'])
        input_data["histogram_tendency"] = float(request.form['histogram_tendency'])

        data = pd.DataFrame(input_data, index=[0])

        loaded_model = joblib.load('model/model.pkl')
        prediction = loaded_model.predict(data)

        return render_template('after.html', data=int(prediction))



if __name__ == '__main__':
    app.run(debug=True)