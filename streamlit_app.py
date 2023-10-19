import joblib
import pandas as pd
import streamlit as st
from pycaret.regression import *


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

def predict_health(input_data):
    data = pd.DataFrame.from_dict(input_data)

    loaded_model = joblib.load('model.pkl')
    prediction = loaded_model.predict(data)

    return class_list[int(prediction)-1]

st.set_page_config(
    page_title="Fetal Health Prediction System",
    page_icon=":baby:",
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a page",
    ["Home", "Team Members"],
)

if __name__ == "__main__":

    if app_mode == "Home":
        st.write("<h1>Fetal Health Prediction System</h1><br>", unsafe_allow_html=True)

        left, mid, right = st.columns(3)

        with left:
            baseline_value = st.number_input("Baseline Value", min_value=-100.0, max_value=100.0, step=0.5, value=0.0)
            accelerations = st.number_input("Accelerations", min_value=-100.0, max_value=100.0, step=0.1, value=0.0)
            fetal_movement = st.number_input("Fetal Movement", min_value=-100.0, max_value=1050.0, step=50.0, value=    0.0)
            uterine_contractions = st.number_input("Uterine Contractions", min_value=-100.0, max_value=5.0, step=0.1, value=0.0)
            light_decelerations = st.number_input("Light Decelerations", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            severe_decelerations = st.number_input("Severe Decelerations", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            prolongued_decelerations = st.number_input("Prolongued Decelerations", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)

        with mid:
            abnormal_short_term_variability = st.number_input("Abnormal Short Term Variability", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            mean_value_of_short_term_variability = st.number_input("Mean Value of Short Term Variability", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            percentage_of_time_with_abnormal_long_term_variability = st.number_input("Percentage of time with abnormal long term variability", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            mean_value_of_long_term_variability = st.number_input("Mean Value Of Long Term Variability", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            histogram_width = st.number_input("Histogram Width", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            histogram_min = st.number_input("Histogram Minimum", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            histogram_max = st.number_input("Histogram Max", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            
        with right:
            histogram_number_of_peaks = st.number_input("Histogram Peaks", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            histogram_number_of_zeroes = st.number_input("Histogram Zeroes", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            histogram_mode = st.number_input("Histogram Mode", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            histogram_mean = st.number_input("Histogram Mean", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            histogram_median = st.number_input("Histogram Median", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            histogram_variance = st.number_input("Histogram Variance", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            histogram_tendency = st.number_input("Histogram Tendency", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)


        input_data["baseline value"] = [baseline_value]
        input_data["accelerations"] = [accelerations]
        input_data["fetal_movement"] = [fetal_movement]
        input_data["uterine_contractions"] = [uterine_contractions]
        input_data["light_decelerations"] = [light_decelerations]
        input_data["severe_decelerations"] = [severe_decelerations]
        input_data["prolongued_decelerations"] = [prolongued_decelerations]
        input_data["abnormal_short_term_variability"] = [abnormal_short_term_variability]
        input_data["mean_value_of_short_term_variability"] = [mean_value_of_short_term_variability]
        input_data["percentage_of_time_with_abnormal_long_term_variability"] = [percentage_of_time_with_abnormal_long_term_variability]
        input_data["mean_value_of_long_term_variability"] = [mean_value_of_long_term_variability]
        input_data["histogram_width"] = [histogram_width]
        input_data["histogram_min"] = [histogram_min]
        input_data["histogram_max"] = [histogram_max]
        input_data["histogram_number_of_peaks"] = [histogram_number_of_peaks]
        input_data["histogram_number_of_zeroes"] = [histogram_number_of_zeroes]
        input_data["histogram_mode"] = [histogram_mode]
        input_data["histogram_mean"] = [histogram_mean]
        input_data["histogram_median"] = [histogram_median]
        input_data["histogram_variance"] = [histogram_variance]
        input_data["histogram_tendency"] = [histogram_tendency]


        button = st.button("Predict")

        if button and len(input_data.values())>20:
            predicted_class = predict_health(input_data=input_data)
            st.info(f"##### Predicted Class: {predicted_class}")
