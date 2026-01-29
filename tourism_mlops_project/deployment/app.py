import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

MODEL_ID = "PalkiiS/tourism-package-prediction-model"

model_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="best_model.joblib",
    repo_type="model"
)

model = joblib.load(model_path)

st.title("Wellness Tourism Package Prediction")

st.write("Enter customer details to predict purchase likelihood")

input_data = {
    "Age": st.number_input("Age", 18, 100, 30),
    "TypeofContact": st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"]),
    "CityTier": st.selectbox("City Tier", [1, 2, 3]),
    "DurationOfPitch": st.number_input("Duration of Pitch", 0, 60, 15),
    "NumberOfPersonVisiting": st.number_input("Number of Persons Visiting", 1, 10, 2),
    "NumberOfFollowups": st.number_input("Number of Followups", 0, 10, 1),
    "PreferredPropertyStar": st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5]),
    "MaritalStatus": st.selectbox("Marital Status", ["Single", "Married", "Divorced"]),
    "NumberOfTrips": st.number_input("Number of Trips", 0, 20, 2),
    "Passport": st.selectbox("Passport", [0, 1]),
    "PitchSatisfactionScore": st.selectbox("Pitch Satisfaction Score", [1,2,3,4,5]),
    "OwnCar": st.selectbox("Own Car", [0, 1]),
    "NumberOfChildrenVisiting": st.number_input("Children Visiting", 0, 5, 0),
    "Designation": st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]),
    "MonthlyIncome": st.number_input("Monthly Income", 1000, 1000000, 30000)
}

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase"
    st.success(result)
