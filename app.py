import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- STEP 1: LOAD MODELS & SCALER ---
@st.cache_resource
def load_assets():
    regressor = joblib.load('student_score_regressor.pkl')
    classifier = joblib.load('student_grade_classifier.pkl')
    scaler = joblib.load('data_scaler.pkl')
    return regressor, classifier, scaler

reg_model, clf_model, data_scaler = load_assets()

# --- STEP 2: UI SETUP ---
st.set_page_config(page_title="Student Performance AI", layout="wide")
st.title("🎓 Student Performance Predictor")
st.markdown("Enter student details to calculate predicted Exam Scores and Grade Categories for a subject.")

# Sidebar for choosing prediction type
prediction_mode = st.sidebar.radio("Select Prediction Goal:", ["Exam Score (Regression)", "Grade Category (Classification)"])

# --- STEP 3: FEATURE INPUTS ---
# We must follow the EXACT order of columns in your X (df.drop('Exam_Score', axis=1))
st.header("📝 Student Profile Factors")

with st.expander("Academic & Study Habits", expanded=True):
    col1, col2, col3 = st.columns(3)
    hours_studied = col1.number_input("Hours Studied per week", 0, 100, 20)
    attendance = col2.slider("Attendance %", 0, 100, 85)
    tutoring = col3.number_input("Tutoring Sessions per month", 0, 10, 2)
    prev_scores = col1.number_input("Previous Scores (%)", 0, 100, 75)
    motivation = col2.selectbox("Motivation Level (0-Low, 2-High)", [0, 1, 2], index=1)

with st.expander("School & Environment"):
    col4, col5, col6 = st.columns(3)
    teacher_qual = col4.number_input("Teacher Quality (0.0 - 10.0)", 0.0, 10.0, 7.5)
    peer_influence = col5.selectbox("Peer Influence (-1 to 1)", [-1, 0, 1], index=1)
    distance = col6.number_input("Distance from Home (0-50)", 0.0, 50.0, 5.0)
    internet = col4.radio("Internet Access", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    school_pvt = col5.checkbox("Is Private School?")
    resources = col6.selectbox("Access to Resources (0-3)", [0, 1, 2, 3], index=2)

with st.expander("Personal & Lifestyle"):
    col7, col8, col9 = st.columns(3)
    sleep = col7.number_input("Sleep Hours (per night)", 0, 15, 7)
    phys_activity = col8.number_input("Physical Activity (Hours per week)", 0, 20, 3)
    extra_curr = col9.radio("Extracurricular Activities", [0, 1])
    disability = col7.radio("Learning Disabilities", [0, 1])
    parent_inv = col8.selectbox("Parental Involvement (0-3)", [0, 1, 2, 3])
    income = col9.selectbox("Family Income (0-Low, 2-High)", [0, 1, 2])

with st.expander("Demographics & Parent Education"):
    col10, col11 = st.columns(2)
    gender = col10.radio("Gender", ["Male", "Female"])
    p_edu = col11.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])

# --- STEP 4: DATA PROCESSING ---
if st.button("🚀 Predict Performance"):
    # Create the One-Hot Encoded variables based on user selection
    # School_Type_Private / Public
    s_pvt = 1 if school_pvt else 0
    s_pub = 0 if school_pvt else 1
    
    # PEL (Parental Education Level)
    pel_clg = 1 if p_edu == "College" else 0
    pel_hs = 1 if p_edu == "High School" else 0
    pel_pg = 1 if p_edu == "Postgraduate" else 0
    
    # Gender
    g_fem = 1 if gender == "Female" else 0
    g_male = 1 if gender == "Male" else 0

    # ARRANGE FEATURES IN THE EXACT ORDER OF YOUR X_TRAIN (23 Columns total)
    input_data = [
        hours_studied, attendance, parent_inv, resources, extra_curr,
        sleep, prev_scores, motivation, internet, tutoring,
        income, teacher_qual, peer_influence, phys_activity,
        disability, distance, # Note: Check your exact order here!
        s_pvt, s_pub, pel_clg, pel_hs, pel_pg, g_fem, g_male
    ]
    
    # Convert to array and Reshape
    features_array = np.array(input_data).reshape(1, -1)
    
    # Apply the Scaler
    scaled_data = data_scaler.transform(features_array)

    if prediction_mode == "Exam Score (Regression)":
        res = reg_model.predict(scaled_data)
        st.success(f"### Predicted Exam Score: {res[0]:.2f}")
    else:
        res = clf_model.predict(scaled_data)
        st.info(f"### Predicted Grade: {res[0]}")