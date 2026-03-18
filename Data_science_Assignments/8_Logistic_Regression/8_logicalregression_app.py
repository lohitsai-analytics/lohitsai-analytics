import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Load saved artefacts ────────────────────────────────────────────────────
model       = pickle.load(open("log_reg_assignment.pkl",          "rb"))
scaler      = pickle.load(open("std_scaler_log_reg_assignment.pkl","rb"))
feature_cols = pickle.load(open("feature_cols.pkl",               "rb"))

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter the patient's details below and click **Predict** to get a diagnosis.")

st.divider()

# ── Input form ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    pregnancies    = st.number_input("Pregnancies",           min_value=0,   max_value=20,   value=1,   step=1)
    glucose        = st.number_input("Glucose (mg/dL)",       min_value=0,   max_value=300,  value=120, step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)",   min_value=0,   max_value=100,  value=20,  step=1)

with col2:
    insulin        = st.number_input("Insulin (mu U/ml)",     min_value=0,   max_value=900,  value=80,  step=1)
    dpf            = st.number_input("Diabetes Pedigree Fn",  min_value=0.0, max_value=3.0,  value=0.5, step=0.01)
    age            = st.number_input("Age (years)",           min_value=1,   max_value=120,  value=30,  step=1)

st.divider()

# ── Prediction ───────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True):
    # Build input DataFrame matching training column order
    input_data = pd.DataFrame([[pregnancies, glucose, skin_thickness,
                                insulin, dpf, age]],
                              columns=feature_cols)

    # Scale the same columns that were scaled during training
    scale_cols = ["Glucose", "SkinThickness", "Insulin"]
    input_data[scale_cols] = scaler.transform(input_data[scale_cols])

    # Predict
    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **Diabetic**  —  Probability: {probability:.1%}")
        st.markdown("The model predicts this patient is **likely diabetic**. "
                    "Please consult a medical professional for further evaluation.")
    else:
        st.success(f"✅ **Non-Diabetic**  —  Probability of Diabetes: {probability:.1%}")
        st.markdown("The model predicts this patient is **likely not diabetic**. "
                    "Maintain a healthy lifestyle and schedule regular check-ups.")

    st.divider()
    st.subheader("Input Summary")
    display = pd.DataFrame({
        "Feature": ["Pregnancies","Glucose","Skin Thickness","Insulin",
                    "Diabetes Pedigree Function","Age"],
        "Value"  : [pregnancies, glucose, skin_thickness, insulin, dpf, age]
    })
    st.dataframe(display, use_container_width=True)