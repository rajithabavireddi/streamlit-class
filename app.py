import streamlit as st
import pickle
import numpy as np

st.title("ML Prediction App")

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")

if st.button("Predict"):
    result = model.predict(np.array([[f1, f2]]))
    st.success(f"Prediction: {result[0]}")
