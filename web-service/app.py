"""app.py

Streamlit single-page app: loads the trained model, collects input
parameters from the user, and displays the predicted value.
"""

import joblib
import streamlit as st

MODEL_PATH = "model.pkl"  # drop the exported model file here


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Model Predictor")

# TODO: replace these placeholder inputs with the project's real features
x1 = st.number_input("Feature 1", value=0.0)
x2 = st.number_input("Feature 2", value=0.0)

if st.button("Predict"):
    try:
        model = load_model()
        result = model.predict([[x1, x2]])
        st.success(f"Predicted value: {result[0]}")
    except FileNotFoundError:
        st.error(
            f"Model file '{MODEL_PATH}' not found. "
            "Export the trained model here before running the app."
        )
