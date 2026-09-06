import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Model Predictor", layout="centered")


@st.cache_resource
def load_model():
    with open("face_model.pkl", "rb") as file:
        return pickle.load(file)


model = load_model()

st.title("Model Prediction App")
st.write("Enter the input feature values below to generate a prediction.")

# Dynamically determine feature count from the loaded model
num_features = getattr(model, "n_features_in_", 4)

# Create input form with dynamic columns
inputs = []
cols = st.columns(2 if num_features <= 6 else 3)

for idx in range(num_features):
    with cols[idx % len(cols)]:
        val = st.number_input(
            label=f"Feature {idx + 1}", value=0.0, step=0.1, format="%.4f"
        )
        inputs.append(val)

if st.button("Predict", type="primary"):
    feature_array = np.array(inputs).reshape(1, -1)

    prediction = model.predict(feature_array)
    st.success(f"**Prediction:** {prediction[0]}")

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(feature_array)[0]
            classes = getattr(
                model, "classes_", [f"Class {i}" for i in range(len(probabilities))]
            )
            prob_dict = {
                str(cls): f"{prob * 100:.2f}%"
                for cls, prob in zip(classes, probabilities)
            }
            st.write("**Prediction Probabilities:**")
            st.json(prob_dict)
        except Exception:
            pass
