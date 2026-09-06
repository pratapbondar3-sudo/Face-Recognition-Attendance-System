from pathlib import Path
import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Face Recognition Predictor", layout="centered")


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "face_model.pkl"
    with open(model_path, "rb") as file:
        data = pickle.load(file)

    # Extract the classifier if the pickle is a dictionary
    if isinstance(data, dict):
        return data.get("classifier", data)
    return data


model = load_model()

st.title("Face Recognition Attendance System")
st.write("Enter feature values or face embeddings to make a prediction.")

num_features = getattr(model, "n_features_in_", 128)

inputs = []
cols = st.columns(2 if num_features <= 6 else 4)

for idx in range(num_features):
    with cols[idx % len(cols)]:
        val = st.number_input(
            label=f"Dim {idx + 1}", value=0.0, step=0.01, format="%.4f"
        )
        inputs.append(val)

if st.button("Predict", type="primary"):
    feature_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(feature_array)
    st.success(f"**Identified Person:** {prediction[0]}")

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
