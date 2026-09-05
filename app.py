import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="KNN Prediction Portal",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8533 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
        transform: translateY(-2px);
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

# Header
st.title("⚡ Classifier Prediction Portal")
st.caption("Adjust the parameters on the left and trigger inference below.")
st.write("---")

# Sidebar - Feature Inputs
st.sidebar.header("🔧 Feature Configuration")
st.sidebar.markdown("Customize your 4 input parameters:")

feat_1 = st.sidebar.slider("Feature 1", min_value=-5.0, max_value=5.0, value=0.0, step=0.01)
feat_2 = st.sidebar.slider("Feature 2", min_value=-5.0, max_value=5.0, value=0.0, step=0.01)
feat_3 = st.sidebar.slider("Feature 3", min_value=-5.0, max_value=5.0, value=0.0, step=0.01)
feat_4 = st.sidebar.slider("Feature 4", min_value=-5.0, max_value=5.0, value=0.0, step=0.01)

# Main Dashboard View
st.subheader("Current Inputs")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Feature 1", f"{feat_1:.2f}")
col2.metric("Feature 2", f"{feat_2:.2f}")
col3.metric("Feature 3", f"{feat_3:.2f}")
col4.metric("Feature 4", f"{feat_4:.2f}")

st.write("")

# Prediction Action
if st.button("🚀 Run Prediction"):
    with st.spinner("Analyzing nearest neighbors..."):
        input_data = np.array([[feat_1, feat_2, feat_3, feat_4]])
        
        prediction = model.predict(input_data)[0]
        
        # Check if probability output is supported
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

    # Celebration visual effect
    st.balloons()
    
    # Result Display Card
    st.markdown(
        f"""
        <div class="prediction-card">
            <h4 style="color: #9E9E9E; margin-bottom: 0.5rem;">PREDICTED CLASS</h4>
            <h1 style="color: #00FFA3; margin: 0; font-size: 2.5rem;">{prediction}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display class probabilities if available
    if probabilities is not None:
        st.write("")
        st.write("##### Prediction Confidence")
        for cls, prob in zip(model.classes_, probabilities):
            st.write(f"**Class {cls}**")
            st.progress(float(prob))
