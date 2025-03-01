# Import necessary libraries
import streamlit as st
import os
from hydra.core.global_hydra import GlobalHydra

# Clear GlobalHydra if it's initialized
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# Define the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Set streamlit page configuration
st.set_page_config(
    page_title="Multifunctional Application",
    page_icon="🔎",
    layout="wide"
)

# Dictionary of available apps
apps = {
    "🏠 Home Page": None,
    "🏡 House Price Prediction": os.path.join(base_dir, "house_price_prediction_app.py"),
    "🌾 Wheat Kernel Classification": os.path.join(base_dir, "wheat_classification_app.py")
}

# Sidebar Navigation
st.sidebar.title("Navigation")
selected_app = st.sidebar.radio("Please choose an application", list(apps.keys()))

# Landing Page Content
if selected_app == "🏠 Home Page":
    st.title("🤗 Welcome To Our Multifunctional Application!")
    st.markdown("""
        ## About this Project
        This application demonstrates our implementation of Machine Learning Operations (MLOps) 
        practices for model development, deployment, and monitoring. Our team has built 2 different
        machine learning models for you to try out!
        
        ### Available Models:
        - **House Price Prediction**: Predict Melbourne property prices based on features like location, size, and amenities
        - **Wheat Kernel Classification**: Classify wheat kernels into Kama, Rosa or Canadian varieties based on geometric parameters
        
        Please use the sidebar to navigate between the different model applications! Thank you!
        """)
else:
    # Ensure app_path is not None before attempting to open the file
    app_path = apps[selected_app]
    if app_path:
        with open(app_path, encoding="utf-8") as f:
            exec(f.read(), globals())
    else:
        st.error("Selected app path is invalid.")
