# Import necessary libraries to run app
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import plotly.express as px
import yaml
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra

# Clear the GlobalHydra singleton before initializing
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# Define the base directory 
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define relative path to the config directory
config_path = os.path.join(base_dir, '..', 'config') 

# Define configuration class structure
@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(cfg: DictConfig):
    """
    Main function to run the Streamlit application.
    """
    # Convert OmegaConf to regular dict for easier access
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Load the CSS
    load_css(config["app"]["css_path"])
    
    # Load the model once
    model = load_model(config)
    
    # App title and description
    st.title("🌾 Wheat Kernel Classification App")
    st.markdown(config["app"]["description"])
    
    # Create tabs for different functionality
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Single Kernel Prediction")
        st.markdown("Enter the geometric parameters of a wheat kernel to classify its variety.")
        
        # Create two columns for input fields
        col1, col2 = st.columns(2)
        
        # Get input field configurations
        input_fields = config["input_fields"]
        
        # Input fields in the first column
        with col1:
            area = st.number_input(
                input_fields["area"]["label"],
                min_value=input_fields["area"]["min_value"],
                max_value=input_fields["area"]["max_value"],
                value=input_fields["area"]["default_value"],
                help=input_fields["area"]["help"]
            )
            
            perimeter = st.number_input(
                input_fields["perimeter"]["label"],
                min_value=input_fields["perimeter"]["min_value"],
                max_value=input_fields["perimeter"]["max_value"],
                value=input_fields["perimeter"]["default_value"],
                help=input_fields["perimeter"]["help"]
            )
            
            compactness = st.number_input(
                input_fields["compactness"]["label"],
                min_value=input_fields["compactness"]["min_value"],
                max_value=input_fields["compactness"]["max_value"],
                value=input_fields["compactness"]["default_value"],
                help=input_fields["compactness"]["help"]
            )
            
            length = st.number_input(
                input_fields["length"]["label"],
                min_value=input_fields["length"]["min_value"],
                max_value=input_fields["length"]["max_value"],
                value=input_fields["length"]["default_value"],
                help=input_fields["length"]["help"]
            )
            
        # Input fields in the second column
        with col2:
            width = st.number_input(
                input_fields["width"]["label"],
                min_value=input_fields["width"]["min_value"],
                max_value=input_fields["width"]["max_value"],
                value=input_fields["width"]["default_value"],
                help=input_fields["width"]["help"]
            )
            
            asymmetry = st.number_input(
                input_fields["asymmetry"]["label"],
                min_value=input_fields["asymmetry"]["min_value"],
                max_value=input_fields["asymmetry"]["max_value"],
                value=input_fields["asymmetry"]["default_value"],
                help=input_fields["asymmetry"]["help"]
            )
            
            groove = st.number_input(
                input_fields["groove"]["label"],
                min_value=input_fields["groove"]["min_value"],
                max_value=input_fields["groove"]["max_value"],
                value=input_fields["groove"]["default_value"],
                help=input_fields["groove"]["help"]
            )
        
        # Button to make prediction
        if st.button("Classify Wheat Kernel", key="single_predict"):
            with st.spinner("Classifying..."):

                # Validate all inputs
                all_valid = all([
                    validate_input(area, input_fields["area"]["min_value"], input_fields["area"]["max_value"], input_fields["area"]["label"]),
                    validate_input(perimeter, input_fields["perimeter"]["min_value"], input_fields["perimeter"]["max_value"], input_fields["perimeter"]["label"]),
                    validate_input(compactness, input_fields["compactness"]["min_value"], input_fields["compactness"]["max_value"], input_fields["compactness"]["label"]),
                    validate_input(length, input_fields["length"]["min_value"], input_fields["length"]["max_value"], input_fields["length"]["label"]),
                    validate_input(width, input_fields["width"]["min_value"], input_fields["width"]["max_value"], input_fields["width"]["label"]),
                    validate_input(asymmetry, input_fields["asymmetry"]["min_value"], input_fields["asymmetry"]["max_value"], input_fields["asymmetry"]["label"]),
                    validate_input(groove, input_fields["groove"]["min_value"], input_fields["groove"]["max_value"], input_fields["groove"]["label"])
                ])
                
                if all_valid and model:
                    # Create a dataframe with the input data
                    input_df = pd.DataFrame({
                        'Area': [area],
                        'Perimeter': [perimeter],
                        'Compactness': [compactness],
                        'Length': [length],
                        'Width': [width],
                        'AsymmetryCoeff': [asymmetry],
                        'Groove': [groove]
                    })
                    
                    # Make prediction
                    time.sleep(0.5)  
                    prediction, confidence, probabilities = predict(model, input_df, config)
                    
                    if prediction:
                        # Determine confidence level class for styling
                        conf_class = "high-conf" if confidence[0] > 0.8 else "medium-conf" if confidence[0] > 0.6 else "low-conf"
                        
                        # Display prediction result
                        st.markdown(f"""
                        <div class='prediction-box {conf_class}'>
                            <h3>Prediction Result</h3>
                            <p>The wheat kernel is classified as:</p>
                            <h2>{prediction[0]}</h2>
                            <p>Confidence: {confidence[0]*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create a bar chart for class probabilities
                        probs_df = pd.DataFrame({
                            'Wheat Type': list(config["wheat_types"].values()),
                            'Probability': probabilities[0]
                        })
                        
                        fig = px.bar(
                            probs_df,
                            x='Wheat Type',
                            y='Probability',
                            color='Probability',
                            color_continuous_scale='viridis',
                            title='Prediction Probabilities',
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Batch Prediction
    with tab2:
        st.header("Batch Prediction")
        st.markdown(f"""
        Upload a CSV file with multiple wheat kernel measurements for batch classification.
        
        **Required CSV format:**
        - The CSV must contain columns: {', '.join(config['required_columns'])}
        - Each row represents one wheat kernel
        """)
        
        # File uploader for CSV
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        # Sample CSV template for download
        st.markdown("#### Need a sample CSV template?")
        sample_data = pd.DataFrame(config["sample_data"])
        
        # Convert sample data to CSV for download
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="Download Sample CSV Template",
            data=csv,
            file_name="wheat_seeds_template.csv",
            mime="text/csv"
        )
        
        if uploaded_file is not None:
            # Process the uploaded file
            batch_data = process_batch_csv(uploaded_file, config)
            
            if batch_data is not None and not batch_data.empty:
                st.write(f"Loaded {len(batch_data)} samples for prediction")
                
                # Preview of the uploaded data
                with st.expander("Preview uploaded data"):
                    st.dataframe(batch_data.head(10))
                
                # Button to make batch predictions
                if st.button("Run Batch Classification", key="batch_predict"):
                    if model:
                        # Show progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Simulate progress for better UX
                        for i in range(101):
                            progress_bar.progress(i)
                            status_text.text(f"Processing: {i}%")
                            time.sleep(0.01)
                        
                        # Make predictions
                        predictions, confidence, probabilities = predict(model, batch_data, config)
                        
                        if predictions:
                            # Create a dataframe with the results
                            results_df = batch_data.copy()
                            results_df['Predicted_Type'] = predictions
                            results_df['Confidence'] = confidence
                            
                            # Display results
                            st.subheader("Batch Prediction Results")
                            st.dataframe(results_df)
                            
                            # Create visualisation for the batch results
                            st.subheader("Results Visualisation")
                            
                            # Count of predictions by variety
                            counts = pd.Series(predictions).value_counts()
                            fig1 = px.pie(
                                names=counts.index,
                                values=counts.values,
                                title="Distribution of Predicted Wheat Types",
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                            
                            # Download the results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Prediction Results",
                                data=csv,
                                file_name="wheat_classification_results.csv",
                                mime="text/csv"
                            )


# Load CSS file
def load_css(css_file):
    try:
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"{css_file} file not found. Please make sure it's in the same directory as this script.")

# Function to load the trained model
@st.cache_resource
def load_model(config):
    """
    Load the trained machine learning model.
    Uses st.cache_resource to avoid reloading the model on each rerun.
    
    Returns:
        The loaded ML pipeline
    """
    model_path = config["app"]["model_path"]
    try:
        # Try loading with pycaret first
        try:
            from pycaret.classification import load_model
            model = load_model(model_path)
            return model
        except:
            # Fall back to pickle if pycaret import fails
            with open(f'{model_path}.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
    except FileNotFoundError:
        st.error("Model file not found. Please check file path.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to make predictions
def predict(model, input_data, config):
    """
    Use the loaded model to make predictions.
    
    Args:
        model: The loaded ML pipeline
        input_data: DataFrame with features for prediction
        
    Returns:
        Predictions and confidence scores
    """
    try:
        # Get predictions
        predictions = model.predict(input_data)
        
        # Map numerical predictions to wheat varieties
        wheat_types = config["wheat_types"]
        prediction_names = [wheat_types.get(int(pred), f"Unknown ({pred})") for pred in predictions]
        
        # Use decision_function to get confidence scores
        decision_scores = model.decision_function(input_data)
        
        # Convert to probabilities using softmax
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1, keepdims=True)
            
        probabilities = softmax(decision_scores)
        
        # Get the confidence as the max probability
        confidence = np.max(probabilities, axis=1)
        
        return prediction_names, confidence, probabilities
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

# Function to validate input values
def validate_input(value, min_val, max_val, feature_name):
    """
    Validate that input values are within acceptable ranges.
    
    Args:
        value: The input value to validate
        min_val: Minimum acceptable value
        max_val: Maximum acceptable value
        feature_name: Name of the feature (for error messages)
        
    Returns:
        Boolean indicating if the value is valid
    """
    try:
        val = float(value)
        if val < min_val or val > max_val:
            st.warning(f"{feature_name} should be between {min_val} and {max_val}.")
            return False
        return True
    except ValueError:
        st.warning(f"{feature_name} must be a valid number.")
        return False

# Function to process batch prediction from CSV
def process_batch_csv(file, config):
    """
    Process a CSV file for batch prediction.
    
    Args:
        file: Uploaded CSV file
        
    Returns:
        Dataframe with the extracted features
    """
    try:
        # Load CSV file as a pandas dataframe
        df = pd.read_csv(file)
        
        # Verify required columns exist
        required_columns = config["required_columns"]
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.error(f"CSV file must contain columns: {', '.join(required_columns)}")
            return None
            
        # Select only the required columns and in the correct order
        df = df[required_columns]
        
        # Check for missing values and remove them
        if df.isnull().any().any():
            st.warning("CSV contains missing values. Rows with missing values will be dropped.")
            df = df.dropna()
            
        return df
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        return None

if __name__ == "__main__":
    main()