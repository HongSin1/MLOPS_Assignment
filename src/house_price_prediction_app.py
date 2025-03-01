import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Get the current working directory
base_dir = os.path.abspath(os.path.dirname(__file__))

# Define relative paths for model and CSV files
model_path = os.path.join(base_dir, '..', 'models', 'pipeline1.pkl')
csv_path = os.path.join(base_dir, '..', 'data', 'processed', 'input_data_for_predictions.csv')

# Load the trained model
model = joblib.load(model_path)

# Load the CSV file containing input data
input_data_df = pd.read_csv(csv_path)

# Define numerical and categorical features as in your PyCaret setup
numerical_features = ['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt']
categorical_features = ['Suburb', 'Type', 'Method', 'Region', 'CouncilArea']
restricted_columns = ['Rooms', 'Bedroom2', 'Car', 'Bathroom', 'YearBuilt']

# Streamlit UI components
st.title('🏡 House Price Prediction Web App')
st.write("Provide details of the property below, or upload a CSV file for batch predictions.")

# Single or batch input selection
prediction_mode = st.radio("Choose input method:", ['Single Entry', 'Batch CSV Upload'])

if prediction_mode == 'Single Entry':
    col1, col2 = st.columns(2)
    input_values = {}
    
    for idx, column in enumerate(input_data_df.columns):
        if input_data_df[column].dtype == 'object':
            with col2:
                unique_values = input_data_df[column].dropna().unique()
                descriptions = {
                    'Suburb': '🏙️ Select the suburb where the house is located.',
                    'Type': '🏠 Choose the type of property (House, Unit, Townhouse, etc.).',
                    'Method': '💰 Select the sales method (Auction, Private, etc.).',
                    'Region': '🌍 Choose the region where the house is situated.',
                    'CouncilArea': '🏛️ Select the governing council area for the property.',
                }
                input_values[column] = st.selectbox(descriptions.get(column, f'Select {column}'), options=unique_values)
        elif column in restricted_columns:
            with col1:
                unique_values = sorted(input_data_df[column].dropna().unique())
                unique_values = [int(value) for value in unique_values]
                descriptions = {
                    'Rooms': '🛏️ Select the total number of rooms in the house.',
                    'Bedroom2': '🛌 Choose the number of secondary bedrooms.',
                    'Car': '🚗 Select the number of car spaces available.',
                    'Bathroom': '🛁 Choose the number of bathrooms.',
                    'YearBuilt': '📅 Choose the year the house was built.',
                }
                input_values[column] = st.selectbox(descriptions.get(column, f'Select {column} (Numeric - Restricted)'), options=unique_values)
        else:
            with col1:
                min_val = float(input_data_df[column].min())
                max_val = float(input_data_df[column].max())
                descriptions = {
                    'Distance': '📏 Enter the distance (in km) from the Melbourne CBD.',
                    'Landsize': '🌿 Enter the total land size in square meters.',
                    'BuildingArea': '🏗️ Enter the total building area in square meters.',
                }
                input_values[column] = st.number_input(descriptions.get(column, f'Enter {column} (Numeric)'), min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2.0)
    
    if st.button('📊 Make Prediction'):
        input_df = pd.DataFrame([input_values])
        for col in numerical_features:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        for col in categorical_features:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str)
        try:
            prediction = model.predict(input_df)
            predicted_value = np.expm1(prediction[0])
            st.success(f'🏡 Predicted House Price: **${predicted_value:,.2f}**')
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")

elif prediction_mode == 'Batch CSV Upload':
    uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview:")
        st.dataframe(batch_data.head())
        
        if st.button("📊 Predict for Batch"):
            try:
                for col in numerical_features:
                    if col in batch_data.columns:
                        batch_data[col] = pd.to_numeric(batch_data[col], errors='coerce')
                for col in categorical_features:
                    if col in batch_data.columns:
                        batch_data[col] = batch_data[col].astype(str)
                batch_predictions = model.predict(batch_data)
                batch_data['Predicted Price'] = np.expm1(batch_predictions)
                st.success("✅ Predictions completed!")
                st.write(batch_data[['Predicted Price']].head())
                
                # Downloadable CSV with predictions
                csv_output = batch_data.to_csv(index=False)
                st.download_button("📥 Download Predictions", data=csv_output, file_name="predictions.csv", mime='text/csv')
            except Exception as e:
                st.error(f"❌ Error during batch prediction: {str(e)}")
