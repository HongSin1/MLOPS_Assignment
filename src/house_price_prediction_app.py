import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import plotly.express as px

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

# Function to calculate confidence based on similarity to training data
def calculate_confidence(input_df):
    # Create a simple confidence score based on how close the values are to the median values in the training data
    confidence_scores = []
    
    for idx, row in input_df.iterrows():
        feature_confidence = []
        
        # Check numerical features
        for feature in numerical_features:
            if feature in row:
                # Calculate how far value is from the median (normalized)
                feature_min = input_data_df[feature].min()
                feature_max = input_data_df[feature].max()
                feature_median = input_data_df[feature].median()
                feature_range = feature_max - feature_min
                
                if feature_range > 0:
                    normalized_distance = abs(row[feature] - feature_median) / feature_range
                    # Convert to confidence (closer to median = higher confidence)
                    feature_conf = max(0, 1 - normalized_distance)
                    feature_confidence.append(feature_conf)
        
        # Calculate average confidence across all features
        if feature_confidence:
            avg_confidence = sum(feature_confidence) / len(feature_confidence)
            confidence_scores.append(min(100, max(0, avg_confidence * 100)))
        else:
            confidence_scores.append(50)  # Default confidence
    
    return confidence_scores

# Function to create visualization of key factors
def create_feature_importance_chart(input_df):
    # Calculate correlation with common house price factors
    comparable_houses = input_data_df.copy()
    
    # Create a figure showing key metrics comparison
    fig = plt.figure(figsize=(10, 5))
    
    # Compare input values against averages for key metrics
    key_metrics = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea']
    avg_values = [input_data_df[metric].mean() for metric in key_metrics]
    input_values = [input_df[metric].iloc[0] for metric in key_metrics]
    
    x = np.arange(len(key_metrics))
    width = 0.35
    
    plt.bar(x - width/2, avg_values, width, label='Average')
    plt.bar(x + width/2, input_values, width, label='This Property')
    
    plt.ylabel('Value')
    plt.title('Property Comparison: Your Property vs Average')
    plt.xticks(x, key_metrics)
    plt.legend()
    
    return fig

# Function to create price range visualization
def create_price_range_visual(predicted_price, confidence):
    # Create a confidence interval based on the confidence score
    confidence_factor = (100 - confidence) / 100
    lower_bound = predicted_price * (1 - confidence_factor)
    upper_bound = predicted_price * (1 + confidence_factor)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Create a price range visualization
    ax.axvline(x=predicted_price, color='green', linestyle='-', linewidth=2, label='Predicted Price')
    ax.axvspan(lower_bound, upper_bound, alpha=0.2, color='green', label='Confidence Interval')
    
    # Add price markers
    ax.scatter([predicted_price], [1], color='green', s=100, zorder=5)
    ax.scatter([lower_bound, upper_bound], [1, 1], color='green', alpha=0.5, s=50)
    
    # Add text annotations
    ax.text(predicted_price, 1.1, f'${predicted_price:,.0f}', ha='center')
    ax.text(lower_bound, 0.9, f'${lower_bound:,.0f}', ha='center')
    ax.text(upper_bound, 0.9, f'${upper_bound:,.0f}', ha='center')
    
    ax.set_yticks([])
    ax.set_title('Predicted Price Range')
    ax.set_xlabel('Price ($)')
    ax.legend(loc='upper right')
    
    return fig

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
            
            # Calculate confidence
            confidence = calculate_confidence(input_df)[0]
            
            # Display prediction with confidence
            st.success(f'🏡 Predicted House Price: **${predicted_value:,.2f}**')
            
            # Display confidence gauge
            st.subheader("Prediction Confidence")
            st.progress(int(confidence))
            st.write(f"Confidence Rating: {confidence:.1f}%")
            
            # Create tabs for visualizations
            viz_tab1, viz_tab2 = st.tabs(["Price Range", "Property Comparison"])
            
            with viz_tab1:
                # Show price range visualization
                price_range_fig = create_price_range_visual(predicted_value, confidence)
                st.pyplot(price_range_fig)
                
                # Add explanation for confidence
                st.info("The confidence interval represents the potential price range based on the prediction confidence. Lower confidence results in a wider range.")
            
            with viz_tab2:
                # Show feature comparison
                feature_fig = create_feature_importance_chart(input_df)
                st.pyplot(feature_fig)
                
                # Add explanation
                st.info("This chart compares your property's key metrics against the average in our dataset. Significant differences may affect the prediction accuracy.")
        
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
                
                # Make predictions
                batch_predictions = model.predict(batch_data)
                batch_data['Predicted Price'] = np.expm1(batch_predictions)
                
                # Calculate confidence for each prediction
                batch_data['Confidence (%)'] = calculate_confidence(batch_data)
                
                st.success("✅ Predictions completed!")
                st.dataframe(batch_data[['Predicted Price', 'Confidence (%)']].head())
                
                # Visualizations for batch data
                st.subheader("Batch Prediction Visualizations")
                
                viz_tab1, viz_tab2 = st.tabs(["Price Distribution", "Confidence Analysis"])
                
                with viz_tab1:
                    # Create histograms of predicted prices
                    fig = px.histogram(batch_data, x='Predicted Price', 
                                      title='Distribution of Predicted House Prices',
                                      labels={'Predicted Price': 'Predicted Price ($)'})
                    st.plotly_chart(fig)
                
                with viz_tab2:
                    # Create scatter plot of price vs confidence
                    fig = px.scatter(batch_data, x='Predicted Price', y='Confidence (%)', 
                                    title='Prediction Confidence vs Price',
                                    labels={'Predicted Price': 'Predicted Price ($)', 
                                            'Confidence (%)': 'Confidence Level (%)'})
                    st.plotly_chart(fig)
                
                # Downloadable CSV with predictions
                csv_output = batch_data.to_csv(index=False)
                st.download_button("📥 Download Predictions", data=csv_output, file_name="predictions.csv", mime='text/csv')
            except Exception as e:
                st.error(f"❌ Error during batch prediction: {str(e)}")
