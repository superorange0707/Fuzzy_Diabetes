import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import io
import collections
import sys
import argparse
from utils import (
    load_models, calculate_bmi, estimate_dpf, 
    get_risk_category, get_prediction, get_factor_contributions
)

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Diabetes Risk Assessment App")
    parser.add_argument("--model", type=str, choices=["anfis", "rf", "knn", "svm"], 
                        default=None, help="Force use of a specific model")
    # Check if streamlit passed additional args after --
    if len(sys.argv) > 1 and sys.argv[1] == "--":
        args, _ = parser.parse_known_args(sys.argv[2:])
    else:
        args = parser.parse_args([])
    return args

args = parse_args()
force_model = args.model

# Page configuration
st.set_page_config(
    page_title="Fuzzy Diabetes Risk Assessment Tool",
    page_icon="🩺",
    layout="wide"
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_cached_models():
    return load_models()

models = load_cached_models()

# Verify models are usable
for model_key, model in models.items():
    if isinstance(model, dict) or isinstance(model, collections.OrderedDict):
        st.warning(f"⚠️ {model_key.upper()} model may not function correctly. Using fallback predictions.")

# Check for logo and create if not exists
logo_path = os.path.join(os.path.dirname(__file__), "fuzzy_diabetes_logo.png")
if not os.path.exists(logo_path):
    from logo_placeholder import create_logo
    create_logo()

# Sidebar
with st.sidebar:
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    else:
        st.image("https://i.ibb.co/yBqVHVz/fuzzy-diabetes-logo.png", width=150)
    
    st.title("Fuzzy Diabetes")
    st.markdown("### AI-Powered Diabetes Risk Assessment")
    
    st.markdown("---")
    
    # If a model is forced via command line, use it and disable the UI selection
    if force_model:
        st.markdown(f"**Using model: {force_model.upper()}**")
        selected_model = models[force_model]
        model_selection = f"{force_model.upper()} (Command Line Selected)"
    else:
        # Always use ANFIS model by default (no UI selection)
        selected_model = models["anfis"]
        st.markdown("**Using ANFIS Model (Best Accuracy: 76%)**")
        st.markdown("*This model combines fuzzy logic and neural networks for the most accurate predictions.*")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool uses fuzzy logic and machine learning to estimate diabetes risk based on key health metrics.
    
    Models are trained on the Pima Indians Diabetes Dataset with careful feature selection.
    """)
    
# Main content
st.title("Diabetes Risk Assessment")
st.markdown("Enter your health information to assess your diabetes risk")

st.markdown("### Disclaimer")
st.markdown("""
<div class="disclaimer">
This tool is for educational and informational purposes only. It does not constitute medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.
</div>
""", unsafe_allow_html=True)

# Input section
st.markdown("## Health Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Personal Information")
    
    # Pregnancies input
    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0,
        max_value=20,
        value=0,
        help="Enter the number of times you have been pregnant. Enter 0 if not applicable."
    )
    
    # BMI calculation options
    bmi_method = st.radio(
        "BMI Input Method",
        ["Calculate from height and weight", "Direct BMI input"]
    )
    
    if bmi_method == "Calculate from height and weight":
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        bmi = calculate_bmi(height, weight)
        st.markdown(f"**Calculated BMI:** {bmi}")
    else:
        bmi = st.number_input(
            "BMI (Body Mass Index)", 
            min_value=10.0, 
            max_value=50.0, 
            value=25.0,
            step=0.1,
            help="Normal range: 18.5-24.9, Overweight: 25-29.9, Obese: 30+"
        )

with col2:
    st.markdown("### Medical Information")
    
    # Glucose input with option
    has_glucose = st.checkbox(
        "I have my blood glucose data",
        value=False,
        help="2-hour plasma glucose concentration from an Oral Glucose Tolerance Test"
    )
    
    if has_glucose:
        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=50,
            max_value=300,
            value=120,
            help="Normal fasting: <100, Prediabetes: 100-125, Diabetes: >126"
        )
    else:
        st.info("Without glucose data, the prediction will be less accurate but still useful.")
        glucose = None
    
    # Family history for DPF estimation
    st.markdown("### Family History")
    st.markdown("This helps estimate your Diabetes Pedigree Function (DPF)")
    
    parent_diabetes = st.checkbox("Do any of your parents have diabetes?")
    sibling_diabetes = st.checkbox("Do any of your siblings have diabetes?")
    other_relatives = st.checkbox("Do any other close relatives have diabetes?")
    
    # Calculate estimated DPF
    dpf = estimate_dpf(parent_diabetes, sibling_diabetes, other_relatives)
    st.markdown(f"**Estimated Diabetes Pedigree Function:** {dpf:.3f}")
    
    # Option to directly enter DPF
    custom_dpf = st.checkbox("I know my exact DPF value")
    if custom_dpf:
        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=2.5,
            value=dpf,
            step=0.01,
            help="A function that scores likelihood of diabetes based on family history"
        )

# Batch upload option
st.markdown("## Batch Assessment (Optional)")
batch_upload = st.checkbox("I want to upload data for multiple people")

if batch_upload:
    st.markdown("Upload a CSV file with columns: Pregnancies, Glucose, BMI, DiabetesPedigreeFunction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            required_cols = ["Pregnancies", "Glucose", "BMI", "DiabetesPedigreeFunction"]
            
            # Check if required columns exist
            if all(col in batch_data.columns for col in required_cols):
                st.success(f"Successfully loaded data for {len(batch_data)} people")
                
                # Process batch data
                predictions = []
                for _, row in batch_data.iterrows():
                    features = [
                        row["Pregnancies"],
                        row["Glucose"],
                        row["BMI"],
                        row["DiabetesPedigreeFunction"]
                    ]
                    prob = get_prediction(selected_model, features)
                    risk_label, _ = get_risk_category(prob)
                    predictions.append({
                        "ID": _ + 1,
                        "Risk Probability": prob,
                        "Risk Category": risk_label
                    })
                
                results_df = pd.DataFrame(predictions)
                st.dataframe(results_df)
                
                # Download batch results option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="batch_diabetes_risk_results.csv",
                    mime="text/csv",
                )
            else:
                missing = [col for col in required_cols if col not in batch_data.columns]
                st.error(f"CSV is missing required columns: {', '.join(missing)}")
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# Analysis button
st.markdown("---")

analyze_button = st.button("Analyze Diabetes Risk")

if analyze_button:
    # Check if we need to proceed with missing glucose
    proceed = True
    if not has_glucose and glucose is None:
        proceed = st.warning("You're proceeding without glucose data. Results will be less accurate. Continue?")
        glucose = 120  # Use average value when missing
    
    if proceed:
        features = [
            pregnancies,
            glucose,
            bmi,
            dpf
        ]
        
        feature_names = ["Pregnancies", "Glucose", "BMI", "DiabetesPedigreeFunction"]
        
        # Get prediction
        probability = get_prediction(selected_model, features)
        risk_label, risk_class = get_risk_category(probability)
        
        # Get feature contributions
        contributions = get_factor_contributions(features, feature_names)
        
        # Display results
        st.markdown("## Risk Assessment Results")
        
        # Main risk result with colored container
        st.markdown(f"""
        <div class="result-container result-{risk_class}">
            <h3>Your Diabetes Risk: {risk_label}</h3>
            <p>Risk probability: {probability:.2f} (0-1 scale)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk meter visualization
        st.markdown("### Risk Meter")
        
        # Create a risk meter with Plotly
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Diabetes Risk"},
            gauge = {
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "gray"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 0.3], 'color': '#10B981'},
                    {'range': [0.3, 0.7], 'color': '#F59E0B'},
                    {'range': [0.7, 1], 'color': '#EF4444'}
                ],
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display contributing factors
        st.markdown("### Contributing Factors")
        
        # Create bar chart of factor contributions
        factor_df = pd.DataFrame({
            'Factor': list(contributions.keys()),
            'Impact': list(contributions.values())
        })
        
        fig_bar = px.bar(
            factor_df,
            x='Impact',
            y='Factor',
            orientation='h',
            color='Impact',
            color_continuous_scale=['green', 'yellow', 'red'],
            title="Relative Impact of Each Factor"
        )
        
        fig_bar.update_layout(
            xaxis_title="Relative Impact (higher values indicate higher risk contribution)",
            yaxis_title="",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Explanation text
        st.markdown("### Analysis Explanation")
        
        # Find top contributing factor
        top_factor = max(contributions.items(), key=lambda x: x[1])
        
        if not has_glucose:
            st.markdown("⚠️ **Note:** This assessment was performed without glucose data, which is a critical factor.")
            
        if top_factor[1] > 0.5:
            st.markdown(f"Your highest risk factor appears to be **{top_factor[0]}**.")
        
        # Personalized recommendations
        st.markdown("### Recommendations")
        
        st.markdown("""
        Based on your assessment, consider these general health recommendations:
        
        - Maintain a healthy diet rich in fruits, vegetables, and whole grains
        - Engage in regular physical activity (aim for 150 minutes/week)
        - Monitor your weight and blood sugar levels regularly
        - Schedule regular check-ups with your healthcare provider
        
        **Remember:** This tool is for informational purposes only. For medical advice, please consult a healthcare professional.
        """)
        
        # Generate PDF report
        st.markdown("### Save Your Results")
        
        # Create CSV data
        result_data = pd.DataFrame({
            'Metric': ['Risk Probability', 'Risk Category', 'Pregnancies', 'Glucose', 'BMI', 'Diabetes Pedigree Function'],
            'Value': [f"{probability:.2f}", risk_label, pregnancies, glucose, bmi, dpf]
        })
        
        csv = result_data.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="diabetes_risk_assessment.csv",
            mime="text/csv",
        )

# Footer
st.markdown("""
<div class="footer">
    <p>© 2023 Fuzzy Diabetes Project. All rights reserved.</p>
    <p>This tool is for educational purposes only and does not provide medical advice.</p>
</div>
""", unsafe_allow_html=True) 