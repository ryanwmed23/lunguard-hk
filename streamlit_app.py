"""
Streamlit App for Lung Cancer VOC Classifier
Based on Wang 2022 Study - 16 VOC Biomarkers
"""

import streamlit as st
import pandas as pd
import numpy as np
from voc_classifier import predict_risk, VOC_NAMES, VOC_MEDIANS_LC
import os

# Page configuration
st.set_page_config(
    page_title="Lung Cancer VOC Classifier",
    page_icon="🫁",
    layout="wide"
)

# Title
st.title("🫁 Lung Cancer VOC Classifier")
st.markdown("**Based on Wang 2022 Study - 16 VOC Biomarkers**")
st.markdown("---")

# Check if model exists
if not os.path.exists('voc_model.pkl') or not os.path.exists('voc_scaler.pkl'):
    st.error("⚠️ Model files not found. Please run `voc_classifier.py` first to train and save the model.")
    st.stop()

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses 16 Volatile Organic Compounds (VOCs) 
    from exhaled breath to predict lung cancer risk.
    
    **VOC Biomarkers:**
    - Butyraldehyde
    - Butyric acid
    - Dicyclohexyl ketone
    - Ethane
    - Toluene
    - Carbon disulfide
    - Sevoflurane
    - Acetone
    - Benzene
    - Pentane
    - Hexane
    - Heptane
    - Octane
    - Isoprene
    - Dimethyl sulfide
    - Indole
    """)
    
    st.markdown("---")
    st.markdown("**Reference:** Wang 2022 Study")

# Main content
st.header("VOC Input")

# Create two columns for input
col1, col2 = st.columns(2)

voc_values = {}

with col1:
    st.subheader("Group 1")
    for i, voc in enumerate(VOC_NAMES[:8]):
        voc_values[voc] = st.number_input(
            f"{voc}",
            min_value=0.0,
            value=float(VOC_MEDIANS_LC[voc]),
            step=0.1,
            key=f"voc_{i}"
        )

with col2:
    st.subheader("Group 2")
    for i, voc in enumerate(VOC_NAMES[8:], start=8):
        voc_values[voc] = st.number_input(
            f"{voc}",
            min_value=0.0,
            value=float(VOC_MEDIANS_LC[voc]),
            step=0.1,
            key=f"voc_{i}"
        )

# Prediction button
st.markdown("---")
if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
    try:
        # Make prediction
        result = predict_risk(voc_values)
        
        # Display results
        st.header("Prediction Results")
        
        # Probability gauge
        prob = result['probability']
        st.metric("Cancer Probability", f"{prob:.1%}")
        
        # Progress bar
        st.progress(prob)
        
        # Risk category
        if prob >= 0.5:
            st.error(f"⚠️ **{result['risk_category']}**")
        else:
            st.success(f"✅ **{result['risk_category']}**")
        
        # Recommendation
        st.info(f"💡 **Recommendation:** {result['recommendation']}")
        
        # Detailed breakdown
        with st.expander("📊 Detailed Analysis"):
            st.write(f"**Probability Score:** {prob:.4f}")
            st.write(f"**Threshold:** 0.5 (50%)")
            st.write(f"**Interpretation:**")
            if prob >= 0.5:
                st.write("- High probability of lung cancer detected")
                st.write("- Immediate referral to Low-Dose CT (LDCT) screening recommended")
                st.write("- Further clinical evaluation advised")
            else:
                st.write("- Low probability of lung cancer")
                st.write("- Continue regular monitoring")
                st.write("- Maintain healthy lifestyle and regular check-ups")
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# Example values button
st.markdown("---")
if st.button("📋 Load Example Values (LC Patient)", use_container_width=True):
    # Set example values (LC patient levels)
    for i, voc in enumerate(VOC_NAMES):
        st.session_state[f"voc_{i}"] = float(VOC_MEDIANS_LC[voc])
    st.rerun()

if st.button("📋 Load Example Values (Healthy Patient)", use_container_width=True):
    # Set example values (70% of LC levels for healthy)
    for i, voc in enumerate(VOC_NAMES):
        st.session_state[f"voc_{i}"] = float(VOC_MEDIANS_LC[voc] * 0.7)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>Lung Cancer VOC Classifier | Based on Wang 2022 Study</small>
    </div>
    """,
    unsafe_allow_html=True
)
