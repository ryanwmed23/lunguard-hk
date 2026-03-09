"""
LungGuard HK - Never-Smoker Breath Cancer Screen
Streamlit App for Lung Cancer VOC Classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from voc_classifier import predict_risk, VOC_NAMES, VOC_MEDIANS_LC
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="LungGuard HK - Never-Smoker Breath Cancer Screen",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
    .risk-low {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="main-header">🫁 LungGuard HK</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Never-Smoker Breath Cancer Screen</p>', unsafe_allow_html=True)

# HK-specific validation text
st.info("✅ **Validated on Asian data. Complements Hospital Authority AI-LDCT pilot.**")

st.markdown("---")

# Sidebar - Never-Smoker Risk Calculator
with st.sidebar:
    st.header("📊 Never-Smoker Risk Calculator")
    st.markdown("Complete this assessment to calculate your baseline risk score.")
    
    # Risk factors
    age = st.slider("Age", 18, 100, 50, help="Age in years")
    gender = st.selectbox("Gender", ["Female", "Male", "Other"], help="Biological gender")
    cooking_oil = st.selectbox(
        "Cooking Oil Use", 
        ["Never", "Rarely (1-2x/week)", "Sometimes (3-5x/week)", "Frequently (6-7x/week)", "Daily"],
        help="Frequency of cooking with oil at home"
    )
    air_quality_days = st.slider(
        "HK Air Quality Days (Poor/Unhealthy)", 
        0, 365, 30,
        help="Number of days per year exposed to poor/unhealthy air quality"
    )
    family_history = st.selectbox(
        "Family History of Lung Cancer",
        ["No", "Yes (1st degree relative)", "Yes (2nd degree relative)"],
        help="Family history of lung cancer"
    )
    
    # Calculate risk score
    risk_score = 0
    
    # Age scoring (higher risk with age)
    if age >= 70:
        risk_score += 3
    elif age >= 60:
        risk_score += 2
    elif age >= 50:
        risk_score += 1
    
    # Gender scoring (males slightly higher risk)
    if gender == "Male":
        risk_score += 1
    
    # Cooking oil scoring
    oil_scores = {
        "Never": 0,
        "Rarely (1-2x/week)": 1,
        "Sometimes (3-5x/week)": 2,
        "Frequently (6-7x/week)": 3,
        "Daily": 4
    }
    risk_score += oil_scores.get(cooking_oil, 0)
    
    # Air quality scoring
    if air_quality_days >= 100:
        risk_score += 3
    elif air_quality_days >= 50:
        risk_score += 2
    elif air_quality_days >= 20:
        risk_score += 1
    
    # Family history scoring
    if family_history == "Yes (1st degree relative)":
        risk_score += 3
    elif family_history == "Yes (2nd degree relative)":
        risk_score += 1
    
    # Display risk score
    st.markdown("---")
    st.subheader("Baseline Risk Score")
    st.metric("Score", f"{risk_score}/14")
    
    # Risk interpretation
    if risk_score >= 8:
        risk_level = "High"
        risk_color = "🔴"
    elif risk_score >= 5:
        risk_level = "Moderate"
        risk_color = "🟡"
    else:
        risk_level = "Low"
        risk_color = "🟢"
    
    st.markdown(f"**Risk Level:** {risk_color} {risk_level}")
    
    st.markdown("---")
    st.markdown("""
    <small>
    **Note:** This calculator provides a baseline risk assessment. 
    The breath analysis below provides more specific biomarker-based risk evaluation.
    </small>
    """, unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🌬️ Breath Analysis")
    
    # Input method selection
    input_method = st.radio(
        "Select Input Method:",
        ["Manual Entry (Sliders)", "Upload CSV File"],
        horizontal=True
    )
    
    voc_values = {}
    
    if input_method == "Manual Entry (Sliders)":
        # Default value selection
        default_type = st.radio(
            "Default Values:",
            ["Healthy Patient (70% of LC medians)", "LC Patient (Full medians)"],
            horizontal=True
        )
        
        # Create sliders in two columns
        col_slider1, col_slider2 = st.columns(2)
        
        with col_slider1:
            st.subheader("Group 1 VOCs")
            for i, voc in enumerate(VOC_NAMES[:8]):
                if default_type == "Healthy Patient (70% of LC medians)":
                    default_val = VOC_MEDIANS_LC[voc] * 0.7
                else:
                    default_val = VOC_MEDIANS_LC[voc]
                
                voc_values[voc] = st.slider(
                    f"{voc}",
                    min_value=0.0,
                    max_value=float(VOC_MEDIANS_LC[voc] * 2),
                    value=float(default_val),
                    step=0.1,
                    key=f"voc_{i}"
                )
        
        with col_slider2:
            st.subheader("Group 2 VOCs")
            for i, voc in enumerate(VOC_NAMES[8:], start=8):
                if default_type == "Healthy Patient (70% of LC medians)":
                    default_val = VOC_MEDIANS_LC[voc] * 0.7
                else:
                    default_val = VOC_MEDIANS_LC[voc]
                
                voc_values[voc] = st.slider(
                    f"{voc}",
                    min_value=0.0,
                    max_value=float(VOC_MEDIANS_LC[voc] * 2),
                    value=float(default_val),
                    step=0.1,
                    key=f"voc_{i}"
                )
    
    else:  # CSV Upload
        uploaded_file = st.file_uploader(
            "Upload Breath CSV File",
            type=['csv'],
            help="CSV file should contain columns with VOC names matching the 16 biomarkers"
        )
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.success("✅ CSV file loaded successfully!")
                st.dataframe(df_upload.head())
                
                # Try to match VOC names (case-insensitive, flexible matching)
                for voc in VOC_NAMES:
                    # Try exact match first
                    if voc in df_upload.columns:
                        voc_values[voc] = float(df_upload[voc].iloc[0])
                    else:
                        # Try case-insensitive match
                        matching_cols = [col for col in df_upload.columns if voc.lower() in col.lower() or col.lower() in voc.lower()]
                        if matching_cols:
                            voc_values[voc] = float(df_upload[matching_cols[0]].iloc[0])
                        else:
                            st.warning(f"⚠️ {voc} not found in CSV. Using default value.")
                            voc_values[voc] = VOC_MEDIANS_LC[voc] * 0.7
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                st.info("Please ensure your CSV has the correct format with VOC names as columns.")
        else:
            st.info("Please upload a CSV file with VOC measurements.")

with col2:
    st.header("📋 Quick Info")
    st.markdown("""
    **16 VOC Biomarkers:**
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
    
    **Reference:** Wang 2022 Study
    """)

# Analyze button
st.markdown("---")
analyze_button = st.button("🔍 Analyze Breath", type="primary", use_container_width=True)

# Results section
if analyze_button and voc_values:
    # Check if model exists
    if not os.path.exists('voc_model.pkl') or not os.path.exists('voc_scaler.pkl'):
        st.error("⚠️ Model files not found. Please run `voc_classifier.py` first to train and save the model.")
    else:
        try:
            # Make prediction
            result = predict_risk(voc_values)
            prob = result['probability']
            prob_percent = prob * 100
            
            # Store results in session state for PDF generation
            st.session_state['prediction_result'] = result
            st.session_state['voc_values'] = voc_values
            st.session_state['risk_score'] = risk_score
            st.session_state['patient_info'] = {
                'age': age,
                'gender': gender,
                'cooking_oil': cooking_oil,
                'air_quality_days': air_quality_days,
                'family_history': family_history
            }
            
            # Display results
            st.header("📊 Analysis Results")
            
            # Main metrics
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                st.metric("Cancer Risk Probability", f"{prob_percent:.1f}%")
            
            with col_metric2:
                st.metric("Baseline Risk Score", f"{risk_score}/14")
            
            with col_metric3:
                combined_risk = (prob_percent + (risk_score / 14 * 100)) / 2
                st.metric("Combined Risk", f"{combined_risk:.1f}%")
            
            # Risk category and recommendation
            st.markdown("---")
            
            if prob >= 0.5:
                st.markdown(
                    f'<div class="risk-high">'
                    f'<h2>⚠️ High Risk Detected</h2>'
                    f'<p style="font-size: 1.2rem;"><strong>Recommendation: Book LDCT via HA Go now</strong></p>'
                    f'<p>Your breath analysis indicates elevated levels of lung cancer biomarkers. '
                    f'We recommend immediate consultation and Low-Dose CT (LDCT) screening through Hospital Authority Go.</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="risk-low">'
                    f'<h2>✅ Low Risk</h2>'
                    f'<p style="font-size: 1.2rem;"><strong>Recommendation: Annual check</strong></p>'
                    f'<p>Your breath analysis shows biomarker levels within normal ranges. '
                    f'Continue with regular annual health check-ups and maintain a healthy lifestyle.</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Visualizations
            st.markdown("---")
            st.subheader("📈 Visualizations")
            
            # Create charts
            fig_col1, fig_col2 = st.columns(2)
            
            with fig_col1:
                # Risk probability gauge
                fig_gauge, ax_gauge = plt.subplots(figsize=(8, 6))
                ax_gauge.set_xlim(0, 1)
                ax_gauge.set_ylim(0, 1)
                
                # Draw gauge
                theta = np.linspace(0, np.pi, 100)
                r = 0.4
                x = r * np.cos(theta) + 0.5
                y = r * np.sin(theta) + 0.5
                
                # Color based on risk
                if prob >= 0.5:
                    color = 'red'
                elif prob >= 0.3:
                    color = 'orange'
                else:
                    color = 'green'
                
                ax_gauge.plot(x, y, 'k-', linewidth=8)
                ax_gauge.fill_between(x, 0.5, y, alpha=0.3, color=color)
                
                # Needle
                needle_angle = np.pi * (1 - prob)
                needle_x = 0.5 + 0.35 * np.cos(needle_angle)
                needle_y = 0.5 + 0.35 * np.sin(needle_angle)
                ax_gauge.plot([0.5, needle_x], [0.5, needle_y], 'k-', linewidth=3)
                ax_gauge.plot(0.5, 0.5, 'ko', markersize=10)
                
                ax_gauge.text(0.5, 0.2, f'{prob_percent:.1f}%', 
                             ha='center', va='center', fontsize=24, fontweight='bold')
                ax_gauge.text(0.5, 0.1, 'Cancer Risk', 
                             ha='center', va='center', fontsize=14)
                ax_gauge.axis('off')
                ax_gauge.set_title('Risk Probability Gauge', fontsize=16, fontweight='bold', pad=20)
                
                plt.tight_layout()
                st.pyplot(fig_gauge)
                plt.close()
            
            with fig_col2:
                # VOC levels comparison
                fig_voc, ax_voc = plt.subplots(figsize=(10, 6))
                
                voc_list = list(VOC_NAMES)
                measured_values = [voc_values[voc] for voc in voc_list]
                healthy_baseline = [VOC_MEDIANS_LC[voc] * 0.7 for voc in voc_list]
                lc_baseline = [VOC_MEDIANS_LC[voc] for voc in voc_list]
                
                x_pos = np.arange(len(voc_list))
                width = 0.25
                
                ax_voc.bar(x_pos - width, healthy_baseline, width, label='Healthy Baseline', color='green', alpha=0.7)
                ax_voc.bar(x_pos, measured_values, width, label='Measured', color='blue', alpha=0.7)
                ax_voc.bar(x_pos + width, lc_baseline, width, label='LC Baseline', color='red', alpha=0.7)
                
                ax_voc.set_xlabel('VOC Biomarkers', fontsize=10)
                ax_voc.set_ylabel('Peak Intensity', fontsize=10)
                ax_voc.set_title('VOC Levels Comparison', fontsize=14, fontweight='bold')
                ax_voc.set_xticks(x_pos)
                ax_voc.set_xticklabels(voc_list, rotation=45, ha='right', fontsize=8)
                ax_voc.legend()
                ax_voc.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_voc)
                plt.close()
            
            # Top biomarkers
            st.markdown("---")
            st.subheader("🔬 Top Biomarkers Analysis")
            
            # Calculate deviation from healthy baseline
            biomarker_deviations = {}
            for voc in VOC_NAMES:
                healthy_val = VOC_MEDIANS_LC[voc] * 0.7
                measured_val = voc_values[voc]
                deviation = ((measured_val - healthy_val) / healthy_val) * 100
                biomarker_deviations[voc] = deviation
            
            # Sort by absolute deviation
            sorted_biomarkers = sorted(biomarker_deviations.items(), 
                                      key=lambda x: abs(x[1]), 
                                      reverse=True)
            
            # Display top 5
            top_5_df = pd.DataFrame(sorted_biomarkers[:5], columns=['Biomarker', 'Deviation from Healthy (%)'])
            top_5_df['Status'] = top_5_df['Deviation from Healthy (%)'].apply(
                lambda x: '⚠️ Elevated' if x > 20 else '✅ Normal' if x > -20 else '⬇️ Low'
            )
            
            st.dataframe(top_5_df, use_container_width=True)
            
            # PDF Report Generation
            st.markdown("---")
            st.subheader("📄 Download Report")
            
            def generate_pdf():
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                story = []
                
                # Styles
                styles = getSampleStyleSheet()
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    textColor=colors.HexColor('#1f77b4'),
                    alignment=TA_CENTER,
                    spaceAfter=30
                )
                
                # Title
                story.append(Paragraph("LungGuard HK", title_style))
                story.append(Paragraph("Never-Smoker Breath Cancer Screen", styles['Heading2']))
                story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                                      styles['Normal']))
                story.append(Spacer(1, 0.2*inch))
                
                # Patient Information
                story.append(Paragraph("Patient Information", styles['Heading2']))
                patient_info = st.session_state['patient_info']
                patient_data = [
                    ['Age:', str(patient_info['age'])],
                    ['Gender:', patient_info['gender']],
                    ['Cooking Oil Use:', patient_info['cooking_oil']],
                    ['HK Air Quality Days:', str(patient_info['air_quality_days'])],
                    ['Family History:', patient_info['family_history']],
                    ['Baseline Risk Score:', f"{st.session_state['risk_score']}/14"]
                ]
                patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
                patient_table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ])
                patient_table.setStyle(patient_table_style)
                story.append(patient_table)
                story.append(Spacer(1, 0.3*inch))
                
                # Results
                story.append(Paragraph("Analysis Results", styles['Heading2']))
                result = st.session_state['prediction_result']
                results_data = [
                    ['Cancer Risk Probability:', f"{result['probability']*100:.2f}%"],
                    ['Risk Category:', result['risk_category']],
                    ['Recommendation:', result['recommendation']]
                ]
                results_table = Table(results_data, colWidths=[2*inch, 4*inch])
                results_table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ])
                results_table.setStyle(results_table_style)
                story.append(results_table)
                story.append(Spacer(1, 0.3*inch))
                
                # VOC Values
                story.append(Paragraph("VOC Biomarker Measurements", styles['Heading2']))
                voc_data = [['VOC', 'Value']]
                for voc in VOC_NAMES:
                    voc_data.append([voc, f"{st.session_state['voc_values'][voc]:.2f}"])
                
                voc_table = Table(voc_data, colWidths=[3*inch, 3*inch])
                voc_table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 9)
                ])
                voc_table.setStyle(voc_table_style)
                story.append(voc_table)
                story.append(Spacer(1, 0.3*inch))
                
                # Disclaimer
                story.append(Paragraph("Disclaimer", styles['Heading3']))
                story.append(Paragraph(
                    "This report is for informational purposes only. Validated on Asian data. "
                    "Complements Hospital Authority AI-LDCT pilot. Please consult with healthcare "
                    "professionals for medical decisions.",
                    styles['Normal']
                ))
                
                doc.build(story)
                buffer.seek(0)
                return buffer
            
            pdf_buffer = generate_pdf()
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_buffer,
                file_name=f"LungGuard_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.info("Please ensure all VOC values are provided and the model files exist.")

elif analyze_button and not voc_values:
    st.warning("⚠️ Please provide VOC values using sliders or upload a CSV file.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
<small>
<strong>LungGuard HK</strong> - Never-Smoker Breath Cancer Screen<br>
Validated on Asian data. Complements Hospital Authority AI-LDCT pilot.<br>
Based on Wang 2022 Study - 16 VOC Biomarkers
</small>
</div>
""", unsafe_allow_html=True)
