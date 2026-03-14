import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import urllib.parse
import time
import glob
import re
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="LungGuard HK - XGBoost", page_icon="🫁", layout="centered", initial_sidebar_state="expanded")

# ==================== TRANSLATIONS (same as before) ====================
translations = {
    "en": {
        "title": "🫁 LungGuard HK (XGBoost Version)",
        "subtitle": "Non-Smoker Lung Cancer Breath Screening – Advanced Model",
        "language": "Language",
        "baseline_title": "Non-Smoker Risk Calculator",
        "breath_title": "Breath Analysis",
        "vocs_tab": "16 VOCs",
        "research_tab": "🔬 Researcher Mode",
        "age": "Age",
        "gender": "Gender",
        "cooking": "Cooking Oil Use",
        "air_days": "HK Poor Air Quality Days (last year)",
        "family": "Family History of Lung Cancer",
        "input_method": "Input Method",
        "manual": "Manual Entry (Sliders)",
        "upload": "Upload CSV (Patient Screening)",
        "analyze_btn": "🚀 Analyze Breath",
        "book_ldct": "📅 Book AI-LDCT Now",
        "risk": "Your Total Risk",
        "breath_risk": "Breath VOC Risk",
        "baseline_risk": "Baseline Risk",
        "high": "HIGH RISK",
        "medium": "MEDIUM RISK",
        "low": "LOW RISK",
        "metrics_title": "Model Performance",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "auc": "AUC",
        "accuracy_demo": "Accuracy (demo)",
        "footer": "LungGuard HK v3.0 – XGBoost Prototype • Research Only • Not a medical device",
        "pdf_title": "LungGuard HK Risk Report (XGBoost)",
        "pdf_date": "Date",
        "pdf_patient": "Patient Summary",
        "pdf_overall": "Overall Risk Assessment",
        "pdf_voc_table": "VOC Analysis",
        "pdf_recommend": "Recommendation",
        "pdf_recommend_text": "Discuss this report with your doctor. Consider AI-LDCT referral if medium or high risk.",
        "prototype_warning": "⚠️ Research prototype only • Not a medical device • For pilot testing purposes",
    },
    "zh": {
        "title": "🫁 LungGuard HK (XGBoost 版本)",
        "subtitle": "非吸煙者肺癌呼吸篩查 – 進階模型",
        "language": "語言",
        "baseline_title": "非吸煙者風險計算器",
        "breath_title": "呼吸分析",
        "vocs_tab": "16 VOCs",
        "research_tab": "🔬 研究者模式",
        "age": "年齡",
        "gender": "性別",
        "cooking": "高溫煮食頻率",
        "air_days": "去年不良空氣質素日數",
        "family": "肺癌家族病史",
        "input_method": "輸入方式",
        "manual": "手動輸入（滑桿）",
        "upload": "上傳 CSV（病人篩查）",
        "analyze_btn": "🚀 分析呼吸樣本",
        "book_ldct": "📅 立即預約 AI-LDCT",
        "risk": "您的總風險",
        "breath_risk": "呼吸 VOC 風險",
        "baseline_risk": "基線風險",
        "high": "高風險",
        "medium": "中風險",
        "low": "低風險",
        "metrics_title": "模型效能",
        "sensitivity": "敏感度",
        "specificity": "特異度",
        "auc": "AUC",
        "accuracy_demo": "準確度（示範）",
        "footer": "LungGuard HK v3.0 – XGBoost 原型 • 僅供研究 • 非醫療儀器",
        "pdf_title": "LungGuard HK 風險報告 (XGBoost)",
        "pdf_date": "日期",
        "pdf_patient": "病人摘要",
        "pdf_overall": "整體風險評估",
        "pdf_voc_table": "VOC 分析",
        "pdf_recommend": "建議",
        "pdf_recommend_text": "請與醫生討論此報告。如為中或高風險，請考慮 AI-LDCT 轉介。",
        "prototype_warning": "⚠️ 僅為研究原型 • 非醫療儀器 • 僅用於試點測試",
    }
}

lang_code = st.sidebar.selectbox("語言 / Language", ["English", "繁體中文"], index=0)
lang = "en" if lang_code == "English" else "zh"
t = translations[lang]

# ==================== MODE SELECTOR ====================
st.markdown("<h3 style='text-align: center; margin-bottom: 8px;'>Select Mode</h3>", unsafe_allow_html=True)

if "mode" not in st.session_state:
    st.session_state.mode = "👤 User / Patient Screening"

current_mode = st.session_state.mode

mode_col1, mode_col2 = st.columns(2)

with mode_col1:
    patient_active = current_mode == "👤 User / Patient Screening"
    if st.button(
        "👤 Patient\nScreening mode",
        use_container_width=True,
        type="primary" if patient_active else "secondary",
        key="btn_patient_v3"
    ):
        st.session_state.mode = "👤 User / Patient Screening"
        st.rerun()

with mode_col2:
    researcher_active = current_mode == "🔬 Researcher Mode"
    if st.button(
        "🔬 Researcher\nTraining mode",
        use_container_width=True,
        type="primary" if researcher_active else "secondary",
        key="btn_researcher_v3"
    ):
        st.session_state.mode = "🔬 Researcher Mode"
        st.rerun()

mode = st.session_state.mode

st.markdown("---")

st.title(t["title"])
st.caption(t["subtitle"])

# ==================== GLOBAL DEFAULTS ====================
defaults = {
    "Acetaldehyde": 100, "2-Hydroxyacetaldehyde": 700, "Isoprene": 5000,
    "Pentanal": 100, "Butyric acid": 250, "Toluene": 3000,
    "2,5-Dimethylfuran": 15, "Cyclohexanone": 130, "Hexanal": 20,
    "Heptanal": 35, "Acetophenone": 130, "Propylcyclohexane": 220,
    "Octanal": 45, "Nonanal": 25, "Decanal": 45, "2,2-Dimethyldecane": 130
}

VOC_COLUMNS = list(defaults.keys())

# ==================== SIDEBAR – BASELINE RISK ====================
if mode == "👤 User / Patient Screening":
    with st.sidebar:
        st.header(t["baseline_title"])
        age = st.slider(t["age"], 18, 85, 45)
        gender = st.selectbox(t["gender"], ["Female", "Male"] if lang == "en" else ["女性", "男性"])
        cooking = st.selectbox(t["cooking"], 
                               ["Never","Rarely", "1-2x/week", "3-5x/week", "Daily"] if lang == "en" 
                               else ["從不", "很少", "每週1-2次", "每週3-5次", "每日"])
        air_days = st.slider(t["air_days"], 0, 365, 120)
        family = st.selectbox(t["family"], 
                              ["No", "Yes (1st degree)", "Yes (2nd degree)"] if lang == "en" 
                              else ["無", "有（一級親屬）", "有（二級親屬）"])

        baseline_score = min(int(
            (age - 18) / (85 - 18) * 40 +
            (15 if "1st" in str(family) or "一級" in str(family) else 7 if "2nd" in str(family) or "二級" in str(family) else 0) +
            (0 if "Never" in str(cooking) or "從不" in str(cooking) else 5 if "Rarely" in str(cooking) or "很少" in str(cooking) else 10 if "1-2" in str(cooking) else 20 if "3-5" in str(cooking) else 30) +
            (air_days / 365 * 20)
        ), 100)

        st.progress(baseline_score / 100)
        st.caption(f"{t['baseline_risk']}: {baseline_score}%")
else:
    age = gender = cooking = air_days = family = baseline_score = None  # not used

# ==================== MAIN CONTENT ====================
if mode == "👤 User / Patient Screening":
    tab1, tab2 = st.tabs([t["breath_title"], t["vocs_tab"]])

    with tab1:
        st.subheader(t["input_method"])
        input_method = st.radio("", [t["manual"], t["upload"]], horizontal=True)

        results = []
        is_batch = False
        voc_values = {}
        samples = []
        voc_dict_list = []

        if input_method == t["manual"]:
            with st.expander("Hardware Connection (Beta)", expanded=False):
                if st.button("🔗 Connect Bluetooth Sensor"):
                    with st.spinner("Connecting... (simulation)"):
                        time.sleep(1.5)
                    st.success("Device connected! Ready to read breath.")
                    st.info("In real version: pairs via Bluetooth, reads 16 VOCs in one breath.")

                if st.button("📡 Read Breath Sample from Device"):
                    st.info("Simulating breath reading from Bluetooth sensor...")
                    simulated_values = {name: np.random.normal(val, val*0.15) for name, val in defaults.items()}
                    st.write("Received VOC values (simulation):")
                    for name, val in simulated_values.items():
                        st.write(f"{name}: {val:.1f}")
                    st.session_state.voc_values = simulated_values
                    st.success("Ready — click 'Analyze Breath' to process!")

            voc_values = st.session_state.get('voc_values', {})
            if not voc_values:
                voc_values = {}
                cols = st.columns(2)
                for i, (name, val) in enumerate(defaults.items()):
                    with cols[i % 2]:
                        voc_values[name] = st.slider(name, 0, int(val * 3), val, step=5, key=f"manual_{name}")

            voc_dict_list = [voc_values]

        else:
            uploaded = st.file_uploader("Upload patient breath CSV (16 VOC columns only)", type="csv")
            if uploaded:
                try:
                    df = pd.read_csv(uploaded)
                    missing = [col for col in VOC_COLUMNS if col not in df.columns]
                    if missing:
                        st.error(f"Missing columns: {', '.join(missing)}")
                    else:
                        st.success(f"Loaded {len(df)} patient sample(s)")
                        st.session_state.uploaded_df = df
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

        if st.button(t["analyze_btn"], type="primary", use_container_width=True):
            model_path = "xgboost_model.pkl"
            explainer_path = "shap_explainer.pkl"

            if not os.path.exists(model_path):
                st.warning("No trained XGBoost model found yet. Please train in Researcher mode first.")
                st.stop()

            model = joblib.load(model_path)
            explainer = joblib.load(explainer_path) if os.path.exists(explainer_path) else None

            if input_method == t["manual"]:
                if not voc_values:
                    st.warning("Please enter VOC values first.")
                    st.stop()
                samples = [pd.Series(voc_values)]
                is_batch = False
                voc_dict_list = [voc_values]
            else:
                if 'uploaded_df' not in st.session_state:
                    st.warning("Please upload a valid CSV first.")
                    st.stop()
                df_upload = st.session_state.uploaded_df
                samples = [row for _, row in df_upload.iterrows()]
                is_batch = len(samples) > 1
                voc_dict_list = [row.to_dict() for row in samples]

            results = []
            for idx, (sample, voc_dict) in enumerate(zip(samples, voc_dict_list)):
                input_row = [voc_dict.get(col, 0) for col in VOC_COLUMNS]
                input_df = pd.DataFrame([input_row], columns=VOC_COLUMNS)

                breath_prob = model.predict_proba(input_df)[0][1] * 100
                total_risk = int(0.6 * breath_prob + 0.4 * (baseline_score or 22))

                level = t["high"] if total_risk >= 70 else t["medium"] if total_risk >= 40 else t["low"]
                color = "#ff4b4b" if total_risk >= 70 else "#ffaa00" if total_risk >= 40 else "#00cc00"

                shap_values = None
                if explainer and not is_batch:
                    shap_values = explainer.shap_values(input_df)

                if is_batch:
                    results.append({
                        "Sample": idx + 1,
                        "Breath VOC Risk (%)": round(breath_prob, 1),
                        "Total Risk (%)": total_risk,
                        "Risk Level": level,
                        **voc_dict
                    })
                else:
                    st.markdown(f"<h2 style='color:{color}'>{t['risk']}: {total_risk}% — {level}</h2>", unsafe_allow_html=True)
                    st.progress(total_risk / 100)
                    st.write(f"**{t['breath_risk']}**: {round(breath_prob, 1)}%")
                    st.write(f"**{t['baseline_risk']}**: {baseline_score or 'N/A'}%")

                    st.subheader("SHAP Explanation – Why this risk?")
                    if shap_values is not None:
                        st_shap(shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0], matplotlib=False))
                        st_shap(shap.summary_plot(shap_values, input_df, plot_type="bar", show=False))
                    else:
                        st.info("SHAP explanations not available yet.")

                    # PDF generation (same as before, but you can add SHAP text later)
                    buffer = BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                    elements = []

                    elements.append(Paragraph(t["pdf_title"], styles['Title']))
                    elements.append(Spacer(1, 12))
                    elements.append(Paragraph(f"{t['pdf_date']}: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
                    elements.append(Spacer(1, 24))

                    elements.append(Paragraph(t["pdf_patient"], styles['Heading2']))
                    patient_data = [
                        [t["age"], str(age)],
                        [t["gender"], gender],
                        [t["cooking"], cooking],
                        [t["air_days"], str(air_days)],
                        [t["family"], family],
                        [t["baseline_risk"], f"{baseline_score or 'N/A'}%"]
                    ]
                    patient_table = Table(patient_data, colWidths=[180, 300])
                    patient_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ]))
                    elements.append(patient_table)
                    elements.append(Spacer(1, 24))

                    elements.append(Paragraph(t["pdf_overall"], styles['Heading2']))
                    risk_color = colors.red if total_risk >= 70 else colors.orange if total_risk >= 40 else colors.green
                    risk_text = f"<font color='{risk_color.hexval()}'>{total_risk}% — {level}</font>"
                    elements.append(Paragraph(risk_text, styles['Heading3']))
                    elements.append(Spacer(1, 12))

                    elements.append(Paragraph(t["pdf_voc_table"], styles['Heading2']))
                    voc_rows = [["VOC", "Your Value", "Risk Level"]]
                    healthy_medians = list(defaults.values())
                    row_colors = []

                    for i, (name, value) in enumerate(voc_dict.items()):
                        healthy_median = healthy_medians[i]
                        if value > healthy_median * 1.8:
                            row_colors.append(colors.red)
                            level_text = "High" if lang == "en" else "高"
                        elif value > healthy_median * 1.2:
                            row_colors.append(colors.orange)
                            level_text = "Medium" if lang == "en" else "中"
                        else:
                            row_colors.append(colors.green)
                            level_text = "Low" if lang == "en" else "低"
                        voc_rows.append([name, f"{float(value):.1f}", level_text])

                    voc_table = Table(voc_rows, colWidths=[200, 120, 160])
                    voc_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ]))

                    for row_idx, color in enumerate(row_colors, start=1):
                        voc_table.setStyle(TableStyle([('BACKGROUND', (2, row_idx), (2, row_idx), color)]))

                    elements.append(voc_table)
                    elements.append(Spacer(1, 24))

                    elements.append(Paragraph(t["pdf_recommend"], styles['Heading2']))
                    elements.append(Paragraph(t["pdf_recommend_text"], styles['Normal']))

                    doc.build(elements)
                    buffer.seek(0)

                    st.download_button(
                        label="📄 Download Professional PDF Report",
                        data=buffer,
                        file_name=f"LungGuard_XGBoost_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

            if is_batch and results:
                results_df = pd.DataFrame(results)
                st.subheader("Batch Analysis Results")
                st.dataframe(results_df.style.format(precision=1), use_container_width=True)

                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Batch Results CSV",
                    data=csv,
                    file_name=f"LungGuard_Batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    with tab2:
        st.markdown("**The 16 Cancer-Associated VOCs** (Wang et al., eClinicalMedicine 2022)")
        voc_data = pd.DataFrame([
            {"VOC": "Isoprene", "Median in LC patients": 6062, "Strength": "Strongest (AUC 0.859)", "Notes": "Highest individual predictor"},
            # ... (keep your full list here)
        ])
        st.dataframe(voc_data, use_container_width=True, hide_index=True)

else:  # Researcher Mode
    st.header("🔬 Researcher / Lab Mode – XGBoost Training")

    # Password persistence
    if "researcher_authenticated" not in st.session_state:
        st.session_state.researcher_authenticated = False

    if st.session_state.researcher_authenticated:
        st.success("✅ Researcher access granted (session remembered)")
    else:
        st.markdown("**Researcher Password** (current: **lunguard2026**)")
        password = st.text_input(
            "",
            type="password",
            placeholder="Enter lunguard2026 and press Enter",
            label_visibility="collapsed",
            key="pw_v3"
        )
        if password and password == "lunguard2026":
            st.session_state.researcher_authenticated = True
            st.success("Access granted!")
            st.rerun()
        elif password:
            st.error("Incorrect password")

    if st.session_state.researcher_authenticated:
        st.markdown("""
        Upload labelled CSV files (16 VOC columns + **target** column: 0=healthy, 1=cancer)
        """)

        latest_file = "training_data_latest.csv"
        if os.path.exists(latest_file):
            df_hist = pd.read_csv(latest_file)
            st.metric("Total labelled samples", len(df_hist))

        uploaded_files = st.file_uploader("Upload labelled CSV(s)", type="csv", accept_multiple_files=True)

        if uploaded_files and st.button("Train XGBoost Model", type="primary"):
            all_data = []
            for file in uploaded_files:
                try:
                    df = pd.read_csv(file)
                    if not all(col in df.columns for col in VOC_COLUMNS + ["target"]):
                        st.error(f"File {file.name} missing required columns")
                        continue
                    all_data.append(df[VOC_COLUMNS + ["target"]])
                except:
                    st.error(f"Error reading {file.name}")

            if all_data:
                df_all = pd.concat(all_data, ignore_index=True).drop_duplicates(subset=VOC_COLUMNS)
                df_all.to_csv(latest_file, index=False)

                X = df_all[VOC_COLUMNS]
                y = df_all["target"]

                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="auc",
                    early_stopping_rounds=20
                )

                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )

                joblib.dump(model, "xgboost_model.pkl")

                explainer = shap.TreeExplainer(model)
                joblib.dump(explainer, "shap_explainer.pkl")

                auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                st.success(f"XGBoost trained! Validation AUC: {auc:.3f}")