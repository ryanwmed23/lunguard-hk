import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors as lib_colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import time
import glob
import re
import shap
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="LungGuard HK – e-nose Advanced", page_icon="🫁", layout="wide")

# ==================== CONFIG & CONSTANTS ====================
NUM_SENSORS = 12
TIME_POINTS = 29

SENSOR_NAMES = [
    "TGS2600", "TGS2602", "TGS2620",
    "MQ2", "MQ3", "MQ4", "MQ5",
    "MQ6", "MQ7", "MQ9", "MC135", "Alkane"
]

FEATURE_COLUMNS = []
for s in SENSOR_NAMES:
    FEATURE_COLUMNS.extend([
        f"{s}_steady", f"{s}_rise_time", f"{s}_max_slope", f"{s}_auc",
        f"{s}_mean", f"{s}_std", f"{s}_max", f"{s}_min",
        f"{s}_fft1", f"{s}_fft2", f"{s}_fft3", f"{s}_fft4", f"{s}_dominant_freq"
    ])

# ==================== TRANSLATIONS ====================
translations = {
    "en": {
        "title": "🫁 LungGuard HK – Advanced e-nose",
        "subtitle": "Breathprint Recognition with Time-Series + FFT Features",
        "language": "Language",
        "baseline_title": "Baseline Risk Factors",
        "breath_title": "Breath Analysis",
        "sensors_tab": "Sensor Features",
        "research_tab": "🔬 Researcher Mode",
        "age": "Age",
        "gender": "Gender",
        "cooking": "Cooking Oil Use",
        "air_days": "HK Poor Air Quality Days (last year)",
        "family": "Family History of Lung Cancer",
        "input_method": "Input Method",
        "manual": "Manual Steady-State Values (simplified)",
        "upload": "Upload Full Time-Series Sensor CSV",
        "analyze_btn": "🚀 Analyze Breathprint",
        "risk": "Estimated Total Risk",
        "breath_risk": "Breathprint Risk (XGBoost)",
        "baseline_risk": "Baseline Risk",
        "high": "HIGH RISK",
        "medium": "MEDIUM RISK",
        "low": "LOW RISK",
        "metrics_title": "Model Performance",
        "footer": "LungGuard HK v4.3 – e-nose + Time-Series + FFT • Research Prototype Only • Not a medical device",
        "pdf_title": "LungGuard HK e-nose Report",
        "pdf_date": "Date",
        "pdf_patient": "Patient Summary",
        "pdf_overall": "Overall Risk Assessment",
        "pdf_sensor_table": "Sensor Steady-State Readings",
        "pdf_shap_table": "Top SHAP Sensor Features",
        "pdf_recommend": "Recommendation",
        "pdf_recommend_text": "Discuss this report with your doctor. Consider AI-LDCT referral if medium or high risk.",
        "prototype_warning": "⚠️ Research prototype only • Not a medical device • For pilot testing purposes",
        "model_ready": "Model loaded and ready for analysis",
        "no_model": "No trained model found. Train in Researcher mode first.",
        "model_found_disk": "Model file found on disk (may be old – retrain if needed)",
    },
    "zh": {
        "title": "🫁 LungGuard HK – 進階電子鼻",
        "subtitle": "帶時間序列與FFT特徵的呼吸印記識別",
        "language": "語言",
        "baseline_title": "基線風險因素",
        "breath_title": "呼吸分析",
        "sensors_tab": "感測器特徵",
        "research_tab": "🔬 研究者模式",
        "age": "年齡",
        "gender": "性別",
        "cooking": "高溫煮食頻率",
        "air_days": "去年不良空氣質素日數",
        "family": "肺癌家族病史",
        "input_method": "輸入方式",
        "manual": "手動穩態值（簡化）",
        "upload": "上傳完整時間序列感測器 CSV",
        "analyze_btn": "🚀 分析呼吸印記",
        "risk": "估計總風險",
        "breath_risk": "呼吸印記風險（XGBoost）",
        "baseline_risk": "基線風險",
        "high": "高風險",
        "medium": "中風險",
        "low": "低風險",
        "metrics_title": "模型效能",
        "footer": "LungGuard HK v4.3 – 電子鼻 + 時間序列 + FFT • 研究原型 • 非醫療儀器",
        "pdf_title": "LungGuard HK 電子鼻報告",
        "pdf_date": "日期",
        "pdf_patient": "病人摘要",
        "pdf_overall": "整體風險評估",
        "pdf_sensor_table": "感測器穩態讀數",
        "pdf_shap_table": "SHAP 前列感測器特徵",
        "pdf_recommend": "建議",
        "pdf_recommend_text": "請與醫生討論此報告。如為中或高風險，請考慮 AI-LDCT 轉介。",
        "prototype_warning": "⚠️ 僅為研究原型 • 非醫療儀器 • 僅用於試點測試",
        "model_ready": "模型已載入，可進行分析",
        "no_model": "未找到已訓練模型。請先在研究者模式中訓練。",
        "model_found_disk": "已找到模型檔案（可能為舊版 – 如有需要請重新訓練）",
    }
}

lang_code = st.sidebar.selectbox("語言 / Language", ["English", "繁體中文"], index=0)
lang = "en" if lang_code == "English" else "zh"
trans = translations[lang]

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
        key="btn_patient_v4"
    ):
        st.session_state.mode = "👤 User / Patient Screening"
        st.rerun()

with mode_col2:
    researcher_active = current_mode == "🔬 Researcher Mode"
    if st.button(
        "🔬 Researcher\nTraining mode",
        use_container_width=True,
        type="primary" if researcher_active else "secondary",
        key="btn_researcher_v4"
    ):
        st.session_state.mode = "🔬 Researcher Mode"
        st.rerun()

mode = st.session_state.mode

st.markdown("---")

st.title(trans["title"])
st.caption(trans["subtitle"])

# ==================== SIDEBAR – BASELINE RISK ====================
baseline_score = 22
if mode == "👤 User / Patient Screening":
    with st.sidebar:
        st.header(trans["baseline_title"])
        age = st.slider(trans["age"], 18, 85, 45)
        gender = st.selectbox(trans["gender"], ["Female", "Male"] if lang == "en" else ["女性", "男性"])
        cooking = st.selectbox(trans["cooking"], 
                               ["Never","Rarely", "1-2x/week", "3-5x/week", "Daily"] if lang == "en" 
                               else ["從不", "很少", "每週1-2次", "每週3-5次", "每日"])
        air_days = st.slider(trans["air_days"], 0, 365, 120)
        family = st.selectbox(trans["family"], 
                              ["No", "Yes (1st degree)", "Yes (2nd degree)"] if lang == "en" 
                              else ["無", "有（一級親屬）", "有（二級親屬）"])

        baseline_score = min(int(
            (age - 18) / (85 - 18) * 40 +
            (15 if "1st" in str(family) or "一級" in str(family) else 7 if "2nd" in str(family) or "二級" in str(family) else 0) +
            (0 if "Never" in str(cooking) or "從不" in str(cooking) else 5 if "Rarely" in str(cooking) or "很少" in str(cooking) else 10 if "1-2" in str(cooking) else 20 if "3-5" in str(cooking) else 30) +
            (air_days / 365 * 20)
        ), 100)

        st.progress(baseline_score / 100)
        st.caption(f"{trans['baseline_risk']}: {baseline_score}%")

# ==================== FEATURE ENGINEERING FUNCTION ====================
def extract_time_series_features(df_row):
    """df_row must be a DataFrame (even single row)"""
    features = {}
    for sensor in SENSOR_NAMES:
        cols = [f"{sensor}_t{i+1}" for i in range(TIME_POINTS)]
        if not all(c in df_row.columns for c in cols):
            if f"{sensor}_steady" in df_row.columns:
                steady = df_row[f"{sensor}_steady"].iloc[0]
                ts = np.full(TIME_POINTS, steady)  # flat line fallback
            else:
                continue
        else:
            ts = df_row[cols].values.flatten().astype(float)

        if len(ts) != TIME_POINTS or np.any(np.isnan(ts)):
            continue

        features[f"{sensor}_steady"]    = ts[-1]
        features[f"{sensor}_mean"]      = np.mean(ts)
        features[f"{sensor}_std"]       = np.std(ts)
        features[f"{sensor}_max"]       = np.max(ts)
        features[f"{sensor}_min"]       = np.min(ts)

        diff = ts - ts[0]
        max_diff = np.max(diff)
        rise_time = 0
        if max_diff > 0:
            rise_idx = np.argmax(diff >= 0.9 * max_diff)
            rise_time = rise_idx
        features[f"{sensor}_rise_time"] = rise_time

        slopes = np.diff(ts)
        features[f"{sensor}_max_slope"] = np.max(np.abs(slopes)) if len(slopes) > 0 else 0
        features[f"{sensor}_auc"]       = np.trapz(ts)

        # FFT features
        fft_vals = np.abs(np.fft.fft(ts))[:TIME_POINTS//2]
        fft_freq = np.fft.fftfreq(TIME_POINTS)[:TIME_POINTS//2]
        top_idx = np.argsort(fft_vals)[::-1][:4]
        for i, idx in enumerate(top_idx, 1):
            features[f"{sensor}_fft{i}"] = fft_vals[idx]
        features[f"{sensor}_dominant_freq"] = fft_freq[top_idx[0]] if len(top_idx) > 0 else 0

    return features

# ==================== RESEARCHER MODE ====================
if mode == "🔬 Researcher Mode":
    st.header("🔬 Researcher Mode – Advanced e-nose Training")

    if "researcher_authenticated" not in st.session_state:
        st.session_state.researcher_authenticated = False

    if not st.session_state.researcher_authenticated:
        st.markdown("**Researcher Password** (current: **lunguard2026**)")
        pw = st.text_input("", type="password", placeholder="Enter lunguard2026 and press Enter", label_visibility="collapsed")
        if pw == "lunguard2026":
            st.session_state.researcher_authenticated = True
            st.success("Access granted! (session remembered)")
            st.rerun()
    else:
        st.success("✅ Researcher access granted")

        st.markdown("""
        **Upload format**: CSV with columns  
        TGS2600_t1 … TGS2600_t29, MQ2_t1 … MQ2_t29, … (12 sensors × 29 points) + target (0/1)
        """)

        if st.button("Generate Synthetic Time-Series Data (200 samples)", use_container_width=True):
            np.random.seed(42)
            n = 200
            healthy_base = 2500
            cancer_shift = 700

            data = []
            targets = []
            for i in range(n):
                base = healthy_base if i < n//2 else healthy_base + cancer_shift
                row = {}
                for s in SENSOR_NAMES:
                    time_axis = np.linspace(0, 10, TIME_POINTS)
                    steady = base + np.random.normal(0, 150)
                    curve = steady / (1 + np.exp(-0.8*(time_axis - 4))) + np.random.normal(0, 80, TIME_POINTS)
                    for j, val in enumerate(curve, 1):
                        row[f"{s}_t{j}"] = val
                data.append(row)
                targets.append(0 if i < n//2 else 1)

            df_syn = pd.DataFrame(data)
            df_syn["target"] = targets
            df_syn.to_csv("synthetic_e_nose_timeseries.csv", index=False)
            st.success("Synthetic dataset created (100 healthy + 100 cancer-like patterns)")
            st.download_button(
                "Download synthetic CSV",
                df_syn.to_csv(index=False).encode('utf-8'),
                "synthetic_e_nose_timeseries.csv",
                mime="text/csv"
            )

        uploaded_files = st.file_uploader("Upload labelled time-series CSV file(s)", type="csv", accept_multiple_files=True)

        if uploaded_files and st.button("📤 Preprocess, Augment & Train XGBoost", type="primary", use_container_width=True):
            all_raw = []
            errors = []

            for file in uploaded_files:
                try:
                    df = pd.read_csv(file)
                    if "target" not in df.columns:
                        errors.append(f"{file.name}: No 'target' column")
                        continue
                    all_raw.append(df)
                except Exception as e:
                    errors.append(f"{file.name}: Read error – {str(e)}")

            if errors:
                st.error("\n".join(errors))
            elif all_raw:
                raw_data = pd.concat(all_raw, ignore_index=True)

                feature_rows = []
                for idx, row_series in raw_data.iterrows():
                    row_df = pd.DataFrame([row_series.to_dict()])
                    feats = extract_time_series_features(row_df)
                    if feats:
                        feats["target"] = row_series["target"]
                        feature_rows.append(feats)

                if not feature_rows:
                    st.error("No valid time-series data found after feature extraction.")
                    st.stop()

                df_features = pd.DataFrame(feature_rows)

                if len(df_features) < 200:
                    st.info(f"Small dataset ({len(df_features)} samples) → applying Gaussian noise augmentation")
                    aug_rows = []
                    for _ in range(3):
                        aug = df_features[FEATURE_COLUMNS].copy()
                        noise = np.random.normal(0, 0.035, aug.shape)
                        aug_noisy = aug * (1 + noise)
                        aug_noisy["target"] = df_features["target"]
                        aug_rows.append(aug_noisy)
                    df_aug = pd.concat(aug_rows, ignore_index=True)
                    df_features = pd.concat([df_features, df_aug], ignore_index=True)

                X = df_features[FEATURE_COLUMNS]
                y = df_features["target"]

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                joblib.dump(scaler, "e_nose_feature_scaler.pkl")

                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y, test_size=0.2, stratify=y, random_state=42
                )

                model = xgb.XGBClassifier(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="auc",
                    early_stopping_rounds=30
                )

                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )

                joblib.dump(model, "e_nose_xgboost_model.pkl")

                explainer = shap.TreeExplainer(model)
                joblib.dump(explainer, "e_nose_shap_explainer.pkl")

                auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                st.success(f"Model trained successfully!\nSamples after augmentation: {len(df_features)}\nValidation AUC: {auc:.3f}")

# ==================== PATIENT MODE ====================
else:
    tab1, tab2 = st.tabs([trans["breath_title"], trans["sensors_tab"]])

    with tab1:
        st.subheader("Input Method – e-nose Time-Series Data")
        st.info("For full accuracy, upload a CSV with 12 sensors × 29 time points. Manual input is simplified to steady-state only.")

        input_method = st.radio("", [trans["manual"], trans["upload"]], horizontal=True)

        sensor_steady = {}
        if input_method == trans["manual"]:
            st.write("Demo: only steady-state values")
            cols = st.columns(3)
            for i, s in enumerate(SENSOR_NAMES):
                with cols[i % 3]:
                    sensor_steady[s] = st.slider(f"{s} steady-state (Ω)", 500, 8000, 2500, step=50)

        else:
            uploaded = st.file_uploader(trans["upload"], type="csv")
            if uploaded:
                try:
                    df = pd.read_csv(uploaded)
                    missing = [f"{s}_t{i+1}" for s in SENSOR_NAMES for i in range(TIME_POINTS) if f"{s}_t{i+1}" not in df.columns]
                    if missing:
                        st.error(f"Missing columns: {', '.join(missing[:5])}...")
                    else:
                        st.success(f"Loaded time-series data ({len(df)} samples)")
                        st.session_state.sensor_timeseries_df = df
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")

        # Model status check
        model_ready = False
        try:
            joblib.load("e_nose_xgboost_model.pkl")
            model_ready = True
        except:
            pass

        if model_ready:
            st.success(trans["model_ready"])
        else:
            st.warning(trans["no_model"])

        if st.button(trans["analyze_btn"], type="primary", use_container_width=True):
            model_path = "e_nose_xgboost_model.pkl"
            scaler_path = "e_nose_feature_scaler.pkl"
            explainer_path = "e_nose_shap_explainer.pkl"

            if not all(os.path.exists(p) for p in [model_path, scaler_path, explainer_path]):
                st.error(trans["no_model"])
                st.stop()

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            explainer = None
            if os.path.exists(explainer_path):
                try:
                    explainer = joblib.load(explainer_path)
                except Exception as e:
                    st.warning("Saved explainer is incompatible (version mismatch). Creating a fresh explainer from the model. This may take a moment.")
                    explainer = shap.TreeExplainer(model)
            else:
                # If no explainer file exists, create one (optional)
                explainer = shap.TreeExplainer(model)

            if input_method == trans["manual"]:
                row = {f"{s}_steady": v for s, v in sensor_steady.items()}
                for s in SENSOR_NAMES:
                    steady = row[f"{s}_steady"]
                    time_axis = np.linspace(0, 10, TIME_POINTS)
                    curve = steady / (1 + np.exp(-1.0*(time_axis - 4))) + np.random.normal(0, 20, TIME_POINTS)
                    for j, val in enumerate(curve, 1):
                        row[f"{s}_t{j}"] = val
                    row[f"{s}_rise_time"] = 10
                    row[f"{s}_max_slope"] = 50
                    row[f"{s}_auc"] = steady * TIME_POINTS
                    row[f"{s}_mean"] = steady
                    row[f"{s}_std"] = 100
                    row[f"{s}_max"] = steady + 200
                    row[f"{s}_min"] = steady - 200
                    row[f"{s}_fft1"] = 100
                    row[f"{s}_fft2"] = 50
                    row[f"{s}_fft3"] = 30
                    row[f"{s}_fft4"] = 10
                    row[f"{s}_dominant_freq"] = 0.1
                input_df = pd.DataFrame([row])
                is_batch = False
            else:
                if 'sensor_timeseries_df' not in st.session_state:
                    st.warning("Please upload a valid CSV first.")
                    st.stop()
                input_df = st.session_state.sensor_timeseries_df
                is_batch = len(input_df) > 1

            feature_list = []
            for _, row_series in input_df.iterrows():
                row_df = pd.DataFrame([row_series.to_dict()])
                feats = extract_time_series_features(row_df)
                if feats:
                    feature_list.append(feats)

            if not feature_list:
                st.error("Could not extract valid features from input. Check CSV format or use manual mode.")
                st.stop()

            df_features = pd.DataFrame(feature_list)
            X_scaled = scaler.transform(df_features[FEATURE_COLUMNS])

            probs = model.predict_proba(X_scaled)[:, 1] * 100

            results = []
            for idx, prob in enumerate(probs):
                total_risk = int(0.6 * prob + 0.4 * baseline_score)
                level = trans["high"] if total_risk >= 70 else trans["medium"] if total_risk >= 40 else trans["low"]
                color = "#ff4b4b" if total_risk >= 70 else "#ffaa00" if total_risk >= 40 else "#00cc00"

                if is_batch:
                    results.append({
                        "Sample": idx + 1,
                        "Breathprint Risk (%)": f"{prob:.3g}",
                        "Total Risk (%)": total_risk,
                        "Risk Level": level
                    })
                else:
                    st.markdown(f"<h2 style='color:{color}'>{trans['risk']}: {total_risk}% — {level}</h2>", unsafe_allow_html=True)
                    st.progress(total_risk / 100)
                    st.write(f"**{trans['breath_risk']}**: {prob:.3g}%")
                    st.write(f"**{trans['baseline_risk']}**: {baseline_score}%")

                    st.subheader("Key Factors Driving This Risk Assessment (SHAP)")

                    shap_values_single = explainer.shap_values(X_scaled[idx:idx+1])[0]

                    # Readable feature names
                    readable_names = []
                    for col in FEATURE_COLUMNS:
                        sensor, feat = col.split('_', 1)
                        feat = feat.replace('_', ' ').title()
                        readable_names.append(f"{sensor} {feat}")

                    shap_df = pd.DataFrame({
                        'Feature': readable_names,
                        'SHAP Value': shap_values_single
                    })

                    shap_df['Abs'] = shap_df['SHAP Value'].abs()
                    shap_df = shap_df.sort_values('Abs', ascending=False).head(8).drop(columns='Abs')

                    fig, ax = plt.subplots(figsize=(12, 6))  # wider figure
                    colors = ['#d62728' if x > 0 else '#2ca02c' for x in shap_df['SHAP Value']]

                    bars = ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors, height=0.7)  # thinner bars

                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2,
                                f'{width:+.3f}',
                                ha='left' if width > 0 else 'right',
                                va='center',
                                fontsize=9,
                                color='white' if abs(width) > 0.05 else 'black')

                    ax.axvline(0, color='gray', linewidth=0.8)
                    ax.set_xlabel('Impact on predicted risk (positive = increases risk)')
                    ax.set_title('Top Factors Influencing This Breathprint Result')
                    ax.invert_yaxis()
                    plt.xticks(fontsize=9)
                    plt.yticks(fontsize=10, rotation=0, ha='right')  # readable y labels
                    plt.tight_layout(pad=2.0)  # extra padding
                    sns.set_style("whitegrid")
                    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                    plt.close(fig)

                    # PDF generation
                    buffer = BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                    elements = []

                    elements.append(Paragraph(trans["pdf_title"], styles['Title']))
                    elements.append(Spacer(1, 12))
                    elements.append(Paragraph(f"{trans['pdf_date']}: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
                    elements.append(Spacer(1, 24))

                    elements.append(Paragraph(trans["pdf_patient"], styles['Heading2']))
                    patient_data = [
                        [trans["age"], str(age or "N/A")],
                        [trans["gender"], gender or "N/A"],
                        [trans["cooking"], cooking or "N/A"],
                        [trans["air_days"], str(air_days or "N/A")],
                        [trans["family"], family or "N/A"],
                        [trans["baseline_risk"], f"{baseline_score}%"]
                    ]
                    patient_table = Table(patient_data, colWidths=[180, 300])
                    patient_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), lib_colors.lightgrey),
                        ('GRID', (0,0), (-1,-1), 0.5, lib_colors.grey),
                    ]))
                    elements.append(patient_table)
                    elements.append(Spacer(1, 24))

                    elements.append(Paragraph(trans["pdf_overall"], styles['Heading2']))
                    risk_text = f"<font color='{color}'>{total_risk}% — {level}</font>"
                    elements.append(Paragraph(risk_text, styles['Heading3']))
                    elements.append(Spacer(1, 12))

                    if not is_batch:
                        elements.append(Paragraph(trans["pdf_sensor_table"], styles['Heading2']))
                        sensor_rows = [["Sensor", "Steady-State (Ω)"]]
                        for s in SENSOR_NAMES:
                            val = sensor_steady.get(s, "N/A") if input_method == trans["manual"] else "N/A"
                            sensor_rows.append([s, f"{val:.0f}" if isinstance(val, (int, float)) else val])
                        sensor_table = Table(sensor_rows, colWidths=[200, 200])
                        sensor_table.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), lib_colors.lightblue),
                            ('GRID', (0,0), (-1,-1), 0.5, lib_colors.grey),
                        ]))
                        elements.append(sensor_table)
                        elements.append(Spacer(1, 24))

                        elements.append(Paragraph(trans["pdf_shap_table"], styles['Heading2']))
                        shap_df_pdf = pd.DataFrame({
                            "Feature": readable_names,
                            "SHAP Value": shap_values_single
                        }).sort_values("SHAP Value", key=abs, ascending=False).head(10)
                        shap_rows = [["Feature", "SHAP Contribution"]]
                        for _, r in shap_df_pdf.iterrows():
                            shap_rows.append([r["Feature"], f"{r['SHAP Value']:.3f}"])
                        shap_table = Table(shap_rows, colWidths=[300, 200])
                        shap_table.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), lib_colors.lightblue),
                            ('GRID', (0,0), (-1,-1), 0.5, lib_colors.grey),
                        ]))
                        elements.append(shap_table)
                    else:
                        elements.append(Paragraph("Batch Analysis Summary", styles['Heading2']))
                        batch_data = [[r["Sample"], r['Breathprint Risk (%)'], f"{r['Total Risk (%)']}%", r["Risk Level"]] for r in results]
                        batch_table = Table([["Sample", "Breathprint Risk (%)", "Total Risk (%)", "Risk Level"]] + batch_data)
                        batch_table.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), lib_colors.lightblue),
                            ('GRID', (0,0), (-1,-1), 0.5, lib_colors.grey),
                        ]))
                        elements.append(batch_table)

                    elements.append(Spacer(1, 24))
                    elements.append(Paragraph(trans["pdf_recommend"], styles['Heading2']))
                    elements.append(Paragraph(trans["pdf_recommend_text"], styles['Normal']))

                    doc.build(elements)
                    buffer.seek(0)

                    st.download_button(
                        label="📄 Download PDF Report",
                        data=buffer,
                        file_name=f"LungGuard_e-nose_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

            if is_batch and results:
                st.subheader("Batch Results")
                st.dataframe(pd.DataFrame(results), use_container_width=True)

    with tab2:
        st.markdown("**12 MOS Sensors + Time-Series + FFT Features**")
        st.info("The model extracts rise time, slope, AUC, and FFT frequency components from each sensor's 29-point response curve.")
        st.dataframe(pd.DataFrame({"Feature Type": [
            "Steady-state", "Rise time", "Max slope", "AUC", "Mean", "Std", "Max", "Min",
            "FFT1", "FFT2", "FFT3", "FFT4", "Dominant frequency"
        ]}), use_container_width=True, hide_index=True)

# ==================== BOTTOM WARNING & FOOTER ====================
warning_text = trans.get('prototype_warning', '⚠️ Research prototype only • Not a medical device • For pilot testing purposes')
st.markdown(
    "<div style='background-color:#ffebee; padding:12px; border-radius:6px; margin: 32px 0 16px 0; text-align:center;'>"
    f"<strong>{warning_text}</strong>"
    "</div>",
    unsafe_allow_html=True
)

st.caption(trans.get("footer", "LungGuard HK v4.3 • Research Prototype Only"))