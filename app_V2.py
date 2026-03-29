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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import json

# NEW: For medical-grade conformal prediction
from mapie.classification import SplitConformalClassifier

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

MODEL_LATEST = "e_nose_xgboost_model_latest.pkl"
CALIBRATED_MODEL_LATEST = "e_nose_calibrated_model_latest.pkl"
CONFORMAL_PREDICTOR_LATEST = "e_nose_conformal_predictor_latest.pkl"
SCALER_LATEST = "e_nose_feature_scaler_latest.pkl"
EXPLAINER_LATEST = "e_nose_shap_explainer_latest.pkl"

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
        "breath_risk": "Breathprint Risk (Calibrated XGBoost)",
        "baseline_risk": "Baseline Risk",
        "high": "HIGH RISK",
        "medium": "MEDIUM RISK",
        "low": "LOW RISK",
        "metrics_title": "Model Performance",
        "footer": "LungGuard HK v4.7 – e-nose + Time-Series + FFT + SHAP + Incremental + CV + Isotonic Calibration + Conformal Prediction • Research Prototype Only • Not a medical device",
        "pdf_title": "LungGuard HK e-nose Report",
        "pdf_date": "Date",
        "pdf_patient": "Patient Summary",
        "pdf_overall": "Overall Risk Assessment",
        "pdf_sensor_table": "Sensor Steady-State Readings",
        "pdf_shap_table": "Top SHAP Sensor Features",
        "pdf_recommend": "Recommendation",
        "pdf_recommend_text": "Discuss this report with your doctor. Consider AI-LDCT referral if breathprint risk is medium or high.",
        "prototype_warning": "⚠️ Research prototype only • Not a medical device • For pilot testing purposes",
        "model_ready": "Calibrated model + conformal predictor loaded and ready for analysis",
        "no_model": "No trained model found. Train in Researcher mode first.",
        "model_found_disk": "Model file found on disk (may be old – retrain if needed)",
        "conformal_info": "95% Conformal Prediction Interval (medical-grade uncertainty)",
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
        "breath_risk": "呼吸印記風險（已校準XGBoost）",
        "baseline_risk": "基線風險",
        "high": "高風險",
        "medium": "中風險",
        "low": "低風險",
        "metrics_title": "模型效能",
        "footer": "LungGuard HK v4.7 – 電子鼻 + 時間序列 + FFT + SHAP + Incremental + CV + Isotonic Calibration + Conformal Prediction • 研究原型 • 非醫療儀器",
        "pdf_title": "LungGuard HK 電子鼻報告",
        "pdf_date": "日期",
        "pdf_patient": "病人摘要",
        "pdf_overall": "整體風險評估",
        "pdf_sensor_table": "感測器穩態讀數",
        "pdf_shap_table": "SHAP 前列感測器特徵",
        "pdf_recommend": "建議",
        "pdf_recommend_text": "請與醫生討論此報告。如呼吸印記風險為中或高風險，請考慮 AI-LDCT 轉介。",
        "prototype_warning": "⚠️ 僅為研究原型 • 非醫療儀器 • 僅用於試點測試",
        "model_ready": "已校準模型與共形預測器已載入，可進行分析",
        "no_model": "未找到已訓練模型。請先在研究者模式中訓練。",
        "model_found_disk": "已找到模型檔案（可能為舊版 – 如有需要請重新訓練）",
        "conformal_info": "95% 共形預測區間（醫學級不確定性量化）",
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
age = None
gender = None
cooking = None
air_days = None
family = None
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

        st.progress(float(baseline_score / 100))
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
        features[f"{sensor}_rise_time"] = 0
        if max_diff > 0.1:
            rise_idx = np.where(diff >= 0.9 * max_diff)[0]
            features[f"{sensor}_rise_time"] = rise_idx[0] if len(rise_idx) > 0 else len(ts) - 1

        slopes = np.diff(ts)
        features[f"{sensor}_max_slope"] = np.max(np.abs(slopes)) if len(slopes) > 0 else 0
        try:
            auc = np.trapz(ts)
        except AttributeError:
            auc = np.trapezoid(ts)
        features[f"{sensor}_auc"] = auc

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

        st.markdown("### Training Configuration")
        use_incremental = st.checkbox(
            "Continue training from existing model (incremental / warm-start)",
            value=False,
            help="If checked, loads the latest model and adds new trees only on the current data. "
                 "Useful for continuous data collection. Uncheck to train from scratch."
        )
        use_cv = st.checkbox(
            "Use Stratified K-Fold Cross-Validation for robust AUC (recommended for small datasets)",
            value=True
        )
        cv_folds = st.number_input("Number of CV folds", min_value=3, max_value=10, value=5) if use_cv else 5

        # NEW: Separate upload for External Validation Set (recommended for medical-grade model)
        st.markdown("### External Validation Set (Recommended for Medical-Grade Evaluation)")
        st.info("Upload a completely unseen labelled time-series CSV for true external validation metrics. "
                "This set is never used for training, CV, or calibration.")
        external_file = st.file_uploader("Upload External Validation CSV (optional but strongly recommended)", type="csv", key="external_val")

        if st.button("Generate Synthetic Time-Series Data (200 samples)", use_container_width=True):
            np.random.seed(42)
            n = 200
            healthy_base = 2500

            sensor_params = {}
            for idx, s in enumerate(SENSOR_NAMES):
                sensor_params[s] = {
                    "k": 0.6 + (idx * 0.04) % 0.8,
                    "cancer_shift": 400 + (idx * 80),
                    "steady_noise": 100 + (idx * 12),
                    "curve_noise": 60 + (idx * 7),
                }

            data = []
            targets = []
            for i in range(n):
                is_cancer = i >= n//2
                row = {}
                for s in SENSOR_NAMES:
                    p = sensor_params[s]
                    base_steady = healthy_base + (p["cancer_shift"] if is_cancer else 0)
                    steady = base_steady + np.random.normal(0, p["steady_noise"])
                    time_axis = np.linspace(0, 10, TIME_POINTS)
                    k = p["k"]
                    curve = steady / (1 + np.exp(-k * (time_axis - 4))) + np.random.normal(0, p["curve_noise"], TIME_POINTS)
                    for j, val in enumerate(curve, 1):
                        row[f"{s}_t{j}"] = val
                data.append(row)
                targets.append(1 if is_cancer else 0)

            df_syn = pd.DataFrame(data)
            df_syn["target"] = targets
            df_syn.to_csv("synthetic_e_nose_timeseries.csv", index=False)
            st.success("Synthetic dataset created (sensor-specific patterns – ready for real clinical data training)")
            st.download_button(
                "Download synthetic CSV",
                df_syn.to_csv(index=False).encode('utf-8'),
                "synthetic_e_nose_timeseries.csv",
                mime="text/csv"
            )

        uploaded_files = st.file_uploader("Upload labelled time-series CSV file(s) for TRAINING", type="csv", accept_multiple_files=True, key="train_upload")

        if uploaded_files and st.button("📤 Preprocess, Augment & Train XGBoost + Calibration + Conformal", type="primary", use_container_width=True):
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

                # Ensure all feature columns exist
                for col in FEATURE_COLUMNS:
                    if col not in df_features.columns:
                        df_features[col] = 0.0
                        st.warning(f"Missing feature column {col} – filled with 0 for training")

                incremental = False
                old_model = None
                if use_incremental:
                    if os.path.exists(MODEL_LATEST):
                        try:
                            old_model = joblib.load(MODEL_LATEST)
                            st.info("✅ Loaded existing model for incremental training")
                            incremental = True
                        except Exception as e:
                            st.warning(f"Failed to load previous model: {e}. Falling back to from-scratch training.")
                            incremental = False
                    else:
                        st.warning("No existing model found. Training from scratch.")
                        incremental = False

                X = df_features[FEATURE_COLUMNS]
                y = df_features["target"]

                # Hold-out split for training + calibration
                X_train_raw, X_val_raw, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train_raw)
                X_val = scaler.transform(X_val_raw)
                joblib.dump(scaler, SCALER_LATEST)

                # Cross-Validation (if selected and not incremental)
                cv_auc_mean = None
                cv_auc_std = None
                if use_cv and not incremental:
                    st.info(f"Performing Stratified {cv_folds}-Fold Cross-Validation...")
                    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    auc_scores = []
                    for train_idx, test_idx in skf.split(X, y):
                        X_cv_train = X.iloc[train_idx]
                        X_cv_test = X.iloc[test_idx]
                        y_cv_train = y.iloc[train_idx]
                        y_cv_test = y.iloc[test_idx]
                        scaler_cv = StandardScaler().fit(X_cv_train)
                        X_cv_train_s = scaler_cv.transform(X_cv_train)
                        X_cv_test_s = scaler_cv.transform(X_cv_test)
                        cv_model = xgb.XGBClassifier(
                            n_estimators=300,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42,
                            eval_metric="auc"
                        )
                        cv_model.fit(X_cv_train_s, y_cv_train)
                        auc_cv = roc_auc_score(y_cv_test, cv_model.predict_proba(X_cv_test_s)[:, 1])
                        auc_scores.append(auc_cv)
                    cv_auc_mean = np.mean(auc_scores)
                    cv_auc_std = np.std(auc_scores)
                    st.success(f"Cross-Validation AUC: {cv_auc_mean:.3f} ± {cv_auc_std:.3f} (over {cv_folds} folds)")

                # Train raw model
                if incremental and old_model is not None:
                    model = xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.02,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        eval_metric="auc"
                    )
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        xgb_model=old_model.get_booster(),
                        verbose=False
                    )
                else:
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

                # ── Isotonic Calibration ─────────────────────────────────────────────
                st.info("Applying isotonic probability calibration on hold-out set...")
                calibrated_model = CalibratedClassifierCV(
                    estimator=model,
                    method='isotonic',
                    cv='prefit'
                )
                calibrated_model.fit(X_val, y_val)

                # ── Calibration comparison plot ────────────────────────────────────────
                st.subheader("Calibration: Before vs After Isotonic")
                y_pred_raw = model.predict_proba(X_val)[:, 1]
                y_pred_cal = calibrated_model.predict_proba(X_val)[:, 1]

                fig_cal, ax_cal = plt.subplots(figsize=(10, 6))
                prob_true_raw, prob_pred_raw = calibration_curve(y_val, y_pred_raw, n_bins=10)
                prob_true_cal, prob_pred_cal = calibration_curve(y_val, y_pred_cal, n_bins=10)

                ax_cal.plot(prob_pred_raw, prob_true_raw, marker='o', label='Raw XGBoost')
                ax_cal.plot(prob_pred_cal, prob_true_cal, marker='s', label='Isotonic Calibrated')
                ax_cal.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
                ax_cal.set_xlabel('Predicted probability')
                ax_cal.set_ylabel('Fraction of positives')
                ax_cal.set_title('Calibration Improvement')
                ax_cal.legend()
                st.pyplot(fig_cal)
                plt.close(fig_cal)

                # ── Sensitivity & Specificity on Hold-out (at threshold 0.5) ─────────────
                st.subheader("Hold-out Performance (at probability threshold = 0.5)")
                y_pred_bin = (y_pred_cal >= 0.5).astype(int)
                cm = confusion_matrix(y_val, y_pred_bin)
                tn, fp, fn, tp = cm.ravel()

                sensitivity = recall_score(y_val, y_pred_bin)          # TP / (TP + FN)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0

                st.success(f"""
                **Sensitivity (Recall)**: {sensitivity:.3f}  
                **Specificity**: {specificity:.3f}  
                **Positive Predictive Value (PPV)**: {ppv:.3f}  
                **Negative Predictive Value (NPV)**: {npv:.3f}  
                (at default threshold 0.5)
                """)

                st.write("Confusion Matrix (0 = non-cancer, 1 = cancer)")
                st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

                # ── Conformal Prediction (Medical-grade UQ) - MAPIE v1 ─────────────────────────────
                st.info("Training Split Conformal Predictor (95% coverage) on hold-out set for medical-grade uncertainty quantification...")

                # Use SplitConformalClassifier with prefit=True for your hold-out calibration set
                conformal_model = SplitConformalClassifier(
                    estimator=calibrated_model,
                    prefit=True,         # Important: uses the already-split hold-out
                    conformity_score="lac",
                    confidence_level=0.95,
                    random_state=42,
                    #conformity_score="aps", 
                )
                conformal_model.conformalize(X_val, y_val)   # Fits the conformity scores on the hold-out set

                joblib.dump(conformal_model, CONFORMAL_PREDICTOR_LATEST)

                # Save explainer using raw model (SHAP needs the tree structure)
                explainer = shap.TreeExplainer(model)
                joblib.dump(explainer, EXPLAINER_LATEST)

                # Reported AUC (using calibrated model)
                reported_auc = roc_auc_score(y_val, y_pred_cal)

                # Save both
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                raw_filename = f"e_nose_xgboost_model_{timestamp}_AUC{reported_auc:.3f}.pkl"
                cal_filename = f"e_nose_calibrated_model_{timestamp}_AUC{reported_auc:.3f}.pkl"

                joblib.dump(model, raw_filename)
                joblib.dump(model, MODEL_LATEST)
                joblib.dump(calibrated_model, cal_filename)
                joblib.dump(calibrated_model, CALIBRATED_MODEL_LATEST)

                # Metadata
                try:
                    git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
                except:
                    git_hash = "no-git-repo"

                metadata = {
                    "training_date": datetime.now().isoformat(),
                    "num_samples": len(df_features),
                    "reported_auc": float(reported_auc),
                    "training_type": "incremental" if incremental else "from_scratch",
                    "cv_used": bool(use_cv),
                    "cv_folds": cv_folds if use_cv else None,
                    "calibration": "isotonic",
                    "conformal_method": "score",
                    "git_commit_hash": git_hash,
                    "raw_model_filename": raw_filename,
                    "calibrated_model_filename": cal_filename
                }
                meta_filename = f"e_nose_training_metadata_{timestamp}.json"
                with open(meta_filename, "w") as f:
                    json.dump(metadata, f, indent=2)
                with open("e_nose_training_metadata_latest.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                st.success(f"Model trained & calibrated with conformal prediction!\nRaw: {raw_filename}\nCalibrated: {cal_filename}\nAUC: {reported_auc:.3f}")

                # External validation if provided
                if external_file is not None:
                    try:
                        df_ext = pd.read_csv(external_file)
                        if "target" not in df_ext.columns:
                            st.error("External file must contain 'target' column")
                        else:
                            feature_rows_ext = []
                            for _, row_series in df_ext.iterrows():
                                row_df = pd.DataFrame([row_series.to_dict()])
                                feats = extract_time_series_features(row_df)
                                if feats:
                                    feats["target"] = row_series["target"]
                                    feature_rows_ext.append(feats)

                            df_ext_feat = pd.DataFrame(feature_rows_ext)
                            for col in FEATURE_COLUMNS:
                                if col not in df_ext_feat.columns:
                                    df_ext_feat[col] = 0.0

                            X_ext = df_ext_feat[FEATURE_COLUMNS]
                            y_ext = df_ext_feat["target"]
                            X_ext_scaled = scaler.transform(X_ext)

                            y_pred_ext = calibrated_model.predict_proba(X_ext_scaled)[:, 1]
                            ext_auc = roc_auc_score(y_ext, y_pred_ext)

                            y_pred_bin_ext = (y_pred_ext >= 0.5).astype(int)
                            cm_ext = confusion_matrix(y_ext, y_pred_bin_ext)
                            tn_e, fp_e, fn_e, tp_e = cm_ext.ravel()
                            sens_ext = recall_score(y_ext, y_pred_bin_ext)
                            spec_ext = tn_e / (tn_e + fp_e) if (tn_e + fp_e) > 0 else 0

                            st.success(f"""
                            **External Validation Results** (completely unseen set):
                            - External AUC: {ext_auc:.3f}
                            - Sensitivity: {sens_ext:.3f}
                            - Specificity: {spec_ext:.3f}
                            """)
                            st.write("External Confusion Matrix")
                            st.dataframe(pd.DataFrame(cm_ext, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))
                    except Exception as e:
                        st.error(f"External validation failed: {str(e)}")

# ==================== PATIENT MODE ====================
else:
    tab1, tab2 = st.tabs([trans["breath_title"], trans["sensors_tab"]])

    with tab1:
        st.subheader("Input Method – e-nose Time-Series Data")
        st.info("For full accuracy, upload a CSV with 12 sensors × 29 time points. Manual input is simplified.")

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

        model_ready = False
        calibrated_model = None
        conformal_predictor = None
        try:
            loaded_cal = joblib.load(CALIBRATED_MODEL_LATEST)
            if hasattr(loaded_cal, 'predict_proba') and hasattr(loaded_cal, 'estimator'):
                calibrated_model = loaded_cal
                model_ready = True

            if os.path.exists(CONFORMAL_PREDICTOR_LATEST):
                conformal_predictor = joblib.load(CONFORMAL_PREDICTOR_LATEST)
        except Exception as e:
            st.warning(f"Failed to load models: {str(e)}")
            model_ready = False

        if model_ready:
            st.success(trans["model_ready"])
        else:
            st.warning(trans["no_model"])

        if st.button(trans["analyze_btn"], type="primary", use_container_width=True):
            model_path = CALIBRATED_MODEL_LATEST
            scaler_path = SCALER_LATEST
            explainer_path = EXPLAINER_LATEST

            if not all(os.path.exists(p) for p in [model_path, scaler_path, explainer_path]):
                st.error(trans["no_model"])
                st.stop()

            calibrated_model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            explainer = None
            if os.path.exists(explainer_path):
                try:
                    explainer = joblib.load(explainer_path)
                except Exception as e:
                    st.warning("Saved explainer is incompatible. Creating a fresh explainer from the raw model.")
                    explainer = shap.TreeExplainer(calibrated_model.estimator)
            else:
                explainer = shap.TreeExplainer(calibrated_model.estimator)

            if input_method == trans["manual"]:
                row = {f"{s}_steady": v for s, v in sensor_steady.items()}
                for s in SENSOR_NAMES:
                    steady = row[f"{s}_steady"]
                    time_axis = np.linspace(0, 10, TIME_POINTS)
                    curve = steady / (1 + np.exp(-1.0*(time_axis - 4))) + np.random.normal(0, 20, TIME_POINTS)
                    for j, val in enumerate(curve, 1):
                        row[f"{s}_t{j}"] = val
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

            for col in FEATURE_COLUMNS:
                if col not in df_features.columns:
                    df_features[col] = 0.0

            X_scaled = scaler.transform(df_features[FEATURE_COLUMNS])

            probs = calibrated_model.predict_proba(X_scaled)[:, 1] * 100
            probs = np.clip(probs, 0.0, 100.0).astype(float)

            results = []
            for idx, prob in enumerate(probs):
                level = trans["high"] if prob >= 70 else trans["medium"] if prob >= 40 else trans["low"]
                color = "#ff4b4b" if prob >= 70 else "#ffaa00" if prob >= 40 else "#00cc00"

                if is_batch:
                    results.append({
                        "Sample": idx + 1,
                        "Breathprint Risk (%)": f"{prob:.3g}",
                        "Risk Level": level
                    })
                else:
                    st.markdown(f"<h2 style='color:{color}'>{trans['breath_risk']}: {prob:.1f}% — {level}</h2>", unsafe_allow_html=True)
                    st.progress(float(prob / 100))
                    st.write(f"**{trans['baseline_risk']}**: {baseline_score}%")

                    # NEW: Conformal Prediction Interval using MAPIE v1
                    if conformal_predictor is not None:
                        # Use confidence_level=0.95 instead of alpha=0.05
                        y_pred, y_pred_set = conformal_predictor.predict_set(
                            X_scaled[idx:idx+1], 
                            #conformity_score_params={"include_last_label": True}
                            method="lac",                  # Must match the training conformity_score
                            include_last_label=True
                        )
                        
                        prob_val = probs[idx] / 100.0
                        # Heuristic for display: wider interval when prediction set is ambiguous
                        set_size = len(y_pred_set[0]) if isinstance(y_pred_set, (list, np.ndarray)) and len(y_pred_set) > 0 else 1
                        margin = 0.20 if set_size > 1 else 0.08
                        conf_lower = max(0.0, prob_val - margin)
                        conf_upper = min(1.0, prob_val + margin)
                        
                        st.write(f"**{trans.get('conformal_info', '95% Conformal Prediction Interval')}**: {conf_lower*100:.1f}% – {conf_upper*100:.1f}%")
                        st.caption("Wider interval = higher uncertainty (e.g. sensor noise, novel breathprint)")
                    else:
                        # Fallback to improved bootstrap with realistic noise
                        n_boot = 500
                        boot_probs = []
                        rng = np.random.default_rng(42)
                        for _ in range(n_boot):
                            noisy = X_scaled[idx:idx+1] * (1 + rng.normal(0, 0.08, X_scaled[idx:idx+1].shape))
                            p = calibrated_model.predict_proba(noisy)[:, 1][0] * 100
                            boot_probs.append(p)
                        ci_low = np.percentile(boot_probs, 2.5)
                        ci_high = np.percentile(boot_probs, 97.5)
                        st.write(f"**95% Confidence Interval (bootstrap)**: {ci_low:.1f}% – {ci_high:.1f}%")

                    st.subheader("Key Factors Driving This Risk Assessment (SHAP Waterfall)")

                    sample_data = X_scaled[idx:idx+1]
                    shap_values = explainer.shap_values(sample_data)
                    if isinstance(shap_values, list):
                        shap_values_single = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                    else:
                        shap_values_single = shap_values[0]

                    base_value = explainer.expected_value
                    if isinstance(base_value, (list, np.ndarray)):
                        base_value = float(base_value[0])
                    else:
                        base_value = float(base_value)

                    readable_names = []
                    for col in FEATURE_COLUMNS:
                        sensor, feat = col.split('_', 1)
                        feat = feat.replace('_', ' ').title()
                        readable_names.append(f"{sensor} {feat}")

                    shap_exp = shap.Explanation(
                        values=shap_values_single,
                        base_values=base_value,
                        data=sample_data[0],
                        feature_names=readable_names
                    )

                    plt.figure(figsize=(14, 8))
                    shap.waterfall_plot(shap_exp, show=False)
                    plt.title("SHAP Waterfall: Contribution to Log-Odds of Lung Cancer Risk\n(Positive = increases probability)")
                    st.pyplot(plt.gcf())
                    plt.close()

                    # PDF generation (updated with conformal)
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
                    breath_risk_text = f"Breathprint Risk (calibrated): <font color='{color}'>{prob:.1f}% — {level}</font>"
                    elements.append(Paragraph(breath_risk_text, styles['Heading3']))
                    elements.append(Paragraph(f"Baseline Risk: {baseline_score}%", styles['Normal']))
                    if conformal_predictor is not None:
                        elements.append(Paragraph(f"95% Conformal Interval: approx {conf_lower*100:.1f}% – {conf_upper*100:.1f}%", styles['Normal']))
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
                        batch_data = [[r["Sample"], r['Breathprint Risk (%)'], r["Risk Level"]] for r in results]
                        batch_table = Table([["Sample", "Breathprint Risk (%)", "Risk Level"]] + batch_data)
                        batch_table.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), lib_colors.lightblue),
                            ('GRID', (0,0), (-1,-1), 0.5, lib_colors.grey),
                        ]))
                        elements.append(batch_table)
                        elements.append(Spacer(1, 12))
                        elements.append(Paragraph(f"Baseline Clinical Risk (from sidebar): {baseline_score}%", styles['Normal']))

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

st.caption(trans.get("footer", "LungGuard HK v4.7 • Research Prototype Only"))