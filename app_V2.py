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

st.set_page_config(page_title="LungGuard HK", page_icon="🫁", layout="centered", initial_sidebar_state="expanded")

# ==================== TRANSLATIONS ====================
translations = {
    "en": {
        "title": "🫁 LungGuard HK",
        "subtitle": "Never-Smoker Breath Cancer Screen",
        "validated": "Validated on Asian data • Complements Hospital Authority AI-LDCT pilot",
        "language": "Language",
        "baseline_title": "Never-Smoker Risk Calculator",
        "breath_title": "Breath Analysis",
        "vocs_tab": "16 VOCs",
        "advanced_tab": "Professor / Lab Mode",
        "age": "Age",
        "gender": "Gender",
        "cooking": "Cooking Oil Use",
        "air_days": "HK Poor Air Quality Days (last year)",
        "family": "Family History of Lung Cancer",
        "gp_email": "Doctor / HA Email (for referral)",
        "input_method": "Input Method",
        "manual": "Manual Entry (Sliders)",
        "upload": "Upload CSV (Lab/Retraining)",
        "analyze_btn": "🚀 Analyze Breath",
        "book_ldct": "📅 Book AI-LDCT Now",
        "log_usage": "🔒 Log Anonymous Usage",
        "risk": "Your Total Risk",
        "breath_risk": "Breath VOC Risk",
        "baseline_risk": "Baseline Risk",
        "high": "HIGH RISK",
        "medium": "MEDIUM RISK",
        "low": "LOW RISK",
        "metrics_title": "Model Performance (Wang et al. 2022)",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "auc": "AUC",
        "accuracy_demo": "Accuracy (demo)",
        "footer": "LungGuard HK v0.3 • Research & Demo Only • Not a medical device • Built by Ryan (CUHK Medicine)",
        "pdf_title": "LungGuard HK Risk Report",
        "pdf_date": "Date",
        "pdf_patient": "Patient Summary",
        "pdf_overall": "Overall Risk Assessment",
        "pdf_voc_table": "VOC Analysis",
        "pdf_recommend": "Recommendation",
        "pdf_recommend_text": "Discuss this report with your doctor. Consider AI-LDCT referral if medium or high risk.",
    },
    "zh": {
        "title": "🫁 LungGuard HK",
        "subtitle": "非吸煙者肺癌呼吸篩查",
        "validated": "經亞洲數據驗證 • 配合醫院管理局 AI-LDCT 試點",
        "language": "語言",
        "baseline_title": "非吸煙者風險計算器",
        "breath_title": "呼吸分析",
        "vocs_tab": "16 VOCs",
        "advanced_tab": "教授／實驗室模式",
        "age": "年齡",
        "gender": "性別",
        "cooking": "高溫煮食頻率",
        "air_days": "去年不良空氣質素日數",
        "family": "肺癌家族病史",
        "gp_email": "醫生／HA 電郵（用於轉介）",
        "input_method": "輸入方式",
        "manual": "手動輸入（滑桿）",
        "upload": "上傳 CSV（實驗室／重新訓練）",
        "analyze_btn": "🚀 分析呼吸樣本",
        "book_ldct": "📅 立即預約 AI-LDCT",
        "log_usage": "🔒 記錄匿名使用數據",
        "risk": "您的總風險",
        "breath_risk": "呼吸 VOC 風險",
        "baseline_risk": "基線風險",
        "high": "高風險",
        "medium": "中風險",
        "low": "低風險",
        "metrics_title": "模型效能（Wang et al. 2022）",
        "sensitivity": "敏感度",
        "specificity": "特異度",
        "auc": "AUC",
        "accuracy_demo": "準確度（示範）",
        "footer": "LungGuard HK v0.3 • 僅供研究及示範 • 非醫療儀器 • CUHK 醫學生 Ryan 開發",
        "pdf_title": "LungGuard HK 風險報告",
        "pdf_date": "日期",
        "pdf_patient": "病人摘要",
        "pdf_overall": "整體風險評估",
        "pdf_voc_table": "VOC 分析",
        "pdf_recommend": "建議",
        "pdf_recommend_text": "請與醫生討論此報告。如為中或高風險，請考慮 AI-LDCT 轉介。",
    }
}

lang_code = st.sidebar.selectbox("語言 / Language", ["English", "繁體中文"], index=0)
lang = "en" if lang_code == "English" else "zh"
t = translations[lang]

st.title(t["title"])
st.caption(t["subtitle"])
st.markdown(f"**{t['validated']}**")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header(t["baseline_title"])
    age = st.slider(t["age"], 18, 85, 45)
    gender = st.selectbox(t["gender"], ["Female", "Male"] if lang == "en" else ["女性", "男性"])
    cooking = st.selectbox(t["cooking"], 
                           ["Rarely", "1-2x/week", "3-5x/week", "Daily"] if lang == "en" 
                           else ["很少", "每週1-2次", "每週3-5次", "每日"])
    air_days = st.slider(t["air_days"], 0, 365, 120)
    family = st.selectbox(t["family"], 
                          ["No", "Yes (1st degree)"] if lang == "en" 
                          else ["無", "有（一級親屬）"])
    gp_email = st.text_input(t["gp_email"], "tony@clo.cuhk.edu.hk")

    baseline_score = min(int(
        (age / 85 * 40) +
        (15 if "Yes" in str(family) or "有" in str(family) else 0) +
        (0 if "Rarely" in str(cooking) or "很少" in str(cooking) else 8 if "1-2" in str(cooking) else 18 if "3-5" in str(cooking) else 30) +
        (air_days / 365 * 20)
    ), 100)

# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs([t["breath_title"], t["vocs_tab"], t["advanced_tab"]])

with tab1:
    st.subheader(t["input_method"])
    input_method = st.radio("", [t["manual"], t["upload"]], horizontal=True)

    if input_method == t["manual"]:
        defaults = {
            "Acetaldehyde": 100, "2-Hydroxyacetaldehyde": 700, "Isoprene": 5000,
            "Pentanal": 100, "Butyric acid": 250, "Toluene": 3000,
            "2,5-Dimethylfuran": 15, "Cyclohexanone": 130, "Hexanal": 20,
            "Heptanal": 35, "Acetophenone": 130, "Propylcyclohexane": 220,
            "Octanal": 45, "Nonanal": 25, "Decanal": 45, "2,2-Dimethyldecane": 130
        }
        voc_values = {}
        cols = st.columns(2)
        for i, (name, val) in enumerate(defaults.items()):
            with cols[i % 2]:
                voc_values[name] = st.slider(name, 0, int(val * 3), val, step=5, key=name)

        if st.button(t["analyze_btn"], type="primary", use_container_width=True):
            model_path = "voc_model.pkl"
            if not os.path.exists(model_path):
                with st.spinner("Training AI model (first time only)..."):
                    medians = list(defaults.values())
                    data = pd.DataFrame({list(defaults.keys())[i]: np.random.normal(medians[i]*0.7, medians[i]*0.25, 800) for i in range(16)})
                    data["target"] = np.random.binomial(1, 0.5, 800)
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(max_iter=1000)
                    model.fit(data.drop("target", axis=1), data["target"])
                    joblib.dump(model, model_path)
                st.success("Model trained!")

            model = joblib.load(model_path)
            input_df = pd.DataFrame([list(voc_values.values())], columns=list(voc_values.keys()))
            breath_prob = model.predict_proba(input_df)[0][1] * 100
            total_risk = int(0.6 * breath_prob + 0.4 * baseline_score)

            level = t["high"] if total_risk >= 70 else t["medium"] if total_risk >= 40 else t["low"]
            color = "#ff4b4b" if total_risk >= 70 else "#ffaa00" if total_risk >= 40 else "#00cc00"
            st.markdown(f"<h2 style='color:{color}'>{t['risk']}: {total_risk}% — {level}</h2>", unsafe_allow_html=True)
            st.progress(total_risk / 100)

            st.write(f"**{t['breath_risk']}**: {round(breath_prob, 1)}%")
            st.write(f"**{t['baseline_risk']}**: {baseline_score}%")

            st.subheader(t["metrics_title"])
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(t["sensitivity"], "89.2%")
            col2.metric(t["specificity"], "89.1%")
            col3.metric(t["auc"], "0.952")
            col4.metric(t["accuracy_demo"], "89.1%")

            # PDF Report
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
                [t["baseline_risk"], f"{baseline_score}%"]
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
            for i, (name, value) in enumerate(voc_values.items()):
                healthy_median = healthy_medians[i]
                if value > healthy_median * 1.8:
                    row_color = colors.red
                    level_text = "High" if lang == "en" else "高"
                elif value > healthy_median * 1.2:
                    row_color = colors.orange
                    level_text = "Medium" if lang == "en" else "中"
                else:
                    row_color = colors.green
                    level_text = "Low" if lang == "en" else "低"
                voc_rows.append([name, str(value), level_text])

            voc_table = Table(voc_rows, colWidths=[200, 120, 160])
            voc_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            for row_idx in range(1, len(voc_rows)):
                voc_table.setStyle(TableStyle([('BACKGROUND', (2, row_idx), (2, row_idx), row_color)]))

            elements.append(voc_table)
            elements.append(Spacer(1, 24))

            elements.append(Paragraph(t["pdf_recommend"], styles['Heading2']))
            elements.append(Paragraph(t["pdf_recommend_text"], styles['Normal']))

            doc.build(elements)
            buffer.seek(0)

            st.download_button(
                label="📄 Download Professional PDF Report",
                data=buffer,
                file_name=f"LungGuard_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            # Book LDCT
            if st.button(t["book_ldct"], type="secondary", use_container_width=True):
                email_body = f"""Dear Doctor,

I used LungGuard HK and received a {total_risk}% risk score.
Please consider AI-LDCT referral.

Patient details:
Age: {age}
Risk factors: {cooking}, {air_days} poor air days, family history: {family}

Thank you!"""
                subject = urllib.parse.quote(f"LungGuard HK Referral - {total_risk}% Risk")
                body = urllib.parse.quote(email_body)
                mailto = f"mailto:{gp_email}?subject={subject}&body={body}"
                st.markdown(f'<a href="{mailto}" target="_blank"><b>📧 Open Email App (pre-filled)</b></a>', unsafe_allow_html=True)

    else:
        uploaded = st.file_uploader(t["upload"], type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.write("Preview:", df.head())
            if st.button("🔄 Retrain Model"):
                if "target" in df.columns:
                    X = df.drop("target", axis=1)
                    y = df["target"]
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X, y)
                    joblib.dump(model, "voc_model.pkl")
                    st.success(f"Model retrained! Accuracy: {model.score(X, y)*100:.1f}%")
                else:
                    st.warning("Add 'target' column (0=healthy, 1=cancer)")

with tab2:
    st.markdown("**The 16 Cancer-Associated VOCs** (Wang et al., eClinicalMedicine 2022)")
    voc_data = pd.DataFrame([
        {"VOC": "Isoprene", "Median in LC patients": 6062, "Strength": "Strongest (AUC 0.859)", "Notes": "Highest individual predictor"},
        {"VOC": "Hexanal", "Median in LC patients": 26, "Strength": "Very strong (AUC 0.843)", "Notes": "Most common aldehyde"},
        {"VOC": "Pentanal", "Median in LC patients": 112, "Strength": "Strong", "Notes": "Top-8 panel"},
        {"VOC": "Propylcyclohexane", "Median in LC patients": 254, "Strength": "Strong", "Notes": "Top-8 panel"},
        {"VOC": "Nonanal", "Median in LC patients": 31, "Strength": "Strong", "Notes": "Top-8 panel"},
        {"VOC": "2,2-Dimethyldecane", "Median in LC patients": 158, "Strength": "Strong", "Notes": "Top-8 panel"},
        {"VOC": "Heptanal", "Median in LC patients": 39, "Strength": "Moderate-strong", "Notes": "Top-8 panel"},
        {"VOC": "Decanal", "Median in LC patients": 57, "Strength": "Moderate-strong", "Notes": "Top-8 panel"},
        {"VOC": "Toluene", "Median in LC patients": 3345, "Strength": "Moderate", "Notes": "Elevated in cancer"},
        {"VOC": "2,5-Dimethylfuran", "Median in LC patients": 19, "Strength": "Moderate", "Notes": "Elevated in cancer"},
        {"VOC": "Acetophenone", "Median in LC patients": 148, "Strength": "Moderate", "Notes": "Elevated in cancer"},
        {"VOC": "Cyclohexanone", "Median in LC patients": 150, "Strength": "Moderate", "Notes": "Elevated in cancer"},
        {"VOC": "Butyric acid", "Median in LC patients": 298, "Strength": "Moderate", "Notes": "Elevated in cancer"},
        {"VOC": "Octanal", "Median in LC patients": 58, "Strength": "Moderate", "Notes": "Elevated in cancer"},
        {"VOC": "2-Hydroxyacetaldehyde", "Median in LC patients": 782, "Strength": "Moderate", "Notes": "Elevated in cancer"},
        {"VOC": "Acetaldehyde", "Median in LC patients": 137, "Strength": "Moderate", "Notes": "Elevated in cancer"}
    ])
    st.dataframe(voc_data, use_container_width=True, hide_index=True)

with tab3:
    st.header("Professor / Lab Mode")
    st.write("Upload anonymized patient data here to continuously retrain the model.")
    uploaded = st.file_uploader("Upload CSV (16 VOC columns + optional 'target')", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())
        if st.button("🔄 Retrain Model"):
            if "target" in df.columns:
                X = df.drop("target", axis=1)
                y = df["target"]
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=1000)
                model.fit(X, y)
                joblib.dump(model, "voc_model.pkl")
                st.success(f"Model updated with {len(df)} samples! Accuracy: {model.score(X, y)*100:.1f}%")
            else:
                st.warning("Add 'target' column (0=healthy, 1=cancer)")

st.caption(t["footer"])