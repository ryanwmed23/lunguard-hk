# Lung Cancer VOC Classifier

A complete Python-based machine learning system for lung cancer detection using 16 Volatile Organic Compounds (VOCs) from exhaled breath analysis, based on the Wang 2022 study.

## Features

- **16 VOC Biomarkers**: Uses exact VOC names and pre-surgery medians from Wang 2022
- **Synthetic Dataset**: 500 samples (300 healthy at 70% of LC levels, 200 LC at full levels + noise)
- **Dual Models**: Logistic Regression and Random Forest classifiers
- **Comprehensive Metrics**: AUC, sensitivity, specificity, accuracy, and confusion matrix
- **Model Persistence**: Saves trained model as `voc_model.pkl`
- **Streamlit App**: Ready-to-use web interface for predictions
- **Risk Prediction Function**: `predict_risk()` returns probability and "High risk - refer to LDCT" recommendation

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

Run the main classifier script to train models and generate the dataset:

```bash
python voc_classifier.py
```

This will:
- Create a synthetic dataset with 500 samples
- Train both Logistic Regression and Random Forest models
- Display performance metrics (AUC, sensitivity, specificity, accuracy, confusion matrix)
- Save the best model (Random Forest) as `voc_model.pkl`
- Save the scaler as `voc_scaler.pkl`
- Generate a ROC curve plot (`roc_curve.png`)

### 2. Use the Prediction Function

```python
from voc_classifier import predict_risk

# Example VOC values dictionary
voc_values = {
    'Butyraldehyde': 85.2,
    'Butyric_acid': 92.5,
    'Dicyclohexyl_ketone': 45.8,
    'Ethane': 38.4,
    'Toluene': 52.1,
    'Carbon_disulfide': 61.3,
    'Sevoflurane': 29.7,
    'Acetone': 105.6,
    'Benzene': 41.2,
    'Pentane': 73.4,
    'Hexane': 58.9,
    'Heptane': 47.3,
    'Octane': 35.6,
    'Isoprene': 88.1,
    'Dimethyl_sulfide': 44.2,
    'Indole': 22.5
}

result = predict_risk(voc_values)
print(f"Probability: {result['probability']:.4f}")
print(f"Risk Category: {result['risk_category']}")
print(f"Recommendation: {result['recommendation']}")
```

### 3. Run Streamlit App

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

The app provides:
- Interactive input fields for all 16 VOCs
- Real-time risk prediction
- Visual probability display
- Example value loaders for testing

## Dataset Details

- **Total Samples**: 500
- **Healthy Samples**: 300 (target = 0)
  - VOC levels: 70% of LC medians
  - Normal distribution with 15% standard deviation
- **LC Samples**: 200 (target = 1)
  - VOC levels: Full medians from Wang 2022
  - Normal distribution with 20% standard deviation

## 16 VOC Biomarkers

1. Butyraldehyde
2. Butyric acid
3. Dicyclohexyl ketone
4. Ethane
5. Toluene
6. Carbon disulfide
7. Sevoflurane
8. Acetone
9. Benzene
10. Pentane
11. Hexane
12. Heptane
13. Octane
14. Isoprene
15. Dimethyl sulfide
16. Indole

## Model Performance

The script trains and evaluates both models:
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Typically higher performance, saved as default model

Metrics reported:
- **Accuracy**: Overall classification accuracy
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate
- **AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification breakdown

## Output Files

- `voc_model.pkl`: Trained Random Forest model
- `voc_scaler.pkl`: StandardScaler for feature normalization
- `roc_curve.png`: ROC curve visualization

## Reference

Based on Wang 2022 study on volatile organic compounds in exhaled breath for lung cancer detection.

## License

This project is for research and educational purposes.
