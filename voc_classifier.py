"""
Lung Cancer VOC Classifier
Based on Wang 2022 study - 16 VOC biomarkers for lung cancer detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, 
    accuracy_score, classification_report
)
import joblib
import os

# Wang 2022 Study: 16 VOCs Pre-surgery Medians (Lung Cancer Patients)
# Peak intensities (arbitrary units)
VOC_MEDIANS_LC = {
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

VOC_NAMES = list(VOC_MEDIANS_LC.keys())
VOC_MEDIANS_VALUES = np.array(list(VOC_MEDIANS_LC.values()))


def create_dataset(n_healthy=300, n_lc=200, random_seed=42):
    """
    Create synthetic dataset with normal distribution around medians.
    
    Args:
        n_healthy: Number of healthy samples (70% of LC levels)
        n_lc: Number of LC samples (full levels + noise)
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with VOC features and target column
    """
    np.random.seed(random_seed)
    
    # Healthy controls: 70% of LC levels + noise
    healthy_data = np.random.normal(
        loc=VOC_MEDIANS_VALUES * 0.7,
        scale=VOC_MEDIANS_VALUES * 0.15,  # 15% standard deviation
        size=(n_healthy, len(VOC_NAMES))
    )
    # Ensure non-negative values
    healthy_data = np.maximum(healthy_data, 0)
    healthy_labels = np.zeros(n_healthy)
    
    # LC patients: full levels + noise
    lc_data = np.random.normal(
        loc=VOC_MEDIANS_VALUES,
        scale=VOC_MEDIANS_VALUES * 0.2,  # 20% standard deviation
        size=(n_lc, len(VOC_NAMES))
    )
    # Ensure non-negative values
    lc_data = np.maximum(lc_data, 0)
    lc_labels = np.ones(n_lc)
    
    # Combine datasets
    X = np.vstack([healthy_data, lc_data])
    y = np.concatenate([healthy_labels, lc_labels])
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=VOC_NAMES)
    df['target'] = y.astype(int)
    
    return df


def calculate_metrics(y_true, y_pred, y_proba):
    """
    Calculate classification metrics.
    
    Returns:
        Dictionary with accuracy, sensitivity, specificity, AUC, and confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True positive rate / Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True negative rate
    auc_score = roc_auc_score(y_true, y_proba)
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc': auc_score,
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


def train_and_evaluate_models(df, test_size=0.3, random_seed=42):
    """
    Train LogisticRegression and RandomForest models and evaluate them.
    
    Returns:
        Dictionary with models, scaler, and metrics
    """
    # Prepare data
    X = df[VOC_NAMES].values
    y = df['target'].values
    
    # Split data (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=random_seed, solver='lbfgs')
    lr_model.fit(X_train_scaled, y_train)
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=random_seed, 
        n_jobs=-1,
        max_depth=10
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Predictions
    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    rf_pred = rf_model.predict(X_test_scaled)
    rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    lr_metrics = calculate_metrics(y_test, lr_pred, lr_proba)
    rf_metrics = calculate_metrics(y_test, rf_pred, rf_proba)
    
    # Print results
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION RESULTS")
    print("="*70)
    print(f"Accuracy:    {lr_metrics['accuracy']:.4f}")
    print(f"Sensitivity: {lr_metrics['sensitivity']:.4f}")
    print(f"Specificity: {lr_metrics['specificity']:.4f}")
    print(f"AUC:         {lr_metrics['auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Healthy    LC")
    print(f"Actual Healthy   {lr_metrics['tn']:4d}    {lr_metrics['fp']:4d}")
    print(f"        LC       {lr_metrics['fn']:4d}    {lr_metrics['tp']:4d}")
    
    print("\n" + "="*70)
    print("RANDOM FOREST RESULTS")
    print("="*70)
    print(f"Accuracy:    {rf_metrics['accuracy']:.4f}")
    print(f"Sensitivity: {rf_metrics['sensitivity']:.4f}")
    print(f"Specificity: {rf_metrics['specificity']:.4f}")
    print(f"AUC:         {rf_metrics['auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Healthy    LC")
    print(f"Actual Healthy   {rf_metrics['tn']:4d}    {rf_metrics['fp']:4d}")
    print(f"        LC       {rf_metrics['fn']:4d}    {rf_metrics['tp']:4d}")
    
    return {
        'lr_model': lr_model,
        'rf_model': rf_model,
        'scaler': scaler,
        'lr_metrics': lr_metrics,
        'rf_metrics': rf_metrics,
        'X_test': X_test,
        'y_test': y_test,
        'lr_proba': lr_proba,
        'rf_proba': rf_proba
    }


def plot_roc_curves(results, save_path='roc_curve.png'):
    """Plot ROC curves for both models."""
    y_test = results['y_test']
    lr_proba = results['lr_proba']
    rf_proba = results['rf_proba']
    lr_metrics = results['lr_metrics']
    rf_metrics = results['rf_metrics']
    
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
    
    plt.figure(figsize=(10, 7))
    plt.plot(fpr_lr, tpr_lr, 
             label=f'Logistic Regression (AUC={lr_metrics["auc"]:.3f})', 
             linewidth=2)
    plt.plot(fpr_rf, tpr_rf, 
             label=f'Random Forest (AUC={rf_metrics["auc"]:.3f})', 
             linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve: Lung Cancer VOC Classifier', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nROC curve saved to {save_path}")
    plt.close()


def save_models(results, model_path='voc_model.pkl', scaler_path='voc_scaler.pkl'):
    """
    Save the best model (Random Forest) and scaler.
    Uses Random Forest as default since it typically performs better.
    """
    joblib.dump(results['rf_model'], model_path)
    joblib.dump(results['scaler'], scaler_path)
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


def predict_risk(voc_values_dict, model_path='voc_model.pkl', scaler_path='voc_scaler.pkl'):
    """
    Predict lung cancer risk from VOC measurements.
    
    Args:
        voc_values_dict: Dictionary with VOC names as keys and peak intensities as values
        model_path: Path to saved model file
        scaler_path: Path to saved scaler file
    
    Returns:
        Dictionary with probability, risk category, and recommendation
    """
    # Check if model files exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Model files not found. Please train the model first. "
            f"Expected: {model_path}, {scaler_path}"
        )
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Convert dict to array in correct order
    voc_array = np.array([voc_values_dict.get(voc, 0.0) for voc in VOC_NAMES]).reshape(1, -1)
    
    # Ensure non-negative
    voc_array = np.maximum(voc_array, 0)
    
    # Scale and predict
    voc_scaled = scaler.transform(voc_array)
    probability = model.predict_proba(voc_scaled)[0, 1]
    
    # Determine risk category
    if probability >= 0.5:
        risk_category = "High risk - refer to LDCT"
        recommendation = "LDCT screening recommended"
    else:
        risk_category = "Low risk"
        recommendation = "Continue monitoring"
    
    return {
        'probability': float(probability),
        'risk_category': risk_category,
        'recommendation': recommendation
    }


def main():
    """Main execution function."""
    print("="*70)
    print("LUNG CANCER VOC CLASSIFIER")
    print("Based on Wang 2022 Study - 16 VOC Biomarkers")
    print("="*70)
    
    # Create dataset
    print("\nCreating synthetic dataset...")
    df = create_dataset(n_healthy=300, n_lc=200)
    
    print(f"\nDataset Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Healthy (0): {sum(df['target'] == 0)}")
    print(f"LC (1): {sum(df['target'] == 1)}")
    print(f"\nVOC Features ({len(VOC_NAMES)}):")
    print(df[VOC_NAMES].head())
    print(f"\nDataset statistics:")
    print(df[VOC_NAMES].describe())
    
    # Train and evaluate models
    results = train_and_evaluate_models(df)
    
    # Plot ROC curves
    plot_roc_curves(results)
    
    # Save models (using Random Forest as default)
    save_models(results)
    
    # Example prediction
    print("\n" + "="*70)
    print("PREDICTION EXAMPLE")
    print("="*70)
    np.random.seed(123)
    example_voc = {
        voc: np.random.normal(VOC_MEDIANS_LC[voc], VOC_MEDIANS_LC[voc] * 0.2) 
        for voc in VOC_NAMES
    }
    # Ensure non-negative
    example_voc = {k: max(0, v) for k, v in example_voc.items()}
    
    result = predict_risk(example_voc)
    print(f"\nExample VOC values:")
    for voc, val in example_voc.items():
        print(f"  {voc}: {val:.2f}")
    print(f"\nPrediction Result:")
    print(f"  Probability: {result['probability']:.4f}")
    print(f"  Risk Category: {result['risk_category']}")
    print(f"  Recommendation: {result['recommendation']}")
    
    print("\n" + "="*70)
    print("Script completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
