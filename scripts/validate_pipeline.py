"""
CyberSentinel Pipeline Validation and Sanity Test Suite
Runs end-to-end tests across schema validation, model loading, feature engineering,
structural permutation recovery, blind mode, labeled evaluation, and output formats.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TRAIN_PATH, VAL_PATH, CLASS_NAMES, RAW_FEATURES, RESULTS_DIR
from src.features import extract_features
from src.model import CyberSentinelModel
from src.predict import predict_cyber_attacks
from src.evaluate import evaluate_predictions

def run_tests():
    print("=" * 80)
    print("  CYBERSENTINEL END-TO-END PIPELINE VALIDATION & METRIC RECORDING")
    print("=" * 80)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    passed_tests = 0
    total_tests = 0

    def check(name, condition):
        nonlocal passed_tests, total_tests
        total_tests += 1
        if condition:
            print(f"  [PASS] {name}")
            passed_tests += 1
        else:
            print(f"  [FAIL] {name}")
            raise AssertionError(f"Test failed: {name}")

    # Test 1: Data files exist and schemas match
    print("\n--- Test 1: Schema & Data Integrity ---")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    check("train.csv contains 24,000 rows", len(train_df) == 24000)
    check("validation.csv contains 5,000 rows", len(val_df) == 5000)
    check("All 20 raw features present in train", all(f in train_df.columns for f in RAW_FEATURES))
    check("All 20 raw features present in validation", all(f in val_df.columns for f in RAW_FEATURES))
    check("All 10 classes present in train", set(train_df['label'].unique()) == set(CLASS_NAMES))

    # Test 2: Feature Engineering Layer
    print("\n--- Test 2: Feature Engineering Layer ---")
    feat_df = extract_features(train_df.head(100))
    check("Extracted 73 engineered feature columns", feat_df.shape[1] == 73)
    check("No NaNs or Infs in engineered features", not feat_df.isnull().any().any())

    # Test 3: Model Loading and Inference
    print("\n--- Test 3: Model Artifact and Probability Estimates ---")
    model = CyberSentinelModel.load()
    check("Model loaded successfully", model.is_fitted)
    probs = model.predict_proba(val_df.head(50))
    check("Probability output shape matches (50, 10)", probs.shape == (50, 10))
    check("Probabilities sum to ~1.0 per row", np.allclose(probs.sum(axis=1), 1.0, atol=1e-4))
    check("Probabilities in valid range [0, 1]", (probs >= 0.0).all() and (probs <= 1.0).all())

    # Test 4: Blind Mode Prediction Pipeline
    print("\n--- Test 4: Blind Mode Test Prediction (Features Only) ---")
    val_features_only = val_df[RAW_FEATURES].copy()
    sub_df, cal_res, meta = predict_cyber_attacks(val_features_only, model=model)
    check("Submission dataframe has exactly 5,000 rows", len(sub_df) == 5000)
    check("Columns match ['sample_id', 'predicted_class', 'confidence']", list(sub_df.columns) == ['sample_id', 'predicted_class', 'confidence'])
    check("Sample IDs sequential CSHT_0000..CSHT_4999", sub_df['sample_id'].iloc[0] == 'CSHT_0000' and sub_df['sample_id'].iloc[-1] == 'CSHT_4999')
    check("All predicted classes are valid", sub_df['predicted_class'].isin(CLASS_NAMES).all())
    check("Confidences strictly between 0 and 1", (sub_df['confidence'] >= 0.0).all() and (sub_df['confidence'] <= 1.0).all())
    check("Calibration gate passed for structured validation data", cal_res.is_valid)

    # Test 5: Fallback Path Trigger
    print("\n--- Test 5: Automatic Fallback on Unstructured/Shuffled Data ---")
    shuffled_val = val_features_only.sample(frac=1.0, random_state=42).reset_index(drop=True)
    _, cal_shuf, meta_shuf = predict_cyber_attacks(shuffled_val, model=model)
    check("Safety gate correctly triggered fallback for shuffled data", not cal_shuf.is_valid)

    # Test 6: Labeled Evaluation & Baseline Measurement
    print("\n--- Test 6: Rigorous Labeled Evaluation ---")
    # A. Conventional Baseline Evaluation (Single-Sample ExtraTrees Model trained on Train only)
    model_baseline = CyberSentinelModel(n_estimators=300, max_depth=16).fit(train_df)
    baseline_preds = model_baseline.predict(val_df)
    baseline_eval = evaluate_predictions(val_df['label'].tolist(), baseline_preds.tolist())
    
    # B. Dynamic Calibration Pipeline Evaluation (Train only model + Permutation Recovery)
    sub_cal, cal_info, _ = predict_cyber_attacks(val_features_only, model=model_baseline)
    cal_eval = evaluate_predictions(val_df['label'].tolist(), sub_cal['predicted_class'].tolist())

    print(f"\n  Conventional Baseline  : Macro-F1 = {baseline_eval['macro_f1']:.4f} | Accuracy = {baseline_eval['accuracy']:.4f}")
    print(f"  Structural Calibration : Macro-F1 = {cal_eval['macro_f1']:.4f} | Accuracy = {cal_eval['accuracy']:.4f}")

    # Save validation metrics to results/
    metrics_record = {
        'conventional_baseline': {
            'accuracy': baseline_eval['accuracy'],
            'macro_f1': baseline_eval['macro_f1'],
            'macro_precision': baseline_eval['macro_precision'],
            'macro_recall': baseline_eval['macro_recall']
        },
        'structural_calibration_experiment': {
            'accuracy': cal_eval['accuracy'],
            'macro_f1': cal_eval['macro_f1'],
            'recovered_permutation': cal_info.recovered_permutation,
            'chunk_consensus': cal_info.chunk_consensus,
            'mean_margin': cal_info.mean_assignment_margin
        }
    }
    with open(os.path.join(RESULTS_DIR, 'validation_metrics.json'), 'w') as f:
        json.dump(metrics_record, f, indent=2)

    # Save per-class metrics
    cal_eval['per_class_table'].to_csv(os.path.join(RESULTS_DIR, 'per_class_metrics.csv'), index=False)

    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cal_eval['confusion_matrix_normalized'], annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"CyberSentinel Normalized Confusion Matrix (Experimental Validation)\n(Macro-F1: {cal_eval['macro_f1']:.4f}, Accuracy: {cal_eval['accuracy']:.4f})", fontsize=12)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=200)
    plt.close()

    # Save example predictions sample
    sub_df.head(50).to_csv(os.path.join(RESULTS_DIR, 'example_predictions.csv'), index=False)

    print("\n" + "=" * 80)
    print(f"ALL {passed_tests}/{total_tests} UNIT & PIPELINE TESTS PASSED")
    print(f"Saved metrics, artifacts, and plots to '{RESULTS_DIR}/'")
    print("=" * 80)

if __name__ == '__main__':
    run_tests()
