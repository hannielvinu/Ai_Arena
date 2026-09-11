"""
Evaluation Module for Optional Labeled Test Sets
Calculates Accuracy, Macro-F1, Precision, Recall, Per-Class Breakdown, and Confusion Matrix.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from src.config import CLASS_NAMES

def evaluate_predictions(
    y_true: List[str],
    y_pred: List[str],
    class_names: List[str] = CLASS_NAMES
) -> Dict[str, Any]:
    """
    Computes rigorous evaluation metrics when true labels are provided.
    Strictly zero synthetic fabrication — calculated directly from inputs.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro'))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted'))
    macro_prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))

    # Per-class metrics
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    per_class_list = []
    for c in class_names:
        if c in report_dict:
            per_class_list.append({
                'Class': c,
                'Precision': round(report_dict[c]['precision'], 4),
                'Recall': round(report_dict[c]['recall'], 4),
                'F1-Score': round(report_dict[c]['f1-score'], 4),
                'Support': int(report_dict[c]['support'])
            })

    per_class_df = pd.DataFrame(per_class_list)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_norm = confusion_matrix(y_true, y_pred, labels=class_names, normalize='true')

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'per_class_table': per_class_df,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_norm,
        'classification_report_str': classification_report(y_true, y_pred, digits=4, zero_division=0)
    }
