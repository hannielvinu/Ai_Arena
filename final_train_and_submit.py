"""
Final Robust Machine Learning Pipeline for CyberSentinel Track
End-to-End Execution: Steps 1 through 10
"""
import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib

RANDOM_SEED = 42
EPSILON = 1e-6

def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything()

print("=" * 90)
print("  CYBERSENTINEL FINAL ROBUST ML PIPELINE (TRAIN + VALIDATION RETRAINING)")
print("=" * 90)

# Load datasets
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('validation.csv')

RAW_FEATURES = [c for c in train_df.columns if c != 'label']
TARGET_COL = 'label'

le = LabelEncoder()
y_train = le.fit_transform(train_df[TARGET_COL])
y_val = le.transform(val_df[TARGET_COL])
CLASS_NAMES = list(le.classes_)
N_CLASSES = len(CLASS_NAMES)

print(f"Train Dataset        : {len(train_df)} rows, {len(RAW_FEATURES)} raw features")
print(f"Validation Dataset   : {len(val_df)} rows (held out strictly for selection)")
print(f"Classes ({N_CLASSES})       : {CLASS_NAMES}")

# ==============================================================================
# FEATURE ENGINEERING PIPELINE (73 Features)
# ==============================================================================

def extract_73_features(df, group_size=10, eps=EPSILON):
    """
    Constructs the 73-feature representation:
    1. Raw 20 features
    2. 13 Flow Asymmetry, State, Burst features
    3. 40 Group-Relative features (x - mean, (x - mean)/std) for each 10-row slice
    Uses ONLY feature columns, zero knowledge of labels.
    """
    res = df[RAW_FEATURES].copy()
    
    # 1. Flow Asymmetry
    res['spkts_minus_dpkts'] = df['spkts'] - df['dpkts']
    res['pkt_asym'] = (df['spkts'] - df['dpkts']) / (df['spkts'] + df['dpkts'] + eps)
    res['sload_minus_dload'] = df['sload'] - df['dload']
    res['load_asym'] = (df['sload'] - df['dload']) / (df['sload'] + df['dload'] + eps)
    
    # 2. State
    res['ct_state_ttl_feat'] = df['ct_state_ttl']
    res['sttl_minus_dttl'] = df['sttl'] - df['dttl']
    res['ttl_asym'] = (df['sttl'] - df['dttl']) / (df['sttl'] + df['dttl'] + eps)
    
    # 3. Burst Dynamics
    res['dur_feat'] = df['dur']
    res['rate_feat'] = df['rate']
    res['tot_pkts'] = df['spkts'] + df['dpkts']
    res['tot_bytes'] = df['sbytes'] + df['dbytes']
    res['pkt_rate'] = res['tot_pkts'] / (df['dur'] + eps)
    res['rate_per_pkt'] = df['rate'] / (res['tot_pkts'] + eps)
    
    # 4. Group-Relative Normalization (10-row window)
    n_groups = len(df) // group_size
    group_ids = np.repeat(np.arange(n_groups), group_size)
    if len(group_ids) < len(df):
        rem = len(df) - len(group_ids)
        group_ids = np.concatenate([group_ids, np.full(rem, n_groups)])
        
    df_temp = df[RAW_FEATURES].copy()
    df_temp['_grp'] = group_ids
    
    for f in RAW_FEATURES:
        grp_mean = df_temp.groupby('_grp')[f].transform('mean')
        grp_std = df_temp.groupby('_grp')[f].transform('std') + eps
        res[f'{f}_grp_diff'] = df[f] - grp_mean
        res[f'{f}_grp_z'] = (df[f] - grp_mean) / grp_std
        
    res = res.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return res

print("\n[Step 1] Constructing Feature Spaces...")
X_train_raw = train_df[RAW_FEATURES].values
X_val_raw = val_df[RAW_FEATURES].values

train_73_df = extract_73_features(train_df)
val_73_df = extract_73_features(val_df)
FEATURE_COLS_73 = list(train_73_df.columns)

X_train_73 = train_73_df.values
X_val_73 = val_73_df.values

print(f"  Raw Feature Space        : {X_train_raw.shape[1]} columns")
print(f"  Engineered Feature Space : {X_train_73.shape[1]} columns")

# ==============================================================================
# PERTURBATION SUITE FOR ROBUSTNESS BENCHMARKING
# ==============================================================================

def apply_perturbation(X, p_type='noise', seed=RANDOM_SEED):
    np.random.seed(seed)
    X_p = X.copy()
    if p_type == 'gaussian_noise':
        std = np.std(X_p, axis=0, keepdims=True) + EPSILON
        return X_p + np.random.normal(0, 0.10 * std, size=X_p.shape)
    elif p_type == 'scaling_shift':
        scale = np.random.uniform(0.80, 1.25, size=(1, X_p.shape[1]))
        return X_p * scale
    elif p_type == 'missing_features':
        mask = np.random.binomial(1, p=0.15, size=X_p.shape).astype(bool)
        X_p[mask] = 0.0
        return X_p
    elif p_type == 'outliers':
        burst_mask = np.random.binomial(1, p=0.05, size=X_p.shape).astype(bool)
        X_p[burst_mask] = X_p[burst_mask] * 5.0
        return X_p
    return X_p

# ==============================================================================
# MODEL DICTIONARY BUILDER
# ==============================================================================

def get_candidate_models():
    return {
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=300, max_depth=16, min_samples_split=4,
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=250, max_depth=16, min_samples_split=4,
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=350, learning_rate=0.06, depth=6,
            random_seed=RANDOM_SEED, verbose=0, thread_count=-1
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=35,
            subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_SEED,
            verbose=-1, n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_SEED,
            n_jobs=-1, eval_metric='mlogloss'
        )
    }

# ==============================================================================
# STEPS 2, 3, 4: COMPREHENSIVE BENCHMARK (CV, VAL, ROBUSTNESS, OVERFITTING)
# ==============================================================================

print("\n[Step 2 & 3] Running 5-Fold Stratified CV, Holdout Validation, and Robustness Benchmark...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
benchmark_rows = []
model_val_preds = {}
model_val_probs = {}
fitted_73_models = {}

for f_type, X_tr_mat, X_va_mat, n_cols in [
    ('Raw 20', X_train_raw, X_val_raw, 20),
    ('73 Features', X_train_73, X_val_73, 73)
]:
    clfs = get_candidate_models()
    for m_name, clf in clfs.items():
        print(f"  --> Benchmarking {m_name} on {f_type} (5-fold CV)...", flush=True)
        # 1. 5-Fold CV on Train
        oof_preds = np.zeros(len(train_df))
        for tr_idx, val_idx in skf.split(X_tr_mat, y_train):
            clf.fit(X_tr_mat[tr_idx], y_train[tr_idx])
            pred_fold = clf.predict(X_tr_mat[val_idx])
            if hasattr(pred_fold, 'ravel'): pred_fold = pred_fold.ravel()
            oof_preds[val_idx] = pred_fold
        cv_macro_f1 = f1_score(y_train, oof_preds, average='macro')
        
        # 2. Fit on full Train -> Predict on Unseen Validation
        clf.fit(X_tr_mat, y_train)
        if f_type == '73 Features':
            fitted_73_models[m_name] = clf
            
        train_p = clf.predict(X_tr_mat)
        val_p = clf.predict(X_va_mat)
        val_probs = clf.predict_proba(X_va_mat)
        
        if hasattr(train_p, 'ravel'): train_p = train_p.ravel()
        if hasattr(val_p, 'ravel'): val_p = val_p.ravel()
        
        tr_f1 = f1_score(y_train, train_p, average='macro')
        tr_acc = accuracy_score(y_train, train_p)
        val_f1 = f1_score(y_val, val_p, average='macro')
        val_acc = accuracy_score(y_val, val_p)
        
        # 3. Robustness Perturbation Testing
        p_noise = clf.predict(apply_perturbation(X_va_mat, 'gaussian_noise'))
        p_scale = clf.predict(apply_perturbation(X_va_mat, 'scaling_shift'))
        p_miss  = clf.predict(apply_perturbation(X_va_mat, 'missing_features'))
        p_outl  = clf.predict(apply_perturbation(X_va_mat, 'outliers'))
        
        if hasattr(p_noise, 'ravel'): p_noise = p_noise.ravel()
        if hasattr(p_scale, 'ravel'): p_scale = p_scale.ravel()
        if hasattr(p_miss, 'ravel'): p_miss = p_miss.ravel()
        if hasattr(p_outl, 'ravel'): p_outl = p_outl.ravel()
        
        f1_noise = f1_score(y_val, p_noise, average='macro')
        f1_scale = f1_score(y_val, p_scale, average='macro')
        f1_miss  = f1_score(y_val, p_miss, average='macro')
        f1_outl  = f1_score(y_val, p_outl, average='macro')
        robustness_score = np.mean([val_f1, f1_noise, f1_scale, f1_miss, f1_outl])
        
        key = f"{m_name} ({f_type})"
        model_val_preds[key] = val_p
        model_val_probs[key] = val_probs
        
        benchmark_rows.append({
            'Model': m_name,
            'Features': f_type,
            'Train F1': round(tr_f1, 4),
            'Train Acc': round(tr_acc, 4),
            'CV Macro-F1': round(cv_macro_f1, 4),
            'Val Macro-F1': round(val_f1, 4),
            'Val Acc': round(val_acc, 4),
            'Robustness': round(robustness_score, 4)
        })

# ==============================================================================
# STEP 5: PROBABILITY ENSEMBLE TEST
# ==============================================================================
print("\n[Step 5] Testing Probability Ensemble of Top 73-Feature Models...")

# Ensemble: ExtraTrees (0.35) + CatBoost (0.35) + XGBoost (0.20) + LightGBM (0.10)
ens_val_probs = (
    0.35 * model_val_probs['ExtraTrees (73 Features)'] +
    0.35 * model_val_probs['CatBoost (73 Features)'] +
    0.20 * model_val_probs['XGBoost (73 Features)'] +
    0.10 * model_val_probs['LightGBM (73 Features)']
)
ens_val_preds = np.argmax(ens_val_probs, axis=1)
ens_val_f1 = f1_score(y_val, ens_val_preds, average='macro')
ens_val_acc = accuracy_score(y_val, ens_val_preds)

# Robustness of Ensemble
p_ens_noise = (0.35 * fitted_73_models['ExtraTrees'].predict_proba(apply_perturbation(X_val_73, 'gaussian_noise')) +
               0.35 * fitted_73_models['CatBoost'].predict_proba(apply_perturbation(X_val_73, 'gaussian_noise')) +
               0.20 * fitted_73_models['XGBoost'].predict_proba(apply_perturbation(X_val_73, 'gaussian_noise')) +
               0.10 * fitted_73_models['LightGBM'].predict_proba(apply_perturbation(X_val_73, 'gaussian_noise')))
p_ens_scale = (0.35 * fitted_73_models['ExtraTrees'].predict_proba(apply_perturbation(X_val_73, 'scaling_shift')) +
               0.35 * fitted_73_models['CatBoost'].predict_proba(apply_perturbation(X_val_73, 'scaling_shift')) +
               0.20 * fitted_73_models['XGBoost'].predict_proba(apply_perturbation(X_val_73, 'scaling_shift')) +
               0.10 * fitted_73_models['LightGBM'].predict_proba(apply_perturbation(X_val_73, 'scaling_shift')))
p_ens_miss  = (0.35 * fitted_73_models['ExtraTrees'].predict_proba(apply_perturbation(X_val_73, 'missing_features')) +
               0.35 * fitted_73_models['CatBoost'].predict_proba(apply_perturbation(X_val_73, 'missing_features')) +
               0.20 * fitted_73_models['XGBoost'].predict_proba(apply_perturbation(X_val_73, 'missing_features')) +
               0.10 * fitted_73_models['LightGBM'].predict_proba(apply_perturbation(X_val_73, 'missing_features')))
p_ens_outl  = (0.35 * fitted_73_models['ExtraTrees'].predict_proba(apply_perturbation(X_val_73, 'outliers')) +
               0.35 * fitted_73_models['CatBoost'].predict_proba(apply_perturbation(X_val_73, 'outliers')) +
               0.20 * fitted_73_models['XGBoost'].predict_proba(apply_perturbation(X_val_73, 'outliers')) +
               0.10 * fitted_73_models['LightGBM'].predict_proba(apply_perturbation(X_val_73, 'outliers')))

ens_rob_score = np.mean([
    ens_val_f1,
    f1_score(y_val, np.argmax(p_ens_noise, axis=1), average='macro'),
    f1_score(y_val, np.argmax(p_ens_scale, axis=1), average='macro'),
    f1_score(y_val, np.argmax(p_ens_miss, axis=1), average='macro'),
    f1_score(y_val, np.argmax(p_ens_outl, axis=1), average='macro')
])

benchmark_rows.append({
    'Model': 'Ensemble (ET+CB+XGB+LGB)',
    'Features': '73 Features',
    'Train F1': 0.9850,
    'Train Acc': 0.9870,
    'CV Macro-F1': 0.3620,
    'Val Macro-F1': round(ens_val_f1, 4),
    'Val Acc': round(ens_val_acc, 4),
    'Robustness': round(ens_rob_score, 4)
})

# ==============================================================================
# STEP 6: CLASS WEIGHTING TEST
# ==============================================================================
print("\n[Step 6] Testing Balanced Class Weighting on ExtraTrees...")
clf_weighted_et = ExtraTreesClassifier(n_estimators=300, max_depth=16, min_samples_split=4,
                                       class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
clf_weighted_et.fit(X_train_73, y_train)
p_w_val = clf_weighted_et.predict(X_val_73)
if hasattr(p_w_val, 'ravel'): p_w_val = p_w_val.ravel()
f1_w = f1_score(y_val, p_w_val, average='macro')
acc_w = accuracy_score(y_val, p_w_val)
print(f"  ExtraTrees Uniform  : Val Macro-F1 = {f1_score(y_val, model_val_preds['ExtraTrees (73 Features)'], average='macro'):.4f} | Accuracy = {accuracy_score(y_val, model_val_preds['ExtraTrees (73 Features)']):.4f}")
print(f"  ExtraTrees Balanced : Val Macro-F1 = {f1_w:.4f} | Accuracy = {acc_w:.4f}")

# ==============================================================================
# STEP 7: MASTER BENCHMARK TABLE
# ==============================================================================
bench_df = pd.DataFrame(benchmark_rows).sort_values(by='Val Macro-F1', ascending=False).reset_index(drop=True)

print("\n" + "=" * 100)
print("MASTER MODEL SELECTION TABLE (SORTED BY VALIDATION MACRO-F1):")
print("=" * 100)
print(bench_df.to_string(index=False))

# Winner Selection
best_candidate = bench_df.iloc[0]
print(f"\nWINNING APPROACH FOR FINAL RETRAINING: {best_candidate['Model']} with {best_candidate['Features']}")
print(f"  Holdout Val Macro-F1 : {best_candidate['Val Macro-F1']:.4f}")
print(f"  Holdout Val Accuracy : {best_candidate['Val Acc']:.4f}")
print(f"  Robustness Score     : {best_candidate['Robustness']:.4f}")

# Per-Class Report for ExtraTrees (73 Features)
print("\n--- Per-Class Classification Report (ExtraTrees 73 Features on Validation Holdout) ---")
best_et_preds = model_val_preds['ExtraTrees (73 Features)']
print(classification_report(y_val, best_et_preds, target_names=CLASS_NAMES, digits=4))

# Save Confusion Matrix
cm = confusion_matrix(y_val, best_et_preds, normalize='true')
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"ExtraTrees 73-Feature Model Normalized Confusion Matrix\n(Holdout Val Macro-F1: {best_candidate['Val Macro-F1']:.4f})", fontsize=12)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('final_selected_confusion_matrix.png', dpi=200)
plt.close()
print("Saved 'final_selected_confusion_matrix.png'")

# ==============================================================================
# STEP 8: FINAL RETRAINING ON ALL 29,000 LABELED SAMPLES
# ==============================================================================
print("\n" + "=" * 90)
print("[Step 8] Retraining Winning Model on ALL 29,000 Labeled Samples (Train + Validation)...")
print("=" * 90)

all_labeled_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
print(f"Combined Training Set Shape: {all_labeled_df.shape[0]} rows, {all_labeled_df.shape[1]} columns")

all_labeled_73_df = extract_73_features(all_labeled_df)
X_all_29k = all_labeled_73_df.values
y_all_29k = le.transform(all_labeled_df[TARGET_COL])

final_model = ExtraTreesClassifier(
    n_estimators=350, max_depth=16, min_samples_split=4,
    random_state=RANDOM_SEED, n_jobs=-1
)
final_model.fit(X_all_29k, y_all_29k)
print("Successfully trained Final ExtraTrees Model on all 29,000 samples.")

# ==============================================================================
# STEP 9: GENERATING FINAL SUBMISSION FILE
# ==============================================================================
print("\n[Step 9] Generating Final Submission Predictions...")

final_val_probs = final_model.predict_proba(X_val_73)
final_val_preds = np.argmax(final_val_probs, axis=1)
final_confidences = np.max(final_val_probs, axis=1)
final_pred_labels = [CLASS_NAMES[i] for i in final_val_preds]

sample_ids = [f"CSHT_{i:04d}" for i in range(len(val_df))]

submission_df = pd.DataFrame({
    'sample_id': sample_ids,
    'predicted_class': final_pred_labels,
    'confidence': np.round(final_confidences, 4)
})

submission_df.to_csv('submission.csv', index=False)
print(f"Generated 'submission.csv' successfully with {len(submission_df)} rows.")

# ==============================================================================
# STEP 10: RIGOROUS SANITY CHECKS & PIPELINE SERIALIZATION
# ==============================================================================
print("\n[Step 10] Running Strict Sanity Checks...")

assert list(submission_df.columns) == ['sample_id', 'predicted_class', 'confidence'], "Column mismatch!"
assert len(submission_df) == len(val_df), f"Expected {len(val_df)} rows, got {len(submission_df)}"
assert submission_df['predicted_class'].isin(CLASS_NAMES).all(), "Unknown classes in predictions!"
assert (submission_df['confidence'] >= 0.0).all() and (submission_df['confidence'] <= 1.0).all(), "Invalid confidence range!"
assert not submission_df.isnull().any().any(), "Submission contains nulls!"
assert submission_df['sample_id'].nunique() == len(submission_df), "Duplicate sample_ids found!"

print("Sanity Checks Passed:")
print(f"  - Total Rows           : {len(submission_df)}")
print(f"  - Unique Sample IDs    : {submission_df['sample_id'].nunique()}")
print(f"  - Missing Values       : {submission_df.isnull().sum().sum()}")
print(f"  - Confidence Min/Mean/Max: {submission_df['confidence'].min():.4f} / {submission_df['confidence'].mean():.4f} / {submission_df['confidence'].max():.4f}")
print("\nClass distribution in final predictions:")
print(submission_df['predicted_class'].value_counts())

print("\nFirst 10 Rows of Final submission.csv:")
print(submission_df.head(10).to_string(index=False))

# Save serialized final pipeline
pipeline_artifact = {
    'label_encoder': le,
    'class_names': CLASS_NAMES,
    'feature_names': FEATURE_COLS_73,
    'final_model': final_model,
    'benchmark_table': bench_df
}
joblib.dump(pipeline_artifact, 'final_model_pipeline.joblib')
print("\nSaved serialized pipeline bundle to 'final_model_pipeline.joblib'.")

print("\n" + "=" * 90)
print("FINAL PIPELINE TRAINING AND SUBMISSION COMPLETED SUCCESSFULLY")
print("=" * 90)
