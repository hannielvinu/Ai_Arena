"""
Manifold-Driven Feature Engineering & Multi-Model Benchmark
Evaluating 6 Feature Representations across CatBoost, LightGBM, XGBoost, ExtraTrees
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

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

RANDOM_SEED = 42
EPSILON = 1e-6

def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything()

# Load train and validation data
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('validation.csv')

RAW_FEATURES = [c for c in train_df.columns if c != 'label']
le = LabelEncoder()
y_train = le.fit_transform(train_df['label'])
y_val = le.transform(val_df['label'])
CLASS_NAMES = list(le.classes_)
N_CLASSES = len(CLASS_NAMES)

print("=" * 80)
print(f"CYBERSENTINEL MANIFOLD EXPERIMENT: 6 FEATURE REPRESENTATIONS x 4 CLASSIFIERS")
print(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)} | Classes: {N_CLASSES}")
print("=" * 80)

# -------------------------------------------------------------
# Feature Engineering Builders
# -------------------------------------------------------------

def build_behavioral_features(df, eps=EPSILON):
    res = pd.DataFrame(index=df.index)
    # 1. Flow Asymmetry
    res['spkts_minus_dpkts'] = df['spkts'] - df['dpkts']
    res['pkt_asym'] = (df['spkts'] - df['dpkts']) / (df['spkts'] + df['dpkts'] + eps)
    res['sload_minus_dload'] = df['sload'] - df['dload']
    res['load_asym'] = (df['sload'] - df['dload']) / (df['sload'] + df['dload'] + eps)
    
    # 2. State
    res['ct_state_ttl'] = df['ct_state_ttl']
    res['sttl_minus_dttl'] = df['sttl'] - df['dttl']
    res['ttl_asym'] = (df['sttl'] - df['dttl']) / (df['sttl'] + df['dttl'] + eps)
    
    # 3. Burst
    res['dur'] = df['dur']
    res['rate'] = df['rate']
    res['tot_pkts'] = df['spkts'] + df['dpkts']
    res['tot_bytes'] = df['sbytes'] + df['dbytes']
    res['pkt_rate'] = res['tot_pkts'] / (df['dur'] + eps)
    res['rate_per_pkt'] = df['rate'] / (res['tot_pkts'] + eps)
    
    res = res.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return res

def build_group_relative_features(df, group_size=10, eps=EPSILON):
    res = pd.DataFrame(index=df.index)
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

print("\n[1/3] Building Feature Representations...")

X_train_raw = train_df[RAW_FEATURES].copy()
X_val_raw = val_df[RAW_FEATURES].copy()

train_beh = build_behavioral_features(train_df)
val_beh = build_behavioral_features(val_df)

train_grp = build_group_relative_features(train_df)
val_grp = build_group_relative_features(val_df)

feature_sets = {}

# Set A: Raw 20
feature_sets['A: Raw 20'] = (
    X_train_raw.values,
    X_val_raw.values,
    RAW_FEATURES
)

# Set B: Raw + Behavioral
feat_cols_b = RAW_FEATURES + list(train_beh.columns)
feature_sets['B: Raw + Behavioral'] = (
    pd.concat([X_train_raw, train_beh], axis=1).values,
    pd.concat([X_val_raw, val_beh], axis=1).values,
    feat_cols_b
)

# Set C: Raw + Group-Relative
feat_cols_c = RAW_FEATURES + list(train_grp.columns)
feature_sets['C: Raw + Group-Relative'] = (
    pd.concat([X_train_raw, train_grp], axis=1).values,
    pd.concat([X_val_raw, val_grp], axis=1).values,
    feat_cols_c
)

# Set D: Raw + Behavioral + Group-Relative
feat_cols_d = RAW_FEATURES + list(train_beh.columns) + list(train_grp.columns)
X_tr_d = pd.concat([X_train_raw, train_beh, train_grp], axis=1)
X_va_d = pd.concat([X_val_raw, val_beh, val_grp], axis=1)
feature_sets['D: Raw + Beh + Group-Rel'] = (
    X_tr_d.values,
    X_va_d.values,
    feat_cols_d
)

# Set E: PCA 3 Latent Dimensions
scaler_e = StandardScaler()
X_tr_scaled = scaler_e.fit_transform(X_tr_d.values)
X_va_scaled = scaler_e.transform(X_va_d.values)

pca_3 = PCA(n_components=3, random_state=RANDOM_SEED)
X_tr_pca3 = pca_3.fit_transform(X_tr_scaled)
X_va_pca3 = pca_3.transform(X_va_scaled)
pca_cols = ['latent_manifold_1', 'latent_manifold_2', 'latent_manifold_3']
feature_sets['E: PCA 3 Latent Manifold'] = (
    X_tr_pca3,
    X_va_pca3,
    pca_cols
)

# Set F: Raw + Behavioral + Group-Relative + PCA 3
X_tr_f = np.hstack([X_tr_d.values, X_tr_pca3])
X_va_f = np.hstack([X_va_d.values, X_va_pca3])
feat_cols_f = feat_cols_d + pca_cols
feature_sets['F: Raw + Beh + Group-Rel + PCA-3'] = (
    X_tr_f,
    X_va_f,
    feat_cols_f
)

def get_classifiers():
    return {
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=35,
            subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_SEED,
            verbose=-1, n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_SEED,
            n_jobs=-1, eval_metric='mlogloss'
        ),
        'CatBoost': CatBoostClassifier(
            iterations=350, learning_rate=0.07, depth=6,
            random_seed=RANDOM_SEED, verbose=0, thread_count=-1
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=250, max_depth=16, min_samples_split=4,
            random_state=RANDOM_SEED, n_jobs=-1
        )
    }

print("\n[2/3] Training and Evaluating Models on Validation Set...")

benchmark_records = []
all_predictions = {}
all_probabilities = {}

for f_name, (X_tr, X_va, cols) in feature_sets.items():
    clfs = get_classifiers()
    for m_name, clf in clfs.items():
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_va)
        probs = clf.predict_proba(X_va)
        if len(preds.shape) > 1:
            preds = preds.ravel()
            
        f1 = f1_score(y_val, preds, average='macro')
        acc = accuracy_score(y_val, preds)
        
        benchmark_records.append({
            'Feature Set': f_name,
            'Model': m_name,
            'Val Macro-F1': round(f1, 4),
            'Val Accuracy': round(acc, 4),
            'Num Features': len(cols)
        })
        
        key = f"{f_name} | {m_name}"
        all_predictions[key] = preds
        all_probabilities[key] = probs

bench_df = pd.DataFrame(benchmark_records).sort_values(by='Val Macro-F1', ascending=False).reset_index(drop=True)

print("\n" + "=" * 90)
print("MASTER BENCHMARK TABLE (SORTED BY VALIDATION MACRO-F1):")
print("=" * 90)
print(bench_df.to_string(index=False))

best_row = bench_df.iloc[0]
best_key = f"{best_row['Feature Set']} | {best_row['Model']}"
best_preds = all_predictions[best_key]
best_probs = all_probabilities[best_key]

print(f"\n[WINNING CONFIGURATION]: {best_key}")
print(f"   Validation Macro-F1 : {best_row['Val Macro-F1']:.4f}")
print(f"   Validation Accuracy : {best_row['Val Accuracy']:.4f}")

print("\n" + "=" * 90)
print(f"DETAILED EVALUATION OF BEST MODEL: {best_key}")
print("=" * 90)

report_str = classification_report(y_val, best_preds, target_names=CLASS_NAMES, digits=4)
print("\n--- Per-Class Classification Report ---")
print(report_str)

cm = confusion_matrix(y_val, best_preds, normalize='true')
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"Best Model Normalized Confusion Matrix\n{best_key} (Macro-F1: {best_row['Val Macro-F1']:.4f})", fontsize=12)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('best_model_confusion_matrix.png', dpi=200)
plt.close()
print("Saved 'best_model_confusion_matrix.png'")

print("\n[3/3] Updating final submission.csv with Best Model...")
confidences = np.max(best_probs, axis=1)
predicted_labels = [CLASS_NAMES[i] for i in best_preds]
sample_ids = [f"CSHT_{i:04d}" for i in range(len(val_df))]

submission_df = pd.DataFrame({
    'sample_id': sample_ids,
    'predicted_class': predicted_labels,
    'confidence': np.round(confidences, 4)
})
submission_df.to_csv('submission.csv', index=False)
print("Updated 'submission.csv' successfully.")
print("\nFirst 10 rows of updated submission.csv:")
print(submission_df.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("EXPERIMENT EXECUTION COMPLETE")
print("=" * 80)
