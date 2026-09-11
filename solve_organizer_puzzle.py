"""
Comprehensive Reverse-Engineering of the Organizer's 2K Puzzle & Latent Generator
Executing Phases 1 through 7
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Load train and validation data
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('validation.csv')
features = [c for c in train_df.columns if c != 'label']
N_FEATS = len(features)

le = LabelEncoder()
y_train_all = le.fit_transform(train_df['label'])
y_val_all = le.transform(val_df['label'])
class_names = list(le.classes_)
N_CLASSES = len(class_names)

print("=" * 90)
print("PHASE 1: REPRODUCING THE ORGANIZER'S 2K PROCESS & MULTI-MODEL COMPARISONS")
print("=" * 90)

# Divide train.csv into twelve 2,000-row blocks
BLOCK_SIZE = 2000
blocks = [train_df.iloc[i*BLOCK_SIZE : (i+1)*BLOCK_SIZE].reset_index(drop=True) for i in range(12)]
y_blocks = [y_train_all[i*BLOCK_SIZE : (i+1)*BLOCK_SIZE] for i in range(12)]

# Test different model architectures on Block 1
print("\n--- Training on Block 1 (First 2,000 rows): Train Accuracy vs OOF CV vs Next Block (B2) ---")
models_to_test = {
    'DecisionTree (depth=None)': DecisionTreeClassifier(random_state=42),
    'DecisionTree (depth=8)': DecisionTreeClassifier(max_depth=8, random_state=42),
    'RandomForest (100 trees)': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'ExtraTrees (100 trees)': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'KNN (k=1)': KNeighborsClassifier(n_neighbors=1),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'LightGBM (default)': lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
    'CatBoost (depth=6)': CatBoostClassifier(iterations=300, random_seed=42, verbose=0, thread_count=-1),
    'MLP (128x64)': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
}

b1_results = []
for m_name, clf in models_to_test.items():
    X_b1 = blocks[0][features].values
    y_b1 = y_blocks[0]
    X_b2 = blocks[1][features].values
    y_b2 = y_blocks[1]
    
    clf.fit(X_b1, y_b1)
    train_preds = clf.predict(X_b1)
    b2_preds = clf.predict(X_b2)
    
    tr_acc = accuracy_score(y_b1, train_preds)
    tr_f1 = f1_score(y_b1, train_preds, average='macro')
    b2_acc = accuracy_score(y_b2, b2_preds)
    b2_f1 = f1_score(y_b2, b2_preds, average='macro')
    
    b1_results.append({
        'Model': m_name,
        'B1 Train Acc': round(tr_acc, 4),
        'B1 Train F1': round(tr_f1, 4),
        'B2 Test Acc': round(b2_acc, 4),
        'B2 Test F1': round(b2_f1, 4)
    })

print(pd.DataFrame(b1_results).to_string(index=False))

# -------------------------------------------------------------
# PHASE 2: FIND WHY 2K IS SPECIAL & SEARCHING FOR LAGS / TRANSFORMS
# -------------------------------------------------------------
print("\n" + "=" * 90)
print("PHASE 2: LAG ANALYSIS & MATHEMATICAL RELATIONS ACROSS LAGS 1 TO 2000")
print("=" * 90)

# Check auto-correlations and difference norms for various lags across 24,000 rows
lags_to_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000]
lag_records = []
for lag in lags_to_test:
    # Check average absolute correlation across 20 features for this lag
    corrs = []
    mean_diffs = []
    for f in features:
        v1 = train_df[f].iloc[:-lag].values
        v2 = train_df[f].iloc[lag:].values
        r, _ = pearsonr(v1, v2)
        corrs.append(r)
        mean_diffs.append(np.mean(np.abs(v1 - v2)))
    
    # Label match rate for this lag
    label_v1 = y_train_all[:-lag]
    label_v2 = y_train_all[lag:]
    label_match = np.mean(label_v1 == label_v2) * 100
    
    lag_records.append({
        'Lag': lag,
        'Mean Feature Autocorr (r)': round(np.mean(corrs), 4),
        'Max Feature Autocorr (r)': round(np.max(corrs), 4),
        'Label Match (%)': round(label_match, 2)
    })

print(pd.DataFrame(lag_records).to_string(index=False))

# -------------------------------------------------------------
# PHASE 3: SEARCHING FOR EXACT DETERMINISTIC CLASSIFICATION EQUATIONS
# -------------------------------------------------------------
print("\n" + "=" * 90)
print("PHASE 3 & 4: SEARCHING FOR EXACT DETERMINISTIC FORMULAS / HIDDEN VARIABLES")
print("=" * 90)

# Let's inspect the decision tree rules on Block 1 to see exact splits
dt_shallow = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_shallow.fit(blocks[0][features], y_blocks[0])
print("\nDecision Tree Rules on Block 1 (max_depth=4):")
tree_rules = export_text(dt_shallow, feature_names=features, class_names=class_names)
print(tree_rules[:1500])

# Let's test combinations of features:
# Ratios: spkts/dpkts, sload/dload, sttl/dttl, ct_state_ttl, rate*dur
X_comb_train = pd.DataFrame(index=train_df.index)
X_comb_train['spkts_dpkts_ratio'] = train_df['spkts'] / (train_df['dpkts'] + 1e-6)
X_comb_train['spkts_dpkts_diff'] = train_df['spkts'] - train_df['dpkts']
X_comb_train['sload_dload_ratio'] = train_df['sload'] / (train_df['dload'] + 1e-6)
X_comb_train['sload_dload_diff'] = train_df['sload'] - train_df['dload']
X_comb_train['sttl_dttl_diff'] = train_df['sttl'] - train_df['dttl']
X_comb_train['ct_state_ttl'] = train_df['ct_state_ttl']

# Group by label in train and print exact mean/std of these key combinations
print("\nMean of Engineered Combinations per Class in Train:")
comb_grouped = X_comb_train.groupby(train_df['label']).mean()
comb_stds = X_comb_train.groupby(train_df['label']).std()
print(comb_grouped.round(3).to_string())

# -------------------------------------------------------------
# PHASE 5 & 6: THE CHRONOLOGICAL 2K EXPERIMENT
# Train -> Predict Next Block -> Add Next Block -> Repeat
# -------------------------------------------------------------
print("\n" + "=" * 90)
print("PHASE 6: THE CHRONOLOGICAL 2K RECURSIVE EXPERIMENT")
print("=" * 90)

# Test 1: LightGBM Cumulative (B1 -> B2, B1+B2 -> B3, ... B1..11 -> B12)
# Test 2: CatBoost Cumulative
# Test 3: ExtraTrees Cumulative
# Test 4: Online Gradient Descent / Incremental Warm Start

chronological_records = []
for step in range(1, 12):
    train_subset_df = pd.concat(blocks[:step], axis=0).reset_index(drop=True)
    test_subset_df = blocks[step]
    
    X_tr = train_subset_df[features].values
    y_tr = y_train_all[:step*BLOCK_SIZE]
    X_te = test_subset_df[features].values
    y_te = y_blocks[step]
    
    # Train LightGBM
    clf_lgb = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.06, num_leaves=35, random_state=42, verbose=-1, n_jobs=-1)
    clf_lgb.fit(X_tr, y_tr)
    preds_lgb = clf_lgb.predict(X_te)
    f1_lgb = f1_score(y_te, preds_lgb, average='macro')
    acc_lgb = accuracy_score(y_te, preds_lgb)
    
    # Train ExtraTrees
    clf_et = ExtraTreesClassifier(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1)
    clf_et.fit(X_tr, y_tr)
    preds_et = clf_et.predict(X_te)
    f1_et = f1_score(y_te, preds_et, average='macro')
    acc_et = accuracy_score(y_te, preds_et)
    
    train_size_desc = f"B1..B{step} ({step*2}k)" if step > 1 else "B1 (2k)"
    chronological_records.append({
        'Train Blocks': train_size_desc,
        'Test Block': f"B{step+1} (2k)",
        'LGBM Macro-F1': round(f1_lgb, 4),
        'LGBM Accuracy': round(acc_lgb, 4),
        'ExtraTrees Macro-F1': round(f1_et, 4),
        'ExtraTrees Accuracy': round(acc_et, 4)
    })

print(pd.DataFrame(chronological_records).to_string(index=False))

# -------------------------------------------------------------
# PHASE 7: TESTING PREDICTION ON VALIDATION.CSV
# -------------------------------------------------------------
print("\n" + "=" * 90)
print("PHASE 7: VALIDATION EVALUATION & TESTING HYPOTHESES")
print("=" * 90)

# Evaluate on Validation using full 24k trained ExtraTrees & CatBoost
clf_final_et = ExtraTreesClassifier(n_estimators=300, max_depth=18, min_samples_split=4, random_state=42, n_jobs=-1)
clf_final_et.fit(train_df[features], y_train_all)
val_preds_et = clf_final_et.predict(val_df[features])
val_f1_et = f1_score(y_val_all, val_preds_et, average='macro')
val_acc_et = accuracy_score(y_val_all, val_preds_et)

clf_final_cb = CatBoostClassifier(iterations=400, learning_rate=0.06, depth=6, random_seed=42, verbose=0, thread_count=-1)
clf_final_cb.fit(train_df[features], y_train_all)
val_preds_cb = clf_final_cb.predict(val_df[features]).ravel()
val_f1_cb = f1_score(y_val_all, val_preds_cb, average='macro')
val_acc_cb = accuracy_score(y_val_all, val_preds_cb)

print(f"Full Train -> Validation [ExtraTrees]: Macro-F1 = {val_f1_et:.4f} | Accuracy = {val_acc_et:.4f}")
print(f"Full Train -> Validation [CatBoost]  : Macro-F1 = {val_f1_cb:.4f} | Accuracy = {val_acc_cb:.4f}")

# Check why accuracy is ~30% instead of ~89%:
# Examine the Bayes Optimal Error / Feature Overlap
print("\n--- Feature Overlap & Class Separability Analysis ---")
# Check how much overlap exists in feature space (e.g. nearest neighbor distance between classes)
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=2)
nn.fit(train_df[features].values)
distances, indices = nn.kneighbors(train_df[features].values)
nearest_neighbor_same_class = (y_train_all == y_train_all[indices[:, 1]]).mean()
print(f"Nearest Neighbor (k=1) Same Class Accuracy in 20-D space: {nearest_neighbor_same_class*100:.2f}%")
print(f"(A random guess in 10 classes is 10.0%, 1-NN reaches {nearest_neighbor_same_class*100:.2f}%)")
