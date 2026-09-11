"""
Investigate 10-Row Group Generative Structure in train.csv
Reverse-Engineering Shared Latent Variables, Cross-Class Transformations, and Within-Group Properties
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, f1_score, accuracy_score
import lightgbm as lgb

# Load train.csv and validation.csv
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('validation.csv')
features = [c for c in train_df.columns if c != 'label']
N_FEATS = len(features)
N_GROUPS = len(train_df) // 10

print("=" * 80)
print(f"ANALYZING 2,400 GROUPS (10 ROWS PER GROUP) IN TRAIN.CSV")
print("=" * 80)

tensor_train = np.zeros((N_GROUPS, 10, N_FEATS))
labels_matrix = []

for g in range(N_GROUPS):
    grp_df = train_df.iloc[g*10 : (g+1)*10]
    tensor_train[g] = grp_df[features].values
    labels_matrix.append(grp_df['label'].tolist())

class_order = labels_matrix[0]
print(f"Class order in each group: {class_order}")

# -------------------------------------------------------------
# 1. Within-Group Variance vs Between-Group Variance (ANOVA / Intra-class Correlation)
# -------------------------------------------------------------
print("\n" + "-" * 80)
print("1. INTRA-GROUP CORRELATION & VARIANCE DECOMPOSITION")
print("-" * 80)

group_means = np.mean(tensor_train, axis=1)
group_stds = np.std(tensor_train, axis=1)
total_stds = np.std(train_df[features].values, axis=0)

icc_list = []
for f_idx, f_name in enumerate(features):
    var_between = np.var(group_means[:, f_idx])
    var_within = np.mean(group_stds[:, f_idx]**2)
    var_total = np.var(train_df[f_name].values)
    icc = var_between / (var_total + 1e-10)
    icc_list.append({
        'Feature': f_name,
        'Total_Std': round(total_stds[f_idx], 3),
        'Mean_Within_Group_Std': round(np.mean(group_stds[:, f_idx]), 3),
        'Between_Group_Std': round(np.sqrt(var_between), 3),
        'Group_Variance_Ratio (ICC)': round(icc, 4)
    })

icc_df = pd.DataFrame(icc_list)
print(icc_df.to_string(index=False))

# -------------------------------------------------------------
# 2. Cross-Class Correlations & Consistent Offsets/Multipliers
# -------------------------------------------------------------
print("\n" + "-" * 80)
print("2. SEARCHING FOR DETERMINISTIC FEATURE TRANSFORMATIONS: Class A -> Class B")
print("-" * 80)

strong_links = []
for f_idx, f_name in enumerate(features):
    for p1 in range(10):
        for p2 in range(p1+1, 10):
            v1 = tensor_train[:, p1, f_idx]
            v2 = tensor_train[:, p2, f_idx]
            r, p = pearsonr(v1, v2)
            if abs(r) > 0.15:
                strong_links.append({
                    'Feature': f_name,
                    'Class_1': class_order[p1],
                    'Class_2': class_order[p2],
                    'Pearson_r': round(r, 4),
                    'p_val': p
                })

print(f"Number of cross-class correlations with |r| > 0.15: {len(strong_links)}")
max_r = 0.0
max_pair = None
for f_idx, f_name in enumerate(features):
    for p1 in range(10):
        for p2 in range(p1+1, 10):
            r, _ = pearsonr(tensor_train[:, p1, f_idx], tensor_train[:, p2, f_idx])
            if abs(r) > max_r:
                max_r = abs(r)
                max_pair = (f_name, class_order[p1], class_order[p2], r)
print(f"Highest cross-class correlation across all features: {max_pair[0]} between {max_pair[1]} and {max_pair[2]} with r = {max_pair[3]:+.4f}")

# -------------------------------------------------------------
# 3. Position-Specific Feature Signatures (The 10-Class Template)
# -------------------------------------------------------------
print("\n" + "-" * 80)
print("3. EXACT FEATURE TEMPLATES PER CLASS POSITION (Mean across 2,400 groups)")
print("-" * 80)
pos_means = np.mean(tensor_train, axis=0)
template_df = pd.DataFrame(pos_means, index=class_order, columns=features)
print(template_df.round(3).to_string())

# -------------------------------------------------------------
# 4. Group Index vs Feature Values (Is Group Number a Temporal / Latent Index?)
# -------------------------------------------------------------
print("\n" + "-" * 80)
print("4. CORRELATION WITH GROUP NUMBER (Time / Index Progression)")
print("-" * 80)
group_indices = np.arange(N_GROUPS)
time_corrs = []
for f_idx, f_name in enumerate(features):
    r_mean, _ = pearsonr(group_indices, group_means[:, f_idx])
    r_raw, _ = pearsonr(np.arange(len(train_df)), train_df[f_name].values)
    time_corrs.append({
        'Feature': f_name,
        'Group_Mean_vs_Index_r': round(r_mean, 4),
        'Raw_Feature_vs_Row_Index_r': round(r_raw, 4)
    })
print(pd.DataFrame(time_corrs).to_string(index=False))

# -------------------------------------------------------------
# 5. STRICT PREDICTIVE EXPERIMENT:
# Use 9 samples in a group to predict the 10th (missing) sample
# -------------------------------------------------------------
print("\n" + "-" * 80)
print("5. STRICT PREDICTIVE EXPERIMENT: Predicting Missing 10th Sample from 9 Neighbors")
print("-" * 80)

skf_groups = np.arange(N_GROUPS) % 5
r2_scores_per_target_class = {}
for target_p in range(10):
    target_class = class_order[target_p]
    other_indices = [p for p in range(10) if p != target_p]
    X_grp = tensor_train[:, other_indices, :].reshape(N_GROUPS, -1)
    y_grp = tensor_train[:, target_p, :]
    
    oof_pred_y = np.zeros_like(y_grp)
    for fold in range(5):
        tr_mask = (skf_groups != fold)
        val_mask = (skf_groups == fold)
        
        reg = Ridge(alpha=100.0)
        reg.fit(X_grp[tr_mask], y_grp[tr_mask])
        oof_pred_y[val_mask] = reg.predict(X_grp[val_mask])
        
    r2_all_feats = [r2_score(y_grp[:, fi], oof_pred_y[:, fi]) for fi in range(N_FEATS)]
    r2_scores_per_target_class[target_class] = np.mean(r2_all_feats)

print("Mean R2 score predicting 10th sample's features from the other 9 in the group:")
for c_name, avg_r2 in r2_scores_per_target_class.items():
    print(f"  Target Class '{c_name:10s}': Mean R2 = {avg_r2:+.4f}")

# -------------------------------------------------------------
# 6. WITHIN-GROUP NORMALIZATION & RELATIVE FEATURE ENGINERING
# -------------------------------------------------------------
print("\n" + "-" * 80)
print("6. WITHIN-GROUP NORMALIZATION & RELATIVE FEATURE ENGINERING")
print("-" * 80)

def extract_group_features(df):
    res = df.copy()
    n_grps = len(df) // 10
    grp_ids = np.repeat(np.arange(n_grps), 10)
    res['group_id'] = grp_ids
    
    for f in features:
        grp_mean = res.groupby('group_id')[f].transform('mean')
        grp_std = res.groupby('group_id')[f].transform('std') + 1e-6
        res[f'{f}_grp_diff'] = res[f] - grp_mean
        res[f'{f}_grp_z'] = (res[f] - grp_mean) / grp_std
        res[f'{f}_grp_mean'] = grp_mean
        
    return res

train_grp_df = extract_group_features(train_df)
val_grp_df = extract_group_features(val_df)

grp_feat_cols = [c for c in train_grp_df.columns if c not in ['label', 'label_encoded', 'group_id']]

le = LabelEncoder()
y_train_enc = le.fit_transform(train_df['label'])
y_val_enc = le.transform(val_df['label'])

clf_standard = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, random_state=42, verbose=-1, n_jobs=-1)
clf_standard.fit(train_df[features], y_train_enc)
p_std = clf_standard.predict(val_df[features])
f1_std = f1_score(y_val_enc, p_std, average='macro')
acc_std = accuracy_score(y_val_enc, p_std)

clf_grp = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, random_state=42, verbose=-1, n_jobs=-1)
clf_grp.fit(train_grp_df[grp_feat_cols], y_train_enc)
p_grp = clf_grp.predict(val_grp_df[grp_feat_cols])
f1_grp = f1_score(y_val_enc, p_grp, average='macro')
acc_grp = accuracy_score(y_val_enc, p_grp)

print(f"Standard Model (Raw Features)         : Val Macro-F1 = {f1_std:.4f} | Accuracy = {acc_std:.4f}")
print(f"Group-Relative Model (Within-Group Diff): Val Macro-F1 = {f1_grp:.4f} | Accuracy = {acc_grp:.4f}")
print(f"Delta from Group-Relative Encoding    : {f1_grp - f1_std:+.4f}")

# -------------------------------------------------------------
# 7. Cluster Structure Within Groups: PCA of Group Profiles
# -------------------------------------------------------------
print("\n" + "-" * 80)
print("7. LATENT GENERATIVE MANIFOLD OF THE 10-CLASS TEMPLATE")
print("-" * 80)
pca_pos = PCA(n_components=3)
pos_pca_emb = pca_pos.fit_transform(pos_means)
print(f"Variance explained by top 3 PCA components across 10 classes: {pca_pos.explained_variance_ratio_ * 100}")
for c_idx, c_name in enumerate(class_order):
    print(f"  Class {c_name:10s}: PC1={pos_pca_emb[c_idx,0]:+6.2f}, PC2={pos_pca_emb[c_idx,1]:+6.2f}, PC3={pos_pca_emb[c_idx,2]:+6.2f}")
