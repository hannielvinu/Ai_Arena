"""
Investigate Row Order Structure & 12 Consecutive 2,000-Row Blocks in train.csv
Reverse-Engineering the Organizer's Hint
"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from scipy.stats import ks_2samp

# 1. Load Data
df = pd.read_csv('train.csv')
features = [c for c in df.columns if c != 'label']
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
class_names = list(le.classes_)
n_classes = len(class_names)

print(f"Total Rows: {len(df)}, Features: {len(features)}, Classes: {n_classes}")
print(f"Classes: {class_names}")

# Split into 12 blocks of 2000 rows
BLOCK_SIZE = 2000
N_BLOCKS = len(df) // BLOCK_SIZE
blocks = [df.iloc[i*BLOCK_SIZE : (i+1)*BLOCK_SIZE].copy().reset_index(drop=True) for i in range(N_BLOCKS)]

print("\n" + "="*90)
print("1. CLASS DISTRIBUTION ACROSS 12 BLOCKS")
print("="*90)
class_dist_df = pd.DataFrame()
for b_idx, b_df in enumerate(blocks, 1):
    counts = b_df['label'].value_counts()
    class_dist_df[f'Block_{b_idx}'] = counts

class_dist_df = class_dist_df.fillna(0).astype(int)
print(class_dist_df.to_string())

# Check if class counts are identical or varying
print("\nClass distribution variance across blocks:")
print(class_dist_df.std(axis=1))

print("\n" + "="*90)
print("2. FEATURE MEAN DRIFT ACROSS 12 BLOCKS")
print("="*90)
means_df = pd.DataFrame()
for b_idx, b_df in enumerate(blocks, 1):
    means_df[f'B{b_idx}'] = b_df[features].mean()

print("Feature Means across Blocks 1 to 12:")
print(means_df.round(3).to_string())

print("\n" + "="*90)
print("3. FEATURE STD / MIN / MAX DRIFT ACROSS 12 BLOCKS")
print("="*90)
stds_df = pd.DataFrame()
for b_idx, b_df in enumerate(blocks, 1):
    stds_df[f'B{b_idx}'] = b_df[features].std()
print("Feature Standard Deviations across Blocks 1 to 12:")
print(stds_df.round(3).to_string())

print("\n" + "="*90)
print("4. DRIFT BETWEEN NEIGHBORING BLOCKS (Kolmogorov-Smirnov Test)")
print("="*90)
ks_summary = []
for i in range(N_BLOCKS - 1):
    b1 = blocks[i]
    b2 = blocks[i+1]
    sig_shifts = 0
    max_stat = 0.0
    worst_feat = ""
    for f in features:
        stat, p = ks_2samp(b1[f], b2[f])
        if p < 0.01:
            sig_shifts += 1
        if stat > max_stat:
            max_stat = stat
            worst_feat = f
    ks_summary.append({
        'Transition': f"B{i+1} -> B{i+2}",
        'Sig_Shifts (p<0.01)': f"{sig_shifts}/{len(features)}",
        'Max_KS_Stat': round(max_stat, 4),
        'Worst_Feature': worst_feat
    })
print(pd.DataFrame(ks_summary).to_string(index=False))

print("\n" + "="*90)
print("5. MODEL EXPERIMENTS: SAME-BLOCK CONTROL VS SEQUENTIAL PREDICTION")
print("="*90)

# Control: Within-Block 5-fold CV
control_results = []
for b_idx, b_df in enumerate(blocks, 1):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(b_df))
    X = b_df[features].values
    y = b_df['label_encoded'].values
    for tr, val in skf.split(X, y):
        clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        oof_preds[val] = clf.predict(X[val])
    f1 = f1_score(y, oof_preds, average='macro')
    acc = accuracy_score(y, oof_preds)
    control_results.append({
        'Block': f"Block {b_idx}",
        'Macro-F1': round(f1, 4),
        'Accuracy': round(acc, 4)
    })
print("\n--- SAME-BLOCK RANDOM 5-FOLD CV (CONTROL) ---")
print(pd.DataFrame(control_results).to_string(index=False))

# Sequential Test 1: Single Previous Block -> Next Block (e.g. B1 -> B2, B2 -> B3 ...)
single_prev_results = []
for i in range(N_BLOCKS - 1):
    tr_df = blocks[i]
    te_df = blocks[i+1]
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    clf.fit(tr_df[features].values, tr_df['label_encoded'].values)
    preds = clf.predict(te_df[features].values)
    f1 = f1_score(te_df['label_encoded'].values, preds, average='macro')
    acc = accuracy_score(te_df['label_encoded'].values, preds)
    single_prev_results.append({
        'TRAIN BLOCKS': f"Block {i+1}",
        'TEST BLOCK': f"Block {i+2}",
        'MACRO-F1': round(f1, 4),
        'ACCURACY': round(acc, 4)
    })
print("\n--- SINGLE PREVIOUS BLOCK -> NEXT BLOCK ---")
print(pd.DataFrame(single_prev_results).to_string(index=False))

# Sequential Test 2: Rolling 2 Blocks -> Next Block (e.g. B1+B2 -> B3, B2+B3 -> B4 ...)
rolling_results = []
for i in range(1, N_BLOCKS - 1):
    tr_df = pd.concat([blocks[i-1], blocks[i]], axis=0).reset_index(drop=True)
    te_df = blocks[i+1]
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    clf.fit(tr_df[features].values, tr_df['label_encoded'].values)
    preds = clf.predict(te_df[features].values)
    f1 = f1_score(te_df['label_encoded'].values, preds, average='macro')
    acc = accuracy_score(te_df['label_encoded'].values, preds)
    rolling_results.append({
        'TRAIN BLOCKS': f"Blocks {i}+{i+1}",
        'TEST BLOCK': f"Block {i+2}",
        'MACRO-F1': round(f1, 4),
        'ACCURACY': round(acc, 4)
    })
print("\n--- ROLLING PREVIOUS 2 BLOCKS -> NEXT BLOCK ---")
print(pd.DataFrame(rolling_results).to_string(index=False))

# Sequential Test 3: Cumulative Previous All Blocks -> Next Block (e.g. B1..i -> B(i+1))
cumulative_results = []
for i in range(1, N_BLOCKS):
    tr_df = pd.concat(blocks[:i], axis=0).reset_index(drop=True)
    te_df = blocks[i]
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    clf.fit(tr_df[features].values, tr_df['label_encoded'].values)
    preds = clf.predict(te_df[features].values)
    f1 = f1_score(te_df['label_encoded'].values, preds, average='macro')
    acc = accuracy_score(te_df['label_encoded'].values, preds)
    cumulative_results.append({
        'TRAIN BLOCKS': f"Blocks 1..{i}" if i > 1 else "Block 1",
        'TEST BLOCK': f"Block {i+1}",
        'MACRO-F1': round(f1, 4),
        'ACCURACY': round(acc, 4)
    })
print("\n--- CUMULATIVE PREVIOUS ALL BLOCKS -> NEXT BLOCK ---")
print(pd.DataFrame(cumulative_results).to_string(index=False))

# Deep Dive: Look at first 2000 rows (Block 1) specifically
print("\n" + "="*90)
print("6. DEEP DIVE INTO BLOCK 1 & SUCCESSIVE BLOCKS")
print("="*90)

# Check correlations within Block 1 vs whole dataset
corr_b1 = blocks[0][features].corr()
corr_all = df[features].corr()
diff_corr = (corr_b1 - corr_all).abs().values
print(f"Max absolute correlation difference between Block 1 and All: {np.max(diff_corr):.4f}")
print(f"Mean absolute correlation difference between Block 1 and All: {np.mean(diff_corr):.4f}")

# Look at label sequencing within blocks (is there a repeating pattern or sequence of labels?)
print("\nFirst 30 labels in Block 1:")
print(blocks[0]['label'].head(30).tolist())

print("\nFirst 30 labels in Block 2:")
print(blocks[1]['label'].head(30).tolist())

print("\nFirst 30 labels in Block 3:")
print(blocks[2]['label'].head(30).tolist())

# Check label cycle length
labels_seq = df['label'].tolist()
# Check autocorrelation or period of labels
is_periodic = True
for period in range(1, 20):
    match_count = sum(1 for idx in range(len(labels_seq) - period) if labels_seq[idx] == labels_seq[idx+period])
    match_rate = match_count / (len(labels_seq) - period)
    print(f"Label Autocorrelation Lag {period:2d}: match rate = {match_rate*100:.2f}%")

# Check if validation.csv also follows a pattern
val_df = pd.read_csv('validation.csv')
print("\nValidation CSV Label count:")
print(val_df['label'].value_counts())
print("\nFirst 30 labels in validation.csv:")
print(val_df['label'].head(30).tolist())
for period in range(1, 15):
    val_labels = val_df['label'].tolist()
    match_count = sum(1 for idx in range(len(val_labels) - period) if val_labels[idx] == val_labels[idx+period])
    match_rate = match_count / (len(val_labels) - period)
    print(f"Val Label Autocorrelation Lag {period:2d}: match rate = {match_rate*100:.2f}%")

# Check if there is an exact deterministic label repetition:
print("\nUnique labels in sequence pattern (first 20 labels in train):")
print(labels_seq[:20])

print("\n" + "="*90)
print("7. FEATURE DIFFERENCES PER CLASS ACROSS BLOCKS")
print("="*90)
# Look at feature mean of 'normal' class across blocks 1 to 12
normal_means = pd.DataFrame()
for b_idx, b_df in enumerate(blocks, 1):
    sub = b_df[b_df['label'] == 'normal']
    normal_means[f'B{b_idx}'] = sub[features].mean()
print("Mean features for class 'normal' across Blocks 1..12:")
print(normal_means.round(3).to_string())

# Look at feature mean of 'fuzzer' class across blocks 1 to 12
fuzzer_means = pd.DataFrame()
for b_idx, b_df in enumerate(blocks, 1):
    sub = b_df[b_df['label'] == 'fuzzer']
    fuzzer_means[f'B{b_idx}'] = sub[features].mean()
print("\nMean features for class 'fuzzer' across Blocks 1..12:")
print(fuzzer_means.round(3).to_string())
