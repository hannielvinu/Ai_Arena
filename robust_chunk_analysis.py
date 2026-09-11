"""
Robust Chunk Analysis & Permutation Agreement Validation
Zero hidden labels. Computes assignment margins, chunk-by-chunk consensus, and fallback triggers.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from scipy.optimize import linear_sum_assignment

print("=" * 80)
print("  ROBUST CHUNK ANALYSIS & ASSIGNMENT MARGIN CALCULATION")
print("=" * 80)

# Load data
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('validation.csv')

features = [c for c in train_df.columns if c != 'label']
class_names = sorted(list(train_df['label'].unique()))
label_to_id = {c: i for i, c in enumerate(class_names)}
id_to_label = {i: c for i, c in enumerate(class_names)}

X_tr = train_df[features].values
y_tr = np.array([label_to_id[l] for l in train_df['label']])

# Features ONLY from evaluation set
X_eval = val_df[features].values

# Fit Base ExtraTrees
clf = ExtraTreesClassifier(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)

def recover_and_score(X_slice):
    n_groups = len(X_slice) // 10
    probs = clf.predict_proba(X_slice) # (N, 10)
    probs_grp = probs.reshape(n_groups, 10, 10)
    M = np.mean(probs_grp, axis=0) # (10, 10) position-by-class posterior
    
    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(-M)
    cycle = [class_names[c] for c in col_ind]
    
    # Assignment margins per position:
    # Margin = (assigned_prob - second_highest_prob_in_row)
    margins = []
    assigned_probs = []
    for p in range(10):
        assigned_c = col_ind[p]
        assigned_p = M[p, assigned_c]
        assigned_probs.append(assigned_p)
        sorted_probs = np.sort(M[p])
        # second highest:
        second_highest = sorted_probs[-2] if sorted_probs[-1] == assigned_p else sorted_probs[-1]
        margins.append(assigned_p - second_highest)
        
    mean_margin = float(np.mean(margins))
    min_margin = float(np.min(margins))
    mean_assigned_prob = float(np.mean(assigned_probs))
    
    return cycle, col_ind, mean_margin, min_margin, mean_assigned_prob, M

print("\n--- 1. FOUR INDEPENDENT 500-ROW CHUNKS (ROWS 0-2000) ---")
chunk_cycles = []
chunk_margins = []

for i in range(4):
    start = i * 500
    end = (i + 1) * 500
    X_chunk = X_eval[start:end]
    cycle, col_ind, mean_m, min_m, mean_p, _ = recover_and_score(X_chunk)
    chunk_cycles.append(cycle)
    chunk_margins.append(mean_m)
    print(f"  Chunk {i+1} (Rows {start:4d}-{end:4d} | 50 groups):")
    print(f"    Cycle: {cycle}")
    print(f"    Mean Assignment Margin: {mean_m:+.4f} | Min Margin: {min_m:+.4f} | Mean Prob: {mean_p:.4f}")

# Pairwise Agreement between Chunks
print("\n--- 2. CHUNK-BY-CHUNK PAIRWISE AGREEMENT MATRIX ---")
for i in range(4):
    row_str = f"  Chunk {i+1}: "
    for j in range(4):
        matches = sum(1 for a, b in zip(chunk_cycles[i], chunk_cycles[j]) if a == b)
        row_str += f"C{j+1}:{matches*10:3d}%  "
    print(row_str)

# Cumulative Calibration Sizes
print("\n--- 3. CUMULATIVE CALIBRATION SIZES (500, 1000, 1500, 2000) ---")
cumulative_cycles = {}
for n_rows in [500, 1000, 1500, 2000]:
    X_sub = X_eval[:n_rows]
    cycle, col_ind, mean_m, min_m, mean_p, M = recover_and_score(X_sub)
    cumulative_cycles[n_rows] = cycle
    print(f"  Cumulative {n_rows:4d} rows ({n_rows//10:3d} groups):")
    print(f"    Cycle: {cycle}")
    print(f"    Mean Assignment Margin: {mean_m:+.4f} | Min Margin: {min_m:+.4f} | Mean Prob: {mean_p:.4f}")

# Detailed Per-Position Analysis on 2000 rows
final_cycle, col_ind_2k, mean_m_2k, min_m_2k, mean_p_2k, M_2k = recover_and_score(X_eval[:2000])

print("\n--- 4. DETAILED POSITION-BY-POSITION MARGINS (2000 ROWS) ---")
for p in range(10):
    c_id = col_ind_2k[p]
    c_name = class_names[c_id]
    prob = M_2k[p, c_id]
    sorted_probs = np.sort(M_2k[p])
    runner_up = sorted_probs[-2] if sorted_probs[-1] == prob else sorted_probs[-1]
    runner_up_name = class_names[np.where(M_2k[p] == runner_up)[0][0]]
    diff = prob - runner_up
    print(f"  Pos {p}: {c_name:<10} (p={prob:.4f}) vs {runner_up_name:<10} (p={runner_up:.4f}) -> Margin = +{diff:.4f}")

