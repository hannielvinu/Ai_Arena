"""
Deep Analysis of Blocks & Feature Relationships Across Blocks
"""
import pandas as pd
import numpy as np

df_tr = pd.read_csv('train.csv')
df_val = pd.read_csv('validation.csv')
features = [c for c in df_tr.columns if c != 'label']

# 12 blocks of 2000 rows
blocks = [df_tr.iloc[i*2000 : (i+1)*2000][features].reset_index(drop=True) for i in range(12)]

print("Checking correlation between corresponding rows of Block 1 and Block 2..12:")
for b in range(1, 12):
    # Element-wise correlation across all 2000x20 entries
    diff = (blocks[b] - blocks[0]).abs().values
    corr = np.corrcoef(blocks[0].values.flatten(), blocks[b].values.flatten())[0, 1]
    print(f"Block 1 vs Block {b+1}: flattened correlation = {corr:.4f}, mean abs diff = {np.mean(diff):.4f}")

# Check if individual features correlate between Block 1 and Block 2 row-by-row
print("\nRow-by-row correlation per feature between Block 1 and Block 2:")
for f in features:
    r = np.corrcoef(blocks[0][f], blocks[1][f])[0, 1]
    print(f"  {f:16s}: r = {r:+.4f}")

# Check if there is an index feature or cumulative sum
print("\nChecking cumulative properties across 24000 rows:")
for f in ['dur', 'rate', 'sbytes', 'dbytes', 'sttl', 'dttl']:
    diff_1 = df_tr[f].diff().dropna()
    print(f"  {f:10s} diff mean: {diff_1.mean():+.4f}, std: {diff_1.std():.4f}")

# Check if validation.csv is a permutation of the 10 classes
print("\nValidation class cycle:")
val_classes_cycle = df_val['label'].iloc[:10].tolist()
print("Cycle in Validation:", val_classes_cycle)
train_classes_cycle = df_tr['label'].iloc[:10].tolist()
print("Cycle in Train     :", train_classes_cycle)

# Check if the validation set's cycle permutation maps to train:
# In train: 0: normal, 1: fuzzer, 2: analysis, 3: backdoor, 4: dos, 5: exploit, 6: generic, 7: recon, 8: shellcode, 9: worm
# In val  : 0: backdoor (tr#3), 1: normal (tr#0), 2: recon (tr#7), 3: dos (tr#4), 4: fuzzer (tr#1), 5: shellcode (tr#8), 6: exploit (tr#5), 7: analysis (tr#2), 8: worm (tr#9), 9: generic (tr#6)
val_to_train_index_map = [train_classes_cycle.index(c) for c in val_classes_cycle]
print("Validation Cycle -> Train Cycle indices:", val_to_train_index_map)
