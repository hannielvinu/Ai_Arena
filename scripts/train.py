"""
CyberSentinel Production Training Script
Trains ExtraTrees model on all 29,000 labeled samples (train.csv + validation.csv)
Saves serialized model artifact to models/final_model.joblib
"""
import os
import sys
import time
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TRAIN_PATH, VAL_PATH, FINAL_MODEL_PATH
from src.model import CyberSentinelModel

def main():
    print("=" * 80)
    print("  CYBERSENTINEL PRODUCTION MODEL TRAINING")
    print("=" * 80)

    # 1. Load Data
    print("\n[1/3] Loading labeled datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    
    full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    print(f"  train.csv      : {len(train_df)} rows")
    print(f"  validation.csv : {len(val_df)} rows")
    print(f"  Total Labeled  : {len(full_df)} rows")

    # 2. Train Model
    print("\n[2/3] Training ExtraTrees Classifier with 73 Engineered Features...")
    t0 = time.time()
    model = CyberSentinelModel(n_estimators=350, max_depth=16, min_samples_split=4)
    model.fit(full_df)
    elapsed = time.time() - t0
    print(f"  Training finished in {elapsed:.2f} seconds.")

    # 3. Save Model Artifact
    print(f"\n[3/3] Saving serialized model to '{FINAL_MODEL_PATH}'...")
    model.save(FINAL_MODEL_PATH)
    print(f"  Model saved successfully (File size: {os.path.getsize(FINAL_MODEL_PATH)/(1024*1024):.2f} MB).")

    print("\n" + "=" * 80)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == '__main__':
    main()
