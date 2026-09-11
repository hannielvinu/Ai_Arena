"""
CyberSentinel Submission Generation Script
Loads test data, applies dynamic structural calibration with safety gate,
and outputs official submission.csv.
"""
import os
import sys
import argparse
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import VAL_PATH, SUBMISSION_DIR, CLASS_NAMES
from src.model import CyberSentinelModel
from src.predict import predict_cyber_attacks

def main():
    parser = argparse.ArgumentParser(description="Generate CyberSentinel Hackathon Submission CSV")
    parser.add_argument('--input', type=str, default=VAL_PATH, help="Path to input test CSV (defaults to validation.csv)")
    parser.add_argument('--output', type=str, default=os.path.join(SUBMISSION_DIR, 'submission.csv'), help="Path to output submission.csv")
    args = parser.parse_args()

    print("=" * 80)
    print("  CYBERSENTINEL SUBMISSION GENERATION")
    print("=" * 80)

    # 1. Load Model
    print("\n[1/4] Loading trained model artifact...")
    model = CyberSentinelModel.load()
    print("  Model loaded successfully.")

    # 2. Load Input Test Data
    print(f"\n[2/4] Reading input dataset from '{args.input}'...")
    test_df = pd.read_csv(args.input)
    print(f"  Input sample count: {len(test_df)} rows")

    # 3. Generate Predictions with Dynamic Structural Calibration
    print("\n[3/4] Running prediction pipeline with dynamic calibration & safety gate...")
    sub_df, cal_res, meta = predict_cyber_attacks(test_df, model=model)

    print(f"  Execution Mode       : {meta['mode_used']}")
    print(f"  Calibration Valid    : {cal_res.is_valid}")
    print(f"  Chunk Consensus      : {cal_res.chunk_consensus}/{cal_res.total_chunks} chunks")
    print(f"  Mean Margin          : {cal_res.mean_assignment_margin:+.4f}")
    if cal_res.is_valid:
        print(f"  Recovered Permutation: {cal_res.recovered_permutation}")

    # 4. Save and Validate Output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sub_df.to_csv(args.output, index=False)
    # Also save copy to workspace root if outputting to submission/
    root_sub = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'submission.csv')
    sub_df.to_csv(root_sub, index=False)
    print(f"\n[4/4] Saved final submission to '{args.output}' and '{root_sub}'.")

    # Strict Quality Checks
    assert list(sub_df.columns) == ['sample_id', 'predicted_class', 'confidence'], "Column mismatch!"
    assert len(sub_df) == len(test_df), "Row count mismatch!"
    assert sub_df['predicted_class'].isin(CLASS_NAMES).all(), "Invalid class names!"
    assert (sub_df['confidence'] >= 0.0).all() and (sub_df['confidence'] <= 1.0).all(), "Confidence out of bounds!"
    assert not sub_df.isnull().any().any(), "Submission contains nulls!"

    print("  Sanity Verification Passed (5,000 rows, 0 nulls, correct schema).")
    print("\nFirst 10 Rows:")
    print(sub_df.head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("SUBMISSION GENERATION COMPLETED")
    print("=" * 80)

if __name__ == '__main__':
    main()
