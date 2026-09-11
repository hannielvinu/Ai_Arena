"""
CyberSentinel Feature Engineering Layer
Extracts Raw (20), Flow Asymmetry (4), State (3), Burst Dynamics (6), and Group-Relative (40) features.
Total: 73 engineered features.
"""
import numpy as np
import pandas as pd
from src.config import RAW_FEATURES, EPSILON

def extract_features(df: pd.DataFrame, group_size: int = 10, eps: float = EPSILON) -> pd.DataFrame:
    """
    Constructs the 73-feature representation from the 20 raw numerical network features:
    1. Raw 20 numerical traffic features
    2. 13 Flow Asymmetry, State, and Burst features
    3. 40 Group-Relative features (x - mean, (x - mean)/std) within each 10-row slice
    Uses ONLY feature columns — strictly zero knowledge of labels.
    """
    # Ensure all required features are present
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in input DataFrame: {missing}")

    res = df[RAW_FEATURES].copy()

    # 1. Flow Asymmetry
    res['spkts_minus_dpkts'] = df['spkts'] - df['dpkts']
    res['pkt_asym'] = (df['spkts'] - df['dpkts']) / (df['spkts'] + df['dpkts'] + eps)
    res['sload_minus_dload'] = df['sload'] - df['dload']
    res['load_asym'] = (df['sload'] - df['dload']) / (df['sload'] + df['dload'] + eps)

    # 2. State Features
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
    if n_groups > 0:
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
    else:
        # Fallback for datasets smaller than 10 rows
        for f in RAW_FEATURES:
            res[f'{f}_grp_diff'] = 0.0
            res[f'{f}_grp_z'] = 0.0

    res = res.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return res
