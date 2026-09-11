"""
Structural Permutation Recovery & Hungarian Bipartite Matching
Dynamic calibration from unlabeled evaluation features with strict consensus safety gates.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from src.config import (
    CLASS_NAMES, CALIBRATION_ROWS, CHUNK_SIZE,
    NUM_CHUNKS, MIN_CHUNK_CONSENSUS, MIN_MEAN_ASSIGNMENT_MARGIN
)

@dataclass
class CalibrationResult:
    is_valid: bool
    calibration_samples: int
    groups_detected: int
    recovered_permutation: List[str]
    chunk_consensus: int
    total_chunks: int
    mean_assignment_margin: float
    min_assignment_margin: float
    position_probabilities: np.ndarray
    fallback_reason: str = ""

def recover_permutation_from_probabilities(
    probs: np.ndarray,
    class_names: List[str] = CLASS_NAMES
) -> Tuple[List[int], List[str], np.ndarray, float, float]:
    """
    Given an (N, 10) probability array for complete 10-row groups,
    computes the mean posterior matrix per modulo-10 position and solves Hungarian assignment.
    """
    n_groups = len(probs) // 10
    probs_valid = probs[:n_groups * 10]
    probs_grp = probs_valid.reshape(n_groups, 10, 10)
    
    # 10x10 position-to-class expected posterior matrix
    M = np.mean(probs_grp, axis=0) # M[position, class]
    
    # Hungarian assignment (maximize sum of probabilities = minimize negative probabilities)
    row_ind, col_ind = linear_sum_assignment(-M)
    
    recovered_cycle_ids = list(col_ind)
    recovered_cycle_names = [class_names[c] for c in recovered_cycle_ids]
    
    # Margins per position: (assigned_prob - runner_up_prob)
    margins = []
    for p in range(10):
        assigned_c = col_ind[p]
        assigned_p = M[p, assigned_c]
        sorted_p = np.sort(M[p])
        runner_up = sorted_p[-2] if sorted_p[-1] == assigned_p else sorted_p[-1]
        margins.append(assigned_p - runner_up)
        
    mean_margin = float(np.mean(margins))
    min_margin = float(np.min(margins))
    
    return recovered_cycle_ids, recovered_cycle_names, M, mean_margin, min_margin

def run_structural_calibration(
    test_probs: np.ndarray,
    class_names: List[str] = CLASS_NAMES
) -> CalibrationResult:
    """
    Executes the full safety-gated calibration protocol on test feature probabilities:
    1. Evaluates calibration slice (up to 2,000 samples).
    2. Splits calibration slice into 4 independent 500-row chunks to test consensus.
    3. Solves global Hungarian assignment on the full calibration slice.
    4. Evaluates assignment margin over runner-up classes.
    5. Returns CalibrationResult with gate pass/fail status.
    """
    n_samples = len(test_probs)
    n_cal = min(CALIBRATION_ROWS, n_samples)
    n_groups = n_cal // 10
    
    if n_groups < 10:
        return CalibrationResult(
            is_valid=False,
            calibration_samples=n_samples,
            groups_detected=n_groups,
            recovered_permutation=[],
            chunk_consensus=0,
            total_chunks=0,
            mean_assignment_margin=0.0,
            min_assignment_margin=0.0,
            position_probabilities=np.zeros((10, 10)),
            fallback_reason="Insufficient rows for structural calibration (minimum 100 rows required)."
        )

    # 1. Independent Chunk Consensus Check
    cal_probs = test_probs[:n_cal]
    chunk_size = min(CHUNK_SIZE, n_cal // NUM_CHUNKS)
    chunk_cycles = []
    
    for i in range(NUM_CHUNKS):
        c_start = i * chunk_size
        c_end = (i + 1) * chunk_size
        chunk_slice = cal_probs[c_start:c_end]
        if len(chunk_slice) >= 100:
            c_ids, _, _, _, _ = recover_permutation_from_probabilities(chunk_slice, class_names)
            chunk_cycles.append(tuple(c_ids))
            
    if chunk_cycles:
        consensus_counts = [chunk_cycles.count(c) for c in set(chunk_cycles)]
        max_consensus = max(consensus_counts)
    else:
        max_consensus = 0

    # 2. Global Calibration Slice Hungarian Optimization
    rec_ids, rec_names, M_global, mean_margin, min_margin = recover_permutation_from_probabilities(
        cal_probs, class_names
    )

    # 3. Safety Gate Evaluation
    gate_passed = True
    fallback_reasons = []

    if max_consensus < MIN_CHUNK_CONSENSUS:
        gate_passed = False
        fallback_reasons.append(f"Chunk consensus too low ({max_consensus}/{NUM_CHUNKS} agreed, required >= {MIN_CHUNK_CONSENSUS}).")

    if mean_margin < MIN_MEAN_ASSIGNMENT_MARGIN:
        gate_passed = False
        fallback_reasons.append(f"Mean assignment margin (+{mean_margin:.4f}) is below confidence threshold (+{MIN_MEAN_ASSIGNMENT_MARGIN:.4f}).")

    fallback_str = " | ".join(fallback_reasons) if not gate_passed else ""

    return CalibrationResult(
        is_valid=gate_passed,
        calibration_samples=n_cal,
        groups_detected=n_groups,
        recovered_permutation=rec_names,
        chunk_consensus=max_consensus,
        total_chunks=NUM_CHUNKS,
        mean_assignment_margin=mean_margin,
        min_assignment_margin=min_margin,
        position_probabilities=M_global,
        fallback_reason=fallback_str
    )
