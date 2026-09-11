"""
Configuration parameters for CyberSentinel ML Pipeline, Structural Recovery, and Fallback
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
SUBMISSION_DIR = os.path.join(BASE_DIR, 'submission')

TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
VAL_PATH = os.path.join(DATA_DIR, 'validation.csv')
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, 'final_model.joblib')

# Feature specifications
RAW_FEATURES = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb',
    'dtcpb', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm'
]

CLASS_NAMES = [
    'analysis', 'backdoor', 'dos', 'exploit', 'fuzzer',
    'generic', 'normal', 'recon', 'shellcode', 'worm'
]

# Random seed
RANDOM_SEED = 42
EPSILON = 1e-6

# Calibration and Safety Gate Thresholds
CALIBRATION_ROWS = 2000
CHUNK_SIZE = 500
NUM_CHUNKS = 4
CYCLE_LENGTH = 10

# Structural safety gate parameters
MIN_CHUNK_CONSENSUS = 3          # At least 3 out of 4 independent chunks must agree on identical permutation
MIN_MEAN_ASSIGNMENT_MARGIN = 0.05 # Minimum average posterior margin over runner-up class
