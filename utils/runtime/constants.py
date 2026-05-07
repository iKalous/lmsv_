"""Constants used in the pretrain_gpt.py script."""

import os

# Log directory configuration
import sys

IS_PTA = True 

for path in sys.path:
    if "msadapter" in path.lower() or "msa" in path.lower():
        IS_PTA = False
        break




# ======== LOG PATH ===========
if IS_PTA:
    LOG_DIR = "res/training_log_pta"
else:
    LOG_DIR = "res/training_log_msa"

MF_LOG_DIR = "res/training_log_mf"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MF_LOG_DIR, exist_ok=True)
# Log file pattern
LOG_FILE_PATTERN = f"{LOG_DIR}/training_log-*.csv"
MF_LOG_FILE_PATTERN = f"{MF_LOG_DIR}/training_log-*.csv"

# Log round tracker (mutation round; external input)
LOG_ROUND_ENV = "MUTATE_ROUND"

def get_log_round():
    val = os.getenv(LOG_ROUND_ENV, "").strip()
    if not val:
        os.environ[LOG_ROUND_ENV] = "0"
        return 0
        # raise ValueError(f"env {LOG_ROUND_ENV} is required for training log round")
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"env {LOG_ROUND_ENV} must be int, got: {val}")


# ======== MODEL SAVE PATH ===========
ROOT_PATH = "/dev/shm"
if IS_PTA:
    MODEL_SAVE_PATH = f"{ROOT_PATH}/pretrained_gpt_pta"
else:
    MODEL_SAVE_PATH = f"{ROOT_PATH}/pretrained_gpt_msa"
MF_MODEL_SAVE_PATH = f"{ROOT_PATH}/pretrained_gpt_mf"