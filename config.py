import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"

# 判断是否为 Kaggle 环境
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").exists()

# 设置工作路径
if IS_KAGGLE:
    WORKING_PATH = Path("/kaggle/working")
    DATA_DIR = Path('/kaggle/input/datasets/xiaonanhaiaichixigua/sftlorabtp/data')
else:
    WORKING_PATH = Path(__file__).resolve().parent
    DATA_DIR = WORKING_PATH / 'data'

MODEL_PATH = WORKING_PATH / "checkpoints" / "final"
CHECKPOINTS = WORKING_PATH / "checkpoints"
MERGED_PATH = WORKING_PATH / "checkpoints" / "merged"
