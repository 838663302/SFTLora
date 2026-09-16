import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"

# 判断是否为 Kaggle 环境
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").exists()

# 设置工作路径
if IS_KAGGLE:
    WORKING_PATH = Path("/kaggle/working")
    DATA_DIR = Path('/kaggle/working/SFTLora/data')
else:
    WORKING_PATH = Path(__file__).resolve().parent
    DATA_DIR = WORKING_PATH / 'data'

MODEL_PATH = WORKING_PATH / "checkpoints" / "final"
CHECKPOINTS = WORKING_PATH / "checkpoints"
MERGED_PATH = WORKING_PATH / "checkpoints" / "merged"

# 基础模型配置：选用纯文本因果语言模型 Qwen3-8B
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen3-8B")
