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

# 工具调用本地智能纠错控制开关：
# False: 关闭本地纠错，完全交由模型原生输出处理（默认）
# True: 开启本地规则纠错（自动修复命令语法、参数格式、路径拼写及文本笔误等）
ENABLE_TOOL_CALL_CORRECTION = False
