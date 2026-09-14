import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import config
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = getattr(config, "BASE_MODEL_NAME", "Qwen/Qwen3-4B")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            local_files_only=True,
        )
    except Exception as exc:
        print(f"[提示] AutoModelForCausalLM 加载异常，尝试 AutoModelForImageTextToText: {exc}")
        from transformers import AutoModelForImageTextToText
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.float16,
            local_files_only=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    lora_model = PeftModel.from_pretrained(
        base_model,
        str(config.MODEL_PATH),
    )

    merged_model = lora_model.merge_and_unload()  # type: ignore

    merged_model.save_pretrained(str(config.MERGED_PATH))
    tokenizer.save_pretrained(str(config.MERGED_PATH))
    print(f"完整合并模型已保存至: {config.MERGED_PATH}")


if __name__ == "__main__":
    main()
