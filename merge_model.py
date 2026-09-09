import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import config
import torch

def main():
    model_name = "Qwen/Qwen3-1.7B"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
