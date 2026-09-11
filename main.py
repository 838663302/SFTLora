import inspect
import os
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import transformers
import trl
import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from process import process
from peft import LoraConfig

# 单条样本的最大 token 数。
# 推理时 agent 会把整套工具定义塞进 prompt（CodeBuddy 档位是 9 条工具，约 1800 token），
# 链式样本还有完整 SOP + 5 步工具调用 + 最终汇报，实测最长约 3650 token、平均约 2200 token。
# TRL 的 SFTConfig 默认 max_length 只有 1024，会把最关键的收尾部分截掉，必须显式放大。
MAX_LENGTH = 8192


def check_max_length(train_dataset, tokenizer, max_length):
    """训练前自检：用 tokenizer 实测每条样本长度，确认没有任何一条会被静默截断。

    按「工具菜单」分组报告最长值——不同 harness 档位的工具定义大小差别很大
    （9 条工具 ≈1800 token，单工具档位 ≈120 token），分组看才知道是谁把样本撑长的。
    超限直接抛错：截断会把尾部的最终汇报砍掉，训出来的模型是废的，不如别训。
    """
    def render(row):
        return tokenizer.apply_chat_template(
            row["messages"],
            tools=row.get("tools"),
            tokenize=False,
            add_generation_prompt=False,
        )

    print("=" * 64)
    print("[数据自检] 各工具档位的最长样本（token 数按 tokenizer 实测）:")
    longest_by_menu = {}
    for row in train_dataset:
        length = len(tokenizer(render(row))["input_ids"])
        fingerprint = tuple(sorted(t["function"]["name"] for t in (row.get("tools") or [])))
        if fingerprint not in longest_by_menu or length > longest_by_menu[fingerprint][0]:
            longest_by_menu[fingerprint] = (length, row)

    if not longest_by_menu:
        print("[数据自检] 训练集为空，跳过长度检查")
        return

    overall_length, longest_row = max(longest_by_menu.values(), key=lambda item: item[0])
    for fingerprint, (length, _row) in sorted(longest_by_menu.items(), key=lambda kv: -kv[1][0]):
        label = fingerprint[0] if len(fingerprint) == 1 else f"{len(fingerprint)} 条工具"
        print(f"    {length:6d} tokens   工具档位: {label}")

    last_message = longest_row["messages"][-1]
    last_kind = "工具调用" if last_message.get("tool_calls") else "文本收尾"
    print(f"[数据自检] 全量最长 {overall_length} tokens / max_length {max_length}"
          f"（该样本最后一回合: {last_kind}）")

    if overall_length > max_length:
        raise ValueError(
            f"[数据自检失败] 最长样本 {overall_length} tokens 超过 max_length {max_length}，"
            f"尾部会被静默截断（很可能正好砍掉最终汇报），请调大 MAX_LENGTH。"
        )
    print(f"[数据自检] 长度检查通过，余量 {max_length - overall_length} tokens")


def make_sft_config():
    """按当前 TRL 版本支持的字段名构造 SFTConfig（max_length / max_seq_length 兼容）。"""
    supported = set(inspect.signature(SFTConfig.__init__).parameters)

    extra_kwargs: dict[str, Any] = {}
    if "max_length" in supported:
        extra_kwargs["max_length"] = MAX_LENGTH
    elif "max_seq_length" in supported:
        extra_kwargs["max_seq_length"] = MAX_LENGTH
    else:
        print("[警告] 当前 TRL 版本没有 max_length / max_seq_length 字段，请手动确认截断行为")

    return SFTConfig(
        output_dir=str(config.CHECKPOINTS),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        fp16=True,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        max_grad_norm=1.0,
        num_train_epochs=3,
        logging_steps=10,
        eval_strategy = "steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        **extra_kwargs,
    )


def main():
    data_dict = process()
    model_name = "Qwen/Qwen3-1.7B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"训练集 {len(data_dict['train'])} 条 / 验证集 {len(data_dict['val'])} 条")
    check_max_length(data_dict["train"], tokenizer, MAX_LENGTH)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    )

    sft_config = make_sft_config()

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        peft_config=peft_config,
        train_dataset=data_dict["train"],
        eval_dataset=data_dict["val"],
        processing_class = tokenizer,
    )

    trainer.train()

    trainer.save_model(str(config.MODEL_PATH))
    tokenizer.save_pretrained(str(config.MODEL_PATH))
    print(f"LoRA 权重与 Tokenizer 已保存至: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
