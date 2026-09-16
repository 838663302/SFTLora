import inspect
import os
from typing import Any

# TPU 环境配置：设置 PJRT 运行时后端为 TPU
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import transformers
import trl
import config
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq)
from trl import SFTConfig, SFTTrainer
from process import process
from peft import LoraConfig

# 单条样本的最大 token 数（与训练集最长样本 3608 匹配，取 4096）
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "4096"))


def is_master_process() -> bool:
    """判定当前进程是否为主进程（Rank 0）。
    优先通过 torch_xla 的 master_ordinal 判定；非 XLA 启动时检查 LOCAL_RANK / RANK。
    """
    try:
        import torch_xla.core.xla_model as xm
        return xm.is_master_ordinal(local=False)
    except Exception:
        pass

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    return local_rank == 0


def check_precision(model):
    """训练前自检：打印模型各参数 dtype，确保 TPU 上均为原生 bfloat16。"""
    if not is_master_process():
        return

    print("=" * 64)
    all_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    named = list(model.named_parameters())
    trainable = [(name, param.dtype) for name, param in named if param.requires_grad]
    frozen = [(name, param.dtype) for name, param in named if not param.requires_grad]
    print(f"[TPU 精度自检] 模型全部参数 dtype: {all_dtypes}")
    print(f"[TPU 精度自检] 可训练参数 {len(trainable)} 个，dtype: {sorted({str(d) for _, d in trainable})}")
    print(f"[TPU 精度自检] 冻结基座参数 {len(frozen)} 个，dtype: {sorted({str(d) for _, d in frozen})}")

    non_bf16 = [name for name, dtype in trainable if dtype != torch.bfloat16]
    if non_bf16:
        print(f"[TPU 精度自检][提示] 部分可训练参数为非 bfloat16 (如 fp32 优化器主权重，此属正常): {non_bf16[:3]}")
    else:
        print("[TPU 精度自检] 所有可训练参数均处于原生 bfloat16 模式，TPU 算力单元已满血就绪。")
    print("=" * 64)


def tokenize_assistant_only(example, tokenizer, max_length):
    """按聊天模板构建序列，并对 assistant 输出段以外的内容进行 loss mask (labels=-100)。"""
    prompt_ids = tokenizer(
        tokenizer.apply_chat_template(example["messages"][:-1], tools=example.get("tools"),
                                      tokenize=False, add_generation_prompt=True))["input_ids"]
    full_ids = tokenizer(
        tokenizer.apply_chat_template(example["messages"], tools=example.get("tools"),
                                      tokenize=False, add_generation_prompt=False))["input_ids"]

    if full_ids[:len(prompt_ids)] != prompt_ids:
        # 模板行为与预期不一致时保守退回「整段监督」，并在 check_supervision 里报出来
        labels = list(full_ids)
    else:
        labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])

    return {"input_ids": full_ids[:max_length], "labels": labels[:max_length]}


def check_supervision(rows):
    """统计并校验 loss 监督 token 的覆盖率。"""
    total = sum(len(row["input_ids"]) for row in rows)
    supervised = sum(sum(1 for token in row["labels"] if token != -100) for row in rows)
    ratio = supervised / total if total else 0.0
    print(f"[监督自检] {len(rows)} 条样本：全量 {total} token / 计入 loss {supervised} token"
          f"（{ratio:.1%}）")
    if supervised == 0:
        raise ValueError("[监督自检失败] 没有任何 token 参与 loss，mask 写错了。")
    if ratio > 0.5:
        print("[监督自检][警告] 计入 loss 的占比过高，mask 很可能没生效（前缀校验全失败？）")
    return ratio


def make_sft_config():
    """构造专为 TPU 优化的 SFTConfig。
    - bf16=True, fp16=False：充分调用 TPU V5e 矩阵乘法单元 (MXU)。
    - optim='adamw_torch'：标准 PyTorch AdamW，替代 CUDA 专属的 paged_adamw_8bit。
    - dataloader_drop_last=True：固定 batch 大小，配合静态 padding 避免 XLA 形状重新编译。
    """
    supported = set(inspect.signature(SFTConfig.__init__).parameters)

    extra_kwargs: dict[str, Any] = {}
    if "max_length" in supported:
        extra_kwargs["max_length"] = MAX_LENGTH
    elif "max_seq_length" in supported:
        extra_kwargs["max_seq_length"] = MAX_LENGTH

    if "dataset_kwargs" not in supported:
        raise RuntimeError(
            "当前 TRL 的 SFTConfig 不支持 dataset_kwargs，无法跳过 TRL 的数据预处理；"
            "assistant 段 mask 依赖它，请升级 TRL 后再训。"
        )
    extra_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
    extra_kwargs["remove_unused_columns"] = False

    from transformers.utils import is_torch_xla_available

    # 精度检测：TPU 环境下使用原生 bf16；若本地在非 TPU 机器自检或调试，做自适应兼容
    tpu_supported = is_torch_xla_available()
    use_bf16 = tpu_supported or (torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    # 若环境变量启用 FSDP (如 USE_FSDP=1 且处于 XLA 环境)，配置 TPU 原生 XLA FSDP
    if os.environ.get("USE_FSDP", "0") == "1" and tpu_supported:
        extra_kwargs["fsdp"] = "full_shard"
        extra_kwargs["fsdp_config"] = {
            "xla": True,
            "xla_fsdp_grad_ckpt": True,
        }

    # 训练批次设置：TPU v5e-8 单核 16GB HBM
    # 单卡 batch=1，梯度累积=4；8 核并行时全局有效 batch size = 1 * 8 * 4 = 32
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    grad_acc = int(os.environ.get("GRAD_ACC", "4"))
    lr = float(os.environ.get("LEARNING_RATE", "1.5e-4"))
    num_epochs = int(os.environ.get("NUM_EPOCHS", "2"))

    return SFTConfig(
        output_dir=str(config.CHECKPOINTS),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        dataloader_drop_last=True,  # 保证静态形状，杜绝末尾不完整 batch 触发 XLA 重编译
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        learning_rate=lr,
        bf16=use_bf16,              # TPU 环境下为 True，原生调用 MXU
        fp16=False,                 # TPU 严禁使用 fp16 + GradScaler
        optim="adamw_torch",        # TPU 使用标准的 adamw_torch
        lr_scheduler_type="cosine",
        warmup_steps=10,
        max_grad_norm=1.0,
        num_train_epochs=num_epochs,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="tensorboard",
        **extra_kwargs,
    )


def patch_peft_tensor_parallel_compat():
    """兼容 peft 与 transformers>=5.16 的 ImportError（张量并行分片）。"""
    try:
        from peft.utils import save_and_load
    except Exception:
        return

    original = getattr(save_and_load, "_maybe_shard_state_dict_for_tp", None)
    if original is None:
        return

    def has_tp_plan(model) -> bool:
        return any(getattr(module, "_hf_tp_plan", None) is not None for module in model.modules())

    def safe_maybe_shard_state_dict_for_tp(model, state_dict, adapter_name):
        try:
            return original(model, state_dict, adapter_name)
        except ImportError as exc:
            if has_tp_plan(model):
                raise
            if is_master_process():
                print(f"[兼容] 跳过 peft 张量并行分片（当前模型非 TP）：{exc}")
            return None

    save_and_load._maybe_shard_state_dict_for_tp = safe_maybe_shard_state_dict_for_tp


def main():
    patch_peft_tensor_parallel_compat()

    is_master = is_master_process()

    data_dict = process()
    # 默认选用 8B 级别模型（可通过 BASE_MODEL_NAME 环境变量覆盖）
    model_name = os.environ.get("BASE_MODEL_NAME", getattr(config, "BASE_MODEL_NAME", "Qwen/Qwen3-8B"))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if is_master:
        world_size = os.environ.get("WORLD_SIZE", "1")
        print("=" * 64)
        print(f"[TPU 训练引擎启动] 模型: {model_name}")
        print(f"[分布式拓扑] world_size={world_size} | 运行环境: {'Kaggle' if config.IS_KAGGLE else 'Local'}")
        print(f"[数据集] 训练集 {len(data_dict['train'])} 条 / 验证集 {len(data_dict['val'])} 条")
        print(f"[序列配置] MAX_LENGTH={MAX_LENGTH}")
        print("=" * 64)

    # 1. 数据映射与 token 级 mask
    train_dataset = data_dict["train"].map(
        tokenize_assistant_only,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH},
        remove_columns=data_dict["train"].column_names,
        desc="tokenize + mask (train)",
    )
    eval_dataset = data_dict["val"].map(
        tokenize_assistant_only,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH},
        remove_columns=data_dict["val"].column_names,
        desc="tokenize + mask (val)",
    )

    if is_master:
        check_supervision(train_dataset)

    # 2. 模型加载（TPU 原生 BF16，移除 bitsandbytes 4-bit 量化与 CUDA device_map）
    # 注：不要显式传入 device_map，由 Trainer 与 PyTorch/XLA 自动将子图分派至各 TPU Core
    from transformers.utils import is_torch_xla_available
    tpu_supported = is_torch_xla_available()
    model_dtype = torch.bfloat16 if (tpu_supported or (torch.cuda.is_available() and torch.cuda.is_bf16_supported())) else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            attn_implementation="sdpa",
        )
    except Exception as exc:
        if is_master:
            print(f"[提示] 使用 sdpa 加载模型失败，回退到默认注意力实现: {exc}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
        )

    model.config.use_cache = False

    # 启用输入梯度，确保 LoRA 与 gradient_checkpointing 兼容
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    sft_config = make_sft_config()

    # 3. LoRA 配置（TPU v5e-8 算力充沛，可配置 r=16 / alpha=32 获得更高表达能力）
    lora_r = int(os.environ.get("LORA_R", "16"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", str(lora_r * 2)))
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
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

    # 4. TPU 核心优化：静态形状 Padding
    # XLA 图编译器对于动态 Tensor 形状会触发极慢的重复 JIT 编译（每步 30-60 秒）。
    # 使用 padding="max_length" 保证每个 batch 形状绝对恒定，实现「一次编译，全程极速」。
    use_static_padding = os.environ.get("PAD_TO_MAX_LENGTH", "1") == "1"
    if use_static_padding:
        collator = DataCollatorForSeq2Seq(
            tokenizer,
            padding="max_length",
            max_length=MAX_LENGTH,
            label_pad_token_id=-100,
        )
    else:
        collator = DataCollatorForSeq2Seq(
            tokenizer,
            padding=True,
            pad_to_multiple_of=128,
            label_pad_token_id=-100,
        )

    # 5. 初始化 SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    # 自检参数 dtype
    check_precision(trainer.model)

    if is_master:
        print("[TPU 训练开始]")

    # 6. 开始训练
    trainer.train()

    # 7. 保存最终 LoRA 权重与 Tokenizer
    trainer.save_model(str(config.MODEL_PATH))
    if is_master:
        tokenizer.save_pretrained(str(config.MODEL_PATH))
        print(f"[训练完成] LoRA 权重与 Tokenizer 已保存至: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
