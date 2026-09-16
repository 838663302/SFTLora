import inspect
import os
from typing import Any

# 必须在 import torch 之前设置，启用虚拟内存段扩展，彻底解决显存碎片导致的 OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if "LOCAL_RANK" not in os.environ:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import transformers
import trl
import config
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq)
from trl import SFTConfig, SFTTrainer
from process import process
from peft import LoraConfig, prepare_model_for_kbit_training

# 单条样本的最大 token 数
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "4352"))




def check_precision(model):
    """训练前自检：打印模型与可训练参数的 dtype，提前拦住「fp16 AMP 撞 bf16 权重」这类必崩组合。

    PyTorch 的 GradScaler 只认识 fp32/fp16 梯度：只要有一个可训练参数是 bf16，
    `unscale_` 就会直接抛
    `NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'`。
    """
    print("=" * 64)
    all_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    named = list(model.named_parameters())
    trainable = [(name, param.dtype) for name, param in named if param.requires_grad]
    frozen = [(name, param.dtype) for name, param in named if not param.requires_grad]
    print(f"[精度自检] 模型参数 dtype: {all_dtypes}")
    print(f"[精度自检] 可训练参数 {len(trainable)} 个，dtype: {sorted({str(d) for _, d in trainable})}")
    # 冻结侧残留 16-bit 说明基座未量化部分没被上转 fp32，T4 上会走 bf16 模拟、白白变慢
    print(f"[精度自检] 冻结参数 {len(frozen)} 个，dtype: {sorted({str(d) for _, d in frozen})}")

    risky = [name for name, dtype in trainable if dtype in (torch.float16, torch.bfloat16)]
    if risky:
        print(f"[精度自检][警告] 以下可训练参数不是 fp32，与 fp16 AMP 的 GradScaler 冲突: {risky[:5]}"
              f"{' ...' if len(risky) > 5 else ''}")


def tokenize_assistant_only(example, tokenizer, max_length):
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
    """按当前 TRL 版本支持的字段名构造 SFTConfig（max_length / max_seq_length 兼容）。"""
    supported = set(inspect.signature(SFTConfig.__init__).parameters)

    extra_kwargs: dict[str, Any] = {}
    if "max_length" in supported:
        extra_kwargs["max_length"] = MAX_LENGTH
    elif "max_seq_length" in supported:
        extra_kwargs["max_seq_length"] = MAX_LENGTH
    else:
        print("[警告] 当前 TRL 版本没有 max_length / max_seq_length 字段，请手动确认截断行为")

    # assistant 段 mask 依赖「跳过 TRL 的数据预处理」（数据已在 main 里 tokenize 并打好 labels）
    # 与「保住 labels 列」。缺任一项都会静默训错，所以这里直接卡死，不做降级。
    if "dataset_kwargs" not in supported:
        raise RuntimeError(
            "当前 TRL 的 SFTConfig 不支持 dataset_kwargs，无法跳过 TRL 的数据预处理；"
            "assistant 段 mask 依赖它，请升级 TRL 后再训。"
        )
    extra_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
    extra_kwargs["remove_unused_columns"] = False

    return SFTConfig(
        output_dir=str(config.CHECKPOINTS),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # 核心修复 1：非重入 checkpointing，反向传播显存及时释放且杜绝 DDP 警告
        ddp_find_unused_parameters=False,                       # 核心修复 2：DDP 显式关闭未用参数查找，杜绝 flow control 报错与额外显存缓冲
        learning_rate=1.5e-4,        # 4B 模型 LoRA 调优为 1.5e-4，平滑梯度收敛
        fp16=True,                   # T4(Turing) 不支持 bf16，继续用 fp16 + GradScaler
        bf16=False,                  # 显式关闭：与上面的 fp16 互斥，避免被环境/版本差异悄悄打开
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=10,
        max_grad_norm=1.0,
        num_train_epochs=2,
        logging_steps=10,            # 每 10 步往 TensorBoard 写一次
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
    """兼容 peft 与 transformers>=5.16 的 ImportError（张量并行分片）。

    transformers 5.16 起把 `transformers.integrations.tensor_parallel` 改成指向
    `transformers.distributed.tensor_parallel` 的兼容 shim，且不再重导出 `EmbeddingParallel`。
    而 peft 的 `_maybe_shard_state_dict_for_tp` 在函数开头**无条件** import 该类，于是即使
    完全不使用张量并行（本训练是单机 DDP），`_load_best_model -> load_adapter ->
    set_peft_model_state_dict` 也会直接 ImportError（peft issue #3628，官方暂无补丁）。

    本训练模型不含 `_hf_tp_plan`（非 TP），TP 分片本就无事可做：检测到该 ImportError 时
    对非 TP 模型直接跳过；真 TP 模型仍原样抛错，避免调换语义后悄悄存出错误的权重。
    """
    try:
        from peft.utils import save_and_load
    except Exception:  # peft 结构变化时静默跳过，不阻断训练
        return

    original = getattr(save_and_load, "_maybe_shard_state_dict_for_tp", None)
    if original is None:  # 旧版 peft 无此函数，或已被上游修复删除，无需兼容
        return

    def has_tp_plan(model) -> bool:
        return any(getattr(module, "_hf_tp_plan", None) is not None for module in model.modules())

    def safe_maybe_shard_state_dict_for_tp(model, state_dict, adapter_name):
        try:
            return original(model, state_dict, adapter_name)
        except ImportError as exc:
            if has_tp_plan(model):
                raise
            print(f"[兼容] 跳过 peft 张量并行分片（当前模型非 TP）：{exc}")
            return None

    save_and_load._maybe_shard_state_dict_for_tp = safe_maybe_shard_state_dict_for_tp


def main():
    patch_peft_tensor_parallel_compat()

    # 只在主进程打印日志
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0

    data_dict = process()
    model_name = getattr(config, "BASE_MODEL_NAME", "Qwen/Qwen3-8B")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if is_main_process:
        print(f"[并行] world_size={os.environ.get('WORLD_SIZE', 1)} "
              f"可见 GPU 数={torch.cuda.device_count()}")
        print(f"训练集 {len(data_dict['train'])} 条 / 验证集 {len(data_dict['val'])} 条")

    # tokenize + assistant 段 mask。放在模型加载之前，早失败早收工。
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
    if is_main_process:
        check_supervision(train_dataset)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank} if torch.cuda.is_available() else None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # dtype 必须写对：名字写错会被 from_pretrained 当无关 kwargs 丢掉且不报错，
    # 模型就会按 checkpoint 自带的 bf16 加载，随后 GradScaler 在首次梯度裁剪时抛
    # `_amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'`。
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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

    # 数据已经 tokenize 好并打好 labels（skip_prepare_dataset=True），collator 只负责 padding
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100,
                                      pad_to_multiple_of=8)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class = tokenizer,
    )

    # 关键修复：peft 会把 adapter 权重对齐成基座的 dtype，而 Qwen3 的 checkpoint 本身是 bf16，
    # 于是 lora_A / lora_B 也变成 bf16 —— 这一点不受 from_pretrained 的 dtype、也不受
    # bnb_4bit_compute_dtype 控制。而 fp16 AMP 的 GradScaler 只接受 fp32 梯度：bf16 会抛
    # `_amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'`，
    # fp16 会抛 `Attempting to unscale FP16 gradients`，两者都必崩。
    # 因此在 optimizer 建好之前（trainer.train() 内部才建）显式归一成 fp32，
    # 做法见 peft 官方 troubleshooting；只动可训练参数，4-bit 基座原样不动。
    for param in trainer.model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    # 打完桩再自检一次：可训练参数 dtype 必须只有 torch.float32。
    if is_main_process:
        check_precision(trainer.model)

    trainer.train()

    trainer.save_model(str(config.MODEL_PATH))
    if is_main_process:
        tokenizer.save_pretrained(str(config.MODEL_PATH))
        print(f"LoRA 权重与 Tokenizer 已保存至: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
