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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from process import process
from peft import LoraConfig, prepare_model_for_kbit_training

# 单条样本的最大 token 数。
# 实测数据集中全量最长样本仅 3390 token（均值约 2082 token）。
# 设置为 3840 即可留足约 450 token 安全余量，避免盲目设为 8192 浪费多余的注意力和位置张量显存。
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "3840"))


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

    # 只在主进程做长度自检与日志打印，否则两个 rank 会把 809 条样本各 tokenize 一遍
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0

    data_dict = process()
    model_name = getattr(config, "BASE_MODEL_NAME", "Qwen/Qwen3-4B")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if is_main_process:
        print(f"[并行] world_size={os.environ.get('WORLD_SIZE', 1)} "
              f"可见 GPU 数={torch.cuda.device_count()}")
        print(f"训练集 {len(data_dict['train'])} 条 / 验证集 {len(data_dict['val'])} 条")
        check_max_length(data_dict["train"], tokenizer, MAX_LENGTH)

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

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        peft_config=peft_config,
        train_dataset=data_dict["train"],
        eval_dataset=data_dict["val"],
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

    # trainer.save_model 内部已按 args.should_save（仅 process_index==0）守住写盘，多进程下只会存一份；
    # 但 tokenizer.save_pretrained 属于 PreTrainedTokenizerBase，完全没有分布式守卫，
    # 两个进程会并发写同一个 tokenizer.json（十几 MB，可能写出半截导致损坏），必须自己挡。
    trainer.save_model(str(config.MODEL_PATH))
    if is_main_process:
        tokenizer.save_pretrained(str(config.MODEL_PATH))
        print(f"LoRA 权重与 Tokenizer 已保存至: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
