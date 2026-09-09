import torch
import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from process import process
from peft import LoraConfig

def main():
    data_dict = process()
    model_name = "Qwen/Qwen3-1.7B"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sft_config = SFTConfig(
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
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

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
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(str(config.MODEL_PATH))
    tokenizer.save_pretrained(str(config.MODEL_PATH))
    print(f"LoRA 权重与 Tokenizer 已保存至: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()


