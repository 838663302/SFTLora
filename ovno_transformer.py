import os
from pathlib import Path
os.environ["HF_HUB_DISABLE_XET"] = "1"
from optimum.intel.openvino import OVModelForCausalLM, OVWeightQuantizationConfig
from transformers import AutoTokenizer

# 模型源：
# 1. 刚下载的 Qwen3-8B 官方基座模型快照
QWEN3_8B_DIR = Path(r"C:\Users\azt1szh\.cache\huggingface\hub\models--Qwen--Qwen3-8B\snapshots\b968826d9c46dd6066d109eabc6255188de91218")
# 2. 微调后合并的完整模型目录（备用）
MERGED_DIR = Path(r"C:\Users\azt1szh\Desktop\set\SFTLora\checkpoints\merged")

# 使用微调后合并的 4B 模型作为输入源
input_dir = MERGED_DIR
output_dir = Path(r"C:\Users\azt1szh\Desktop\set\checkpoints\ov_model")

print("=" * 60)
print(f"正在将模型转换为 OpenVINO IR 格式（开启 INT4 权重压缩）...")
print(f"输入源: {input_dir}")
print(f"输出目录: {output_dir}")
print("=" * 60)

# 1. 4-bit 权重对称量化配置（大幅降低显存/内存占用，加速 Intel GPU/CPU 推理）
quantization_config = OVWeightQuantizationConfig(
    bits=4,
    sym=True,
    group_size=128,
    ratio=0.8,
)

# 2. 导出模型（export=True 会自动转为 4-bit openvino_model.xml 和 openvino_model.bin）
model = OVModelForCausalLM.from_pretrained(
    input_dir,
    export=True,
    compile=False,
    quantization_config=quantization_config,
)
model.save_pretrained(output_dir)

# 2. 保存分词器
tokenizer = AutoTokenizer.from_pretrained(input_dir)
tokenizer.save_pretrained(output_dir)

print(f"\n转换完成！模型已保存在: {output_dir}")

# 3. 快速推理验证测试（使用 Intel Arc 核显 GPU）
print("\n正在验证加载 OpenVINO 模型并使用 Intel Arc GPU 进行生成测试...")
ov_test_model = OVModelForCausalLM.from_pretrained(
    output_dir,
    device="GPU",
    ov_config={
        "PERFORMANCE_HINT": "LATENCY",
        "CACHE_DIR": str(output_dir / "ov_cache"),
    },
)
test_prompt = "你好，请介绍一下你自己。"
inputs = tokenizer(test_prompt, return_tensors="pt")
outputs = ov_test_model.generate(**inputs, max_new_tokens=30)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"\n[验证测试输出]")
print(f"输入: {test_prompt}")
print(f"生成: {response}")
print("\nOpenVINO 模型验证通过！")
