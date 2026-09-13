import os
from pathlib import Path
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

# 模型源：微调后合并的完整模型目录
MERGED_DIR = Path(r"C:\Users\azt1szh\Desktop\set\checkpoints\merged")
HF_CACHE_SNAPSHOT = Path(r"C:\Users\azt1szh\.cache\huggingface\hub\models--Qwen--Qwen3-1.7B\snapshots\70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")

# 使用微调后合并的完整模型目录作为输入源
input_dir = MERGED_DIR
output_dir = Path(r"C:\Users\azt1szh\Desktop\set\checkpoints\ov_model")

print("=" * 60)
print(f"正在将模型转换为 OpenVINO IR 格式...")
print(f"输入源: {input_dir}")
print(f"输出目录: {output_dir}")
print("=" * 60)

# 1. 导出模型（export=True 会自动转为 openvino_model.xml 和 openvino_model.bin）
model = OVModelForCausalLM.from_pretrained(
    input_dir,
    export=True,
    compile=False
)
model.save_pretrained(output_dir)

# 2. 保存分词器
tokenizer = AutoTokenizer.from_pretrained(input_dir)
tokenizer.save_pretrained(output_dir)

print(f"\n转换完成！模型已保存在: {output_dir}")

# 3. 快速推理验证测试
print("\n正在验证加载 OpenVINO 模型并进行生成测试...")
ov_test_model = OVModelForCausalLM.from_pretrained(output_dir, device="CPU")
test_prompt = "你好，请介绍一下你自己。"
inputs = tokenizer(test_prompt, return_tensors="pt")
outputs = ov_test_model.generate(**inputs, max_new_tokens=30)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"\n[验证测试输出]")
print(f"输入: {test_prompt}")
print(f"生成: {response}")
print("\nOpenVINO 模型验证通过！")
