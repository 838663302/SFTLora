from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer
from pathlib import Path
input_dir = Path(r"C:\Users\azt1szh\Desktop\set\checkpoints\merged")
output_dir = Path(r"C:\Users\azt1szh\Desktop\set\checkpoints\ov_model")

print("正在转换为 OpenVINO IR 格式...")
# 1. 导出模型（export=True 会自动转为 openvino_model.xml 和 openvino_model.bin）
model = OVModelForCausalLM.from_pretrained(input_dir, export=True)
model.save_pretrained(output_dir)

# 2. 保存分词器
tokenizer = AutoTokenizer.from_pretrained(input_dir)
tokenizer.save_pretrained(output_dir)

print(f"转换完成！模型已保存在: {output_dir}")
