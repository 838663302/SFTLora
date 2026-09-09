import os

from typing import Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config


def infer(
    prompt: str,
    system_prompt: Optional[str] = None,
    tools: Optional[list] = None,
) -> Tuple[str, str]:
    if system_prompt is None:
        system_prompt = (
            "你是一个专业的开发运维助手。根据用户提供的工作区、项目和环境信息，"
            "输出对应的登录 code 和 MTA 部署文件；当用户需要执行登录时，使用可用的命令行工具调用本地 API 执行登录；"
            "当用户仅查询知识库信息时，直接回答相应信息，无需调用工具。"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        str(config.MERGED_PATH),
        dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(config.MERGED_PATH))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    chat_kwargs = {
        "messages": messages,
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if tools:
        chat_kwargs["tools"] = tools

    text = tokenizer.apply_chat_template(**chat_kwargs)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32768,
        )

    generated_ids = outputs[0][len(inputs["input_ids"][0]) :].tolist()
    try:
        index = len(generated_ids) - generated_ids[::-1].index(151668)
    except ValueError:
        index = 0
    think = tokenizer.decode(generated_ids[:index], skip_special_tokens = True)

    response = tokenizer.decode(generated_ids[index:], skip_special_tokens=False)
    return think, response


def main():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "系统终端工具，用于执行命令行指令",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "要执行的命令行指令",
                        }
                    },
                    "required": ["command"],
                },
            },
        }
    ]

    prompt = "告诉我 CMP 工作区下 CPT-HC 项目的 d环境的登陆code和mta文件是什么"
    print(f"User Query: {prompt}\n")

    think, response = infer(prompt, tools=tools)
    print("===== 模型回复 =====")
    print(f"think {think}")
    print(f"response {response}")


if __name__ == "__main__":
    main()
