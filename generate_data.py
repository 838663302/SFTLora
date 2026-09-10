import json
import random
import uuid
from pathlib import Path

# 1. 来自 config.txt 的 14 条明确业务规则
RULES = [
    # CMP 工作区
    {"workspace": "CMP", "project": "CPT", "env": "dev", "code": "162-d-cpt", "mta": "mta-quality.yaml"},
    {"workspace": "CMP", "project": "PT", "env": "dev", "code": "162-d-pt", "mta": "mta-prequality-pt.yaml"},
    {"workspace": "CMP", "project": "HC", "env": "dev", "code": "162-d-hc", "mta": "mta-prequality-hc.yaml"},
    {"workspace": "CMP", "project": "HCS1T", "env": "dev", "code": "162-d-hc", "mta": "mta-prequality-hc-s1t.yaml"},
    {"workspace": "CMP", "project": "CPT-PT", "env": "dev", "code": "162-d-pt", "mta": "mta-prequality-cpt-pt.yaml"},
    {"workspace": "CMP", "project": "CPT-PT", "env": "quality", "code": "162-q-pt", "mta": "mta-quality-cpt-pt.yaml"},
    {"workspace": "CMP", "project": "CPT-HC", "env": "dev", "code": "162-d-hc", "mta": "mta-prequality-cpt-hc.yaml"},
    {"workspace": "CMP", "project": "CPT-HC", "env": "quality", "code": "162-q-hc", "mta": "mta-quality-cpt-hc.yaml"},
    {"workspace": "CMP", "project": "CPT", "env": "quality", "code": "162-q-cpt", "mta": "mta-quality.yaml"},
    {"workspace": "CMP", "project": "PT", "env": "quality", "code": "162-q-pt", "mta": "mta-quality-pt.yaml"},
    {"workspace": "CMP", "project": "HC", "env": "quality", "code": "162-q-hc", "mta": "mta-quality-hc.yaml"},
    # MPB 工作区
    {"workspace": "MPB", "project": "HC", "env": "dev", "code": "163-d-hc", "mta": "mta-develop-hc.yaml"},
    {"workspace": "MPB", "project": "CPT", "env": "dev", "code": "162-d-cpt", "mta": "mta-develop.yaml"},
    {"workspace": "MPB", "project": "PT", "env": "dev", "code": "162-d-pt", "mta": "mta-develop-pt.yaml"},
]

# 2. 常见 Agent 风格的命令行工具池（支持泛化）
COMMAND_TOOL_VARIANTS = [
    {
        "name": "execute_command",
        "param": "command",
        "schema": {
            "type": "function",
            "function": {
                "name": "execute_command",
                "description": "执行系统命令行命令或终端命令",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "待执行的完整命令行字符串"}
                    },
                    "required": ["command"]
                }
            }
        }
    },
    {
        "name": "run_command",
        "param": "command",
        "schema": {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "在终端中运行指定的 shell 命令",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "要运行的 shell 命令"}
                    },
                    "required": ["command"]
                }
            }
        }
    },
    {
        "name": "bash",
        "param": "cmd",
        "schema": {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a command in the bash terminal",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string", "description": "The command line to execute"}
                    },
                    "required": ["cmd"]
                }
            }
        }
    },
    {
        "name": "terminal",
        "param": "command",
        "schema": {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "系统终端工具，用于执行命令行指令",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "要执行的命令行指令"}
                    },
                    "required": ["command"]
                }
            }
        }
    },
    {
        "name": "run_terminal_cmd",
        "param": "cmd",
        "schema": {
            "type": "function",
            "function": {
                "name": "run_terminal_cmd",
                "description": "执行终端控制台命令",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string", "description": "终端命令"}
                    },
                    "required": ["cmd"]
                }
            }
        }
    },
    {
        "name": "powershell",
        "param": "command",
        "schema": {
            "type": "function",
            "function": {
                "name": "powershell",
                "description": "Execute a PowerShell command or script",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "PowerShell command to execute"}
                    },
                    "required": ["command"]
                }
            }
        }
    }
]

# 3. 干扰工具
DISTRACTOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定路径的文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件绝对路径"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入或覆盖指定文件的内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "目标文件路径"},
                    "content": {"type": "string", "description": "文件内容"}
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出指定目录下的所有文件和文件夹",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory_path": {"type": "string", "description": "目录路径"}
                },
                "required": ["directory_path"]
            }
        }
    }
]

ENV_SYNONYMS = {
    "dev": ["开发环境", "开发", "dev", "dev环境", "d环境"],
    "quality": ["quality环境", "quality", "qa", "qa环境", "测试环境", "q环境"]
}

# 4. Action（工具调用类）提示词模版
ACTION_TEMPLATES = [
    "帮我准备 {ws} 工作区下 {proj} 项目的 {env}",
    "请登录 {ws} 的 {proj} {env}，并给我 mta 文件",
    "登录 {ws} {proj} {env}",
    "在 {ws} 工作区下部署 {proj} 的 {env}，帮我调登录接口并告知 mta 文件",
    "需要登录 {proj} 的 {env}，工作区是 {ws}",
    "准备发布 {ws} 下面的 {proj} 项目 {env}，执行登录",
    "帮我调一下本地登录 API：工作区 {ws}，项目 {proj}，环境 {env}",
    "查一下 {ws} - {proj} - {env} 的配置并登录",
    "请协助登录 {ws} 的 {proj} 项目（{env}），告知对应的 mta 部署文件",
    "麻烦登录 {ws} 工作区的 {proj} {env}",
    "{ws} 工作区，{proj} 项目，{env}，帮我登录并输出 mta 文件",
    "执行 {ws} 中 {proj} 的 {env} 登录",
    "准备在 {ws} 部署 {proj} {env}，调用登录接口"
]

SYSTEM_PROMPT = "你是一个专业的开发运维助手。根据用户提供的工作区、项目和环境信息，输出对应的登录 code 和 MTA 部署文件；当用户需要执行登录时，使用可用的命令行工具调用本地 API 执行登录；当用户仅查询知识库信息时，直接回答相应信息，无需调用工具。"


def get_random_tools():
    """获取包含 1 个命令行工具和 0~2 个干扰工具的随机工具列表"""
    cmd_tool = random.choice(COMMAND_TOOL_VARIANTS)
    tools_list = [cmd_tool["schema"]]
    distractors = random.sample(DISTRACTOR_TOOLS, k=random.choice([0, 1, 2]))
    tools_list.extend(distractors)
    random.shuffle(tools_list)
    return cmd_tool["name"], cmd_tool["param"], tools_list


def build_multiturn_action_samples():
    """生成原生多轮 Agent 工具交互样本（包含工具调用与结果回传后的自主总结）"""
    samples = []
    for rule in RULES:
        ws = rule["workspace"]
        proj = rule["project"]
        code = rule["code"]
        mta = rule["mta"]
        env_names = ENV_SYNONYMS[rule["env"]]
        env_disp = "开发环境" if rule["env"] == "dev" else "quality环境"

        for tmpl in ACTION_TEMPLATES:
            env_str = random.choice(env_names)
            user_text = tmpl.format(ws=ws, proj=proj, env=env_str)
            tool_name, param_name, tools_list = get_random_tools()
            call_id = f"call_{uuid.uuid4().hex[:8]}"

            curl_cmd = f"curl -X POST http://localhost:3000/space/login -H \"Content-Type: application/json\" -d '{{\"space\": \"{code}\"}}'"
            tool_args = {param_name: curl_cmd}

            # 1. 成功回传分支 (Success Multi-turn)
            tool_res_success = random.choice([
                json.dumps({"status": "success", "code": 200, "message": f"Successfully logged into space {code}"}, ensure_ascii=False),
                json.dumps({"success": True, "space": code, "output": "Login successful"}, ensure_ascii=False),
                f"HTTP/1.1 200 OK\nContent-Type: application/json\n\n{{\"status\": 200, \"space\": \"{code}\"}}"
            ])

            assistant_final_success = random.choice([
                f"✅ 登录接口已成功调用！\n- **工作区**：{ws}\n- **项目**：{proj}（{env_disp}）\n- **登录 code**：`{code}`\n- **部署 MTA 文件**：`{mta}`\n开发环境已就绪，可以开始后续开发部署。",
                f"已成功为您完成登录操作：\n1. **项目与环境**：{ws} 工作区 / {proj} 项目（{env_disp}）\n2. **登录 code**：`{code}`\n3. **MTA 配置文件**：`{mta}`\n接口调用状态正常，环境已准备就绪。",
                f"登录操作完成！\n- 登录 code: `{code}`\n- MTA 部署文件: `{mta}`\n本地 API 返回登录成功。"
            ])

            multi_turn_sample = {
                "tools": tools_list,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_args, ensure_ascii=False)
                                }
                            }
                        ]
                    },
                    {
                        "role": "tool",
                        "name": tool_name,
                        "tool_call_id": call_id,
                        "content": tool_res_success
                    },
                    {
                        "role": "assistant",
                        "content": assistant_final_success
                    }
                ]
            }
            samples.append(multi_turn_sample)

            # 2. 失败异常回传分支 (Error Multi-turn) - 少量样本提升健壮性
            if random.random() < 0.2:
                tool_res_error = json.dumps({"status": 500, "error": "Connection refused to http://localhost:3000/space/login"}, ensure_ascii=False)
                assistant_final_error = f"❌ 调用登录接口失败（连接本地 3000 端口被拒绝）。\n- 目标登录 code：`{code}`\n- 部署 MTA 文件：`{mta}`\n请检查本地登录服务是否正常启动后重试。"

                error_sample = {
                    "tools": tools_list,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps(tool_args, ensure_ascii=False)
                                    }
                                }
                            ]
                        },
                        {
                            "role": "tool",
                            "name": tool_name,
                            "tool_call_id": call_id,
                            "content": tool_res_error
                        },
                        {
                            "role": "assistant",
                            "content": assistant_final_error
                        }
                    ]
                }
                samples.append(error_sample)

            # 3. 单轮直接 Action 分支 (Single-turn Action) - 兼容单轮客户端
            single_turn_sample = {
                "tools": tools_list,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                    {
                        "role": "assistant",
                        "content": f"登录 code: {code}\nMTA 文件: {mta}",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_args, ensure_ascii=False)
                                }
                            }
                        ]
                    }
                ]
            }
            samples.append(single_turn_sample)

    return samples


def build_qa_samples():
    """生成纯知识库问答样本（不触发 tool_call，即便上下文里有 tools）"""
    samples = []

    # A. 正向单点查询：只查 code 或 只查 mta
    code_templates = [
        "{ws} 工作区下 {proj} 项目的 {env} 登录 code 是多少？",
        "查一下 {ws} 的 {proj} {env} 对应的登录 code",
        "{ws} {proj} {env} 的登陆 code 是什么？",
        "请问在 {ws} 里的 {proj}（{env}）登录 code 是什么"
    ]
    mta_templates = [
        "{ws} 工作区下 {proj} 项目的 {env} 部署 mta 文件是什么？",
        "请问 {ws} 的 {proj} {env} 用的 mta 文件叫什么？",
        "{ws} {proj} {env} 的 mta 部署配置文件名？",
        "{ws} 工作区中 {proj} 的 {env} 对应哪个 mta yaml？"
    ]

    for rule in RULES:
        ws = rule["workspace"]
        proj = rule["project"]
        code = rule["code"]
        mta = rule["mta"]
        env_raw = rule["env"]
        env_names = ENV_SYNONYMS[env_raw]

        # 查 code
        for tmpl in code_templates:
            env_str = random.choice(env_names)
            _, _, tools_list = get_random_tools()
            samples.append({
                "tools": tools_list,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": tmpl.format(ws=ws, proj=proj, env=env_str)},
                    {"role": "assistant", "content": f"{ws} 工作区下 {proj} 项目 {env_names[0]}的登录 code 是 `{code}`。"}
                ]
            })

        # 查 mta
        for tmpl in mta_templates:
            env_str = random.choice(env_names)
            _, _, tools_list = get_random_tools()
            samples.append({
                "tools": tools_list,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": tmpl.format(ws=ws, proj=proj, env=env_str)},
                    {"role": "assistant", "content": f"{ws} 工作区下 {proj} 项目 {env_names[0]}的部署 MTA 文件是 `{mta}`。"}
                ]
            })

    # B. 反向查询：根据 code 查归属
    code_to_rules = {}
    for r in RULES:
        code_to_rules.setdefault(r["code"], []).append(r)

    for code, matched in code_to_rules.items():
        _, _, tools_list = get_random_tools()
        user_queries = [
            f"登录 code `{code}` 对应的是哪个项目和环境？",
            f"code 是 `{code}` 的环境有哪些？",
            f"谁使用登录 code `{code}`？"
        ]
        info_lines = []
        for r in matched:
            env_disp = "开发环境" if r["env"] == "dev" else "quality环境"
            info_lines.append(f"- **{r['workspace']} 工作区** 的 **{r['project']} 项目**（{env_disp}，MTA: `{r['mta']}`）")
        resp = f"登录 code `{code}` 对应的项目与环境如下：\n" + "\n".join(info_lines)

        for uq in user_queries:
            samples.append({
                "tools": tools_list,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": uq},
                    {"role": "assistant", "content": resp}
                ]
            })

    # C. 跨工作区对比（HC, CPT, PT 在 CMP 与 MPB 下的区别）
    comparisons = [
        (
            "HC",
            "CMP 和 MPB 下的 HC 开发环境有什么区别？",
            "两者的区别如下：\n- **CMP 工作区**：HC 开发环境的登录 code 是 `162-d-hc`，部署 MTA 文件是 `mta-prequality-hc.yaml`。\n- **MPB 工作区**：HC 开发环境的登录 code 是 `163-d-hc`，部署 MTA 文件是 `mta-develop-hc.yaml`。"
        ),
        (
            "CPT",
            "CPT 项目在 CMP 和 MPB 的开发环境部署 MTA 文件一样吗？",
            "不相同，两者的 MTA 部署文件不同：\n- **CMP 工作区**：CPT 开发环境的 MTA 文件是 `mta-quality.yaml`（登录 code 是 `162-d-cpt`）。\n- **MPB 工作区**：CPT 开发环境的 MTA 文件是 `mta-develop.yaml`（登录 code 同样是 `162-d-cpt`）。"
        ),
        (
            "PT",
            "对比一下 CMP 和 MPB 下 PT 开发环境的配置",
            "配置对比：\n- **CMP 工作区**：PT 开发环境登录 code 为 `162-d-pt`，MTA 文件为 `mta-prequality-pt.yaml`。\n- **MPB 工作区**：PT 开发环境登录 code 为 `162-d-pt`，MTA 文件为 `mta-develop-pt.yaml`。"
        )
    ]
    for proj_name, q, a in comparisons:
        for _ in range(3):
            _, _, tools_list = get_random_tools()
            samples.append({
                "tools": tools_list,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })

    # D. 全局列表查询
    _, _, tools_list = get_random_tools()
    samples.append({
        "tools": tools_list,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "列出 MPB 工作区下的所有项目和环境配置"},
            {
                "role": "assistant",
                "content": (
                    "MPB 工作区包含以下 3 个项目的开发环境配置：\n"
                    "1. **HC 项目**：登录 code 为 `163-d-hc`，MTA 文件为 `mta-develop-hc.yaml`\n"
                    "2. **CPT 项目**：登录 code 为 `162-d-cpt`，MTA 文件为 `mta-develop.yaml`\n"
                    "3. **PT 项目**：登录 code 为 `162-d-pt`，MTA 文件为 `mta-develop-pt.yaml`"
                )
            }
        ]
    })
    samples.append({
        "tools": tools_list,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "请列出 CMP 工作区下所有的项目和环境清单"},
            {
                "role": "assistant",
                "content": (
                    "CMP 工作区下包含的项目与环境如下：\n"
                    "- **CPT**：开发环境（code: `162-d-cpt`，mta: `mta-quality.yaml`）；quality环境（code: `162-q-cpt`，mta: `mta-quality.yaml`）\n"
                    "- **PT**：开发环境（code: `162-d-pt`，mta: `mta-prequality-pt.yaml`）；quality环境（code: `162-q-pt`，mta: `mta-quality-pt.yaml`）\n"
                    "- **HC**：开发环境（code: `162-d-hc`，mta: `mta-prequality-hc.yaml`）；quality环境（code: `162-q-hc`，mta: `mta-quality-hc.yaml`）\n"
                    "- **HCS1T**：开发环境（code: `162-d-hc`，mta: `mta-prequality-hc-s1t.yaml`）\n"
                    "- **CPT-PT**：开发环境（code: `162-d-pt`，mta: `mta-prequality-cpt-pt.yaml`）；quality环境（code: `162-q-pt`，mta: `mta-quality-cpt-pt.yaml`）\n"
                    "- **CPT-HC**：开发环境（code: `162-d-hc`，mta: `mta-prequality-cpt-hc.yaml`）；quality环境（code: `162-q-hc`，mta: `mta-quality-cpt-hc.yaml`）"
                )
            }
        ]
    })

    return samples


def generate_dataset():
    random.seed(42)

    action_samples = build_multiturn_action_samples()
    qa_samples = build_qa_samples()

    print(f"原始生成: Action 样本 {len(action_samples)} 条, QA 样本 {len(qa_samples)} 条")

    random.shuffle(action_samples)
    random.shuffle(qa_samples)

    # 抽取高质量混合数据 (约 350 条样本)
    selected_actions = action_samples[:220]
    selected_qa = qa_samples[:130]

    all_samples = selected_actions + selected_qa
    random.shuffle(all_samples)

    # 按照 85% : 15% 划分训练集和验证集
    train_count = int(len(all_samples) * 0.85)
    train_data = all_samples[:train_count]
    val_data = all_samples[train_count:]

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("原生 Multi-Turn Agent 复合数据集生成完成！")
    print(f"总生成样本数: {len(all_samples)} 条")
    print(f"  - Action 多轮与行动样本: {len(selected_actions)} 条")
    print(f"  - 知识库问答 QA 样本: {len(selected_qa)} 条")
    print(f"划分结果:")
    print(f"  - 训练集 (train.jsonl): {len(train_data)} 条 -> {train_file}")
    print(f"  - 验证集 (val.jsonl): {len(val_data)} 条 -> {val_file}")
    print("=" * 60)

if __name__ == "__main__":
    generate_dataset()

