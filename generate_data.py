import json
import os
import random
import re
import uuid
from collections import defaultdict
from pathlib import Path

# ============================================================
# 0. 全局开关
# ============================================================

SEED = 42

# ============================================================
# 工具层设计原则（重要）
# ============================================================
# 推理时工具定义是 agent 自己注入 prompt 的，模型不需要"背"schema。
# 但模型必须学会「照着 prompt 里给的那套 schema 发出调用」，这是必须训的能力。
# 所以这里的原则是：
#
#   固定（invariant）  : 任务本身 + 每步要执行的 shell 命令 + 业务映射（code/mta/路径）
#   随机（augmentation）: 调用外壳 —— 工具名、参数名、菜单里有几个工具、system prompt、语言
#   覆盖（coverage）   : agent 菜单里声明的每一条工具，都必须有「真实发出调用」的示范样本
#                        （只给 schema 不给示范 = 让模型猜参数怎么填，见 MIN_TOOL_COVERAGE）
#
# 权重向 CodeBuddy 倾斜（当前主用 harness），其余档位提供泛化能力，
# 这样换到别的 agent 时模型能靠 prompt 里的 schema 自适应，而不是死记一个工具名。
#
# 证据来源：
#   C:\CodeBuddy CN\resources\app\extensions\genie\out\extension\index.js
#     → METADATA_EXECUTE_COMMAND_TOOL / getToolRequiredParams()
#       CodeBuddy: execute_command, required = ["command", "requires_approval"]，无 timeout
#   ~/.codebuddy/agents/btp-deploy.md
#     → agent 暴露 9 条工具（去掉没有 schema 可查的 read_rules）

FAIL_BRANCHES_PER_TRAJECTORY = 3

# agent 提示词（工具菜单）里声明的每条工具，至少要在这条数以上的样本里被真正调用过。
# 只给 schema 不给示范，模型只能猜参数怎么填，到了真实环境就会拿 shell 命令硬凑。
MIN_TOOL_COVERAGE = 5

# 关键命令至少要在这么多样本里当「监督目标」（= 最后一条 assistant，即要求模型生成它的位置）。
# 实测教训：旧数据里 mbt build 出现在 159 处，但真正要求模型生成的只有 23 条，
# 结果动作序列学会了、命令字面量没学会（线上表现为瞎试 -l/-f/-d、登录 code 与路由幻觉）。
MIN_KEY_COMMAND_TARGET = 50

# (标签, 命令里必须出现的片段)，供「命令目标曝光」自检使用
KEY_COMMANDS = [
    ("空间登录", "space/login"),
    ("MTA 复制", "Copy-Item"),
    ("MTA 构建", "mbt build"),
    ("CF 部署", "cf deploy"),
    ("环境清理", "Remove-Item"),
]
# 采样重复：规则少了之后（14 -> 3），靠重复把「登录 code / mta 文件」映射的曝光量补回来。
# 基础每条规则 2 轮；歧义规则（code 历史上跨工作区复用）3 轮。
BASE_REPEAT = 2
AMBIGUOUS_REPEAT = 3

# 单步动作族（构建 / 部署 / 清理等）的重复轮数。
# 这一族是「关键命令当答案」的主力：动作列表复制 STEP_REPEAT 遍，命令作为监督目标的次数直接 ×N。
STEP_REPEAT = 4

rng = random.Random(SEED)


# ============================================================
# 1. 来自 config.txt 的明确业务规则（仅 MPB 工作区；CMP 相关配置已按需求移除）
# ============================================================

RULES = [
    # MPB 工作区
    {"workspace": "MPB", "project": "HC", "env": "dev", "code": "163-d-hc", "mta": "mta-develop-hc.yaml"},
    {"workspace": "MPB", "project": "CPT", "env": "dev", "code": "162-d-cpt", "mta": "mta-develop.yaml"},
    {"workspace": "MPB", "project": "PT", "env": "dev", "code": "162-d-pt", "mta": "mta-develop-pt.yaml"},
]

# 需要过采样的规则（CMP 工作区移除后歧义已大幅减少；
# MPB 这两条保留过采样：162-d-cpt / 162-d-pt 在历史上也出现在其它工作区，容易混）
AMBIGUOUS_KEYS = {
    ("MPB", "CPT", "dev"),
    ("MPB", "HC", "dev"),
}


def rule_key(rule):
    return f"{rule['workspace']}|{rule['project']}|{rule['env']}"


# ============================================================
# 2. 工具定义（严格对齐 CodeBuddy 的真实签名）
# ============================================================

def _tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


_EXPLANATION_PROP = {
    "type": "string",
    "description": "Optional. One sentence explanation of why this tool is used, in the user's language.",
}

# 描述做了精简：官方原文里 execute_command 单条就 1300+ 字符，9 条工具全量塞进去
# 光是工具菜单就要 8000+ token，每条样本会撑到 10000 token 以上。
# 这里保留了 100% 准确的工具名、参数名与 required 列表（这三样决定调用能否被解析），
# 只压缩了自然语言描述。想要 100% 原文保真，从
# C:\CodeBuddy CN\resources\app\extensions\genie\out\extension\index.js 里把
# METADATA_*_TOOL 的 description 贴回来，同时把 main.py 的 MAX_LENGTH 提到 16384。
CODEBUDDY_TOOL_MENU = [
    _tool(
        "list_dir",
        "Lists files and directories in a given path. target_directory must be returned before other fields.",
        {
            "target_directory": {"type": "string", "description": "Path to directory to list contents of."},
            "ignore_globs": {"type": "array", "description": "Optional glob patterns to ignore."},
        },
        ["target_directory"],
    ),
    _tool(
        "search_file",
        "File search with wildcard pattern matching and recursive directory search. Returns relative paths.",
        {
            "target_directory": {"type": "string", "description": "Absolute path to the directory to search in."},
            "pattern": {"type": "string", "description": 'REQUIRED: File pattern (e.g., "*.js"). Supports wildcards.'},
            "recursive": {"type": "boolean", "description": "REQUIRED: Set to true to search in subdirectories."},
            "ignore_globs": {"type": "array", "description": "Optional glob patterns to ignore."},
        },
        ["pattern", "recursive"],
    ),
    _tool(
        "search_content",
        "Search file contents with regular expressions (built on ripgrep). Prefer this over terminal grep/rg.",
        {
            "pattern": {"type": "string", "description": "REQUIRED: The regular expression pattern to search for."},
            "path": {"type": "string", "description": "File or directory to search in. Must be an absolute path."},
            "glob": {"type": "string", "description": 'Glob pattern to filter files (e.g. "*.js").'},
        },
        ["pattern"],
    ),
    _tool(
        "read_file",
        "Reads a file from the local filesystem. filePath must be an absolute path. "
        "filePath must be returned before other fields.",
        {
            "filePath": {"type": "string",
                         "description": "REQUIRED: The absolute path of the file to read, NOT a directory."},
            "offset": {"type": "number", "description": "The line number to start reading from."},
            "limit": {"type": "number", "description": "The number of lines to read."},
        },
        ["filePath"],
    ),
    _tool(
        "read_lints",
        "Read and display linter errors from the current workspace.",
        {
            "paths": {"type": "array", "description": "Optional. File or directory paths to read diagnostics for."},
            "severity": {"type": "array", "description": 'Optional. Filter by severity, e.g. ["error", "warning"].'},
        },
        [],
    ),
    _tool(
        "replace_in_file",
        "Performs exact string replacements in an existing file. "
        "REQUIRED PARAMETERS - filePath, old_str and new_str are ALL MANDATORY. "
        "To create or overwrite a file, prefer write_to_file.",
        {
            "filePath": {"type": "string", "description": "REQUIRED: The absolute path of the file to edit."},
            "old_str": {"type": "string", "description": "REQUIRED: The exact text to replace."},
            "new_str": {"type": "string", "description": "REQUIRED: The replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences of old_str."},
        },
        ["filePath", "old_str", "new_str"],
    ),
    _tool(
        "write_to_file",
        "Writes a file to the local filesystem, overwriting the existing file if there is one. "
        "filePath MUST be an absolute path and must be returned before content.",
        {
            "filePath": {"type": "string", "description": "REQUIRED: Target file absolute path."},
            "content": {"type": "string",
                        "description": "REQUIRED: The complete content to write. Always provide the full file content."},
            "explanation": _EXPLANATION_PROP,
        },
        ["filePath", "content"],
    ),
    _tool(
        "execute_command",
        "PROPOSE a command to run on behalf of the user.\n"
        "If you have this tool, note that you DO have the ability to run commands directly on the USER's system. "
        "Note that the user may have to approve the command before it is executed.\n\n"
        "In using these tools, adhere to the following guidelines:\n"
        "1. Execute system commands directly, adapting to the user's OS and shell.\n"
        "2. By default, the shell will initialize in the project root. If in a new shell, cd to the appropriate "
        "directory and do necessary setup in addition to running the command.\n"
        "3. For ANY commands that would require user interaction, ASSUME THE USER IS NOT AVAILABLE TO INTERACT "
        "and PASS THE NON-INTERACTIVE FLAGS.\n"
        "4. Dont include any newlines in the command.\n"
        "5. CRITICAL: Commands touching files outside the workspace need user approval for security.",
        {
            "command": {"type": "string",
                        "description": "The CLI command to execute. Must be valid for the current OS and free of "
                                       "harmful instructions."},
            "requires_approval": {
                "type": "boolean",
                "description": "Set to true if the command requires user approval. Required for: destructive "
                               "operations, commands operating outside workspace boundaries, or potentially risky "
                               "operations. Set to false for safe operations within workspace.",
            },
            "explanation": _EXPLANATION_PROP,
        },
        ["command", "requires_approval"],
    ),
    _tool(
        "delete_file",
        "Deletes a file at the specified path. The operation fails gracefully if the file doesn't exist "
        "or the operation is rejected for security reasons.",
        {
            "target_file": {"type": "string",
                            "description": "REQUIRED: The absolute path of the file to delete."},
            "explanation": _EXPLANATION_PROP,
        },
        ["target_file"],
    ),
]


# ---------------- 其它 harness 风格的工具定义（用于泛化） ----------------

_BASH_TOOL = _tool(
    "Bash",
    "Executes a given bash command in a persistent shell session with optional timeout, "
    "ensuring proper handling and security measures.\n"
    "Before executing the command, please follow these steps:\n"
    "1. Working directory: by default the shell starts in the project root.\n"
    "2. For commands that require user interaction, pass the non-interactive flags.\n"
    "3. Avoid using the shell for file operations that dedicated tools can do better.",
    {
        "command": {"type": "string", "description": "The command to execute"},
        "description": {"type": "string",
                        "description": "Clear, concise description of what this command does in 5-10 words"},
        "timeout": {"type": "number",
                    "description": "Optional timeout in milliseconds (max 600000)"},
    },
    ["command"],
)

_TERMINAL_TOOL = _tool(
    "terminal",
    "系统终端工具，用于执行命令行指令。命令会在项目根目录下以非交互模式执行。",
    {
        "command": {"type": "string", "description": "要执行的命令行指令"},
    },
    ["command"],
)

_RUN_TERMINAL_CMD_TOOL = _tool(
    "run_terminal_cmd",
    "请求在用户的终端中执行一条命令。命令会在项目根目录初始化，"
    "需要交互的命令必须带上非交互参数，并等待命令完整返回。",
    {
        "cmd": {"type": "string", "description": "要执行的终端命令"},
        "explanation": {"type": "string", "description": "一句话说明为什么执行这条命令"},
    },
    ["cmd"],
)

# 每个档位 = 一套真实的 harness 工具签名 + 对应的 arguments 构造规则
TOOL_PROFILES = [
    {
        "id": "codebuddy",
        "weight": 6,
        "tools": CODEBUDDY_TOOL_MENU,
    },
    {
        "id": "claude-bash",
        "weight": 1,
        "tools": [_BASH_TOOL],
    },
    {
        "id": "terminal",
        "weight": 1,
        "tools": [_TERMINAL_TOOL],
    },
    {
        "id": "run_terminal_cmd",
        "weight": 1,
        "tools": [_RUN_TERMINAL_CMD_TOOL],
    },
]

_profile_bag = []


def pick_profile():
    """按权重发牌（袋装抽样），保证各档位占比精确等于 weight 比例，
    而不是随机抽样那种小样本下的高方差分布。"""
    global _profile_bag
    if not _profile_bag:
        bag = []
        for p in TOOL_PROFILES:
            bag.extend([p] * p["weight"])
        rng.shuffle(bag)
        _profile_bag = bag
    return _profile_bag.pop()


# 每个档位里"那个能跑命令的工具"叫什么名字
_PROFILE_SHELL_TOOL = {
    "codebuddy": "execute_command",
    "claude-bash": "Bash",
    "terminal": "terminal",
    "run_terminal_cmd": "run_terminal_cmd",
}


def shell_tool_name(profile):
    return _PROFILE_SHELL_TOOL[profile["id"]]


def build_arguments(profile, command, short_desc, requires_approval, is_long_running=False):
    """按档位构造 tool_call 的 arguments。

    模型要学的正是这件事：参数名照 prompt 里给的 schema 抄，命令本身才是真正的技能。
    """
    pid = profile["id"]
    if pid == "codebuddy":
        return {"command": command, "requires_approval": requires_approval}
    if pid == "claude-bash":
        args = {"command": command, "description": short_desc}
        if is_long_running:
            args["timeout"] = 600000
        return args
    if pid == "terminal":
        return {"command": command}
    if pid == "run_terminal_cmd":
        return {"cmd": command}
    raise ValueError(f"未知工具档位: {pid}")


def make_tools_list(profile):
    """返回该档位的工具菜单（深拷贝一份，避免外部改动污染模板）"""
    return json.loads(json.dumps(profile["tools"], ensure_ascii=False))


def make_call(content, tool, args):
    """构造一条「assistant + tool_call」消息。

    tool 可以是菜单里的任意工具——既能表达 shell 工具（execute_command / Bash / ...），
    也能表达 CodeBuddy 的文件工具（search_file / read_file / write_to_file）。
    """
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)},
        }],
    }


def make_result(call_message, content, tool=None):
    """配对一条 tool 结果消息；name 默认取对应调用里的工具名。"""
    return {
        "role": "tool",
        "name": tool or call_message["tool_calls"][0]["function"]["name"],
        "tool_call_id": call_message["tool_calls"][0]["id"],
        "content": content,
    }


def wrap_tool_result(profile, tool_name, content, is_error=False):
    """根据 agent 档位，对 shell 工具的输出做环境真实感包装。

    在真实 CodeBuddy 环境中，execute_command 的返回往往带有 JSON 外壳：
      - 成功：{"status":"success","success":true,"result":{"type":"execute_command_result","stdout":...,"stderr":"","exitCode":0}}
      - 失败：{"status":"error","success":false,"result":{"type":"execute_command_result","stdout":...,"stderr":"","exitCode":1}}
    
    60% 概率添加 CodeBuddy JSON 外壳，40% 保持原始文本，使模型同时适应结构化返回与原始终端输出。
    """
    if not content:
        return ""
    if profile.get("id") == "codebuddy" and tool_name == "execute_command":
        if rng.random() < 0.6:
            if is_error:
                envelope = {
                    "status": "error",
                    "success": False,
                    "result": {
                        "type": "execute_command_result",
                        "stdout": content,
                        "stderr": "",
                        "exitCode": 1,
                    },
                }
            else:
                envelope = {
                    "status": "success",
                    "success": True,
                    "result": {
                        "type": "execute_command_result",
                        "stdout": content,
                        "stderr": "",
                        "exitCode": 0,
                    },
                }
            return json.dumps(envelope, ensure_ascii=False)
    return content


def make_mta_content(project):
    """构造一段真实感的 mta.yaml 内容，用于演示「读取源文件内容 -> 写入根目录 mta.yaml」。"""
    return (
        "_schema-version: '3.1'\n"
        f"ID: {project.lower()}\n"
        "version: 1.0.0\n"
        "parameters:\n"
        "  enable-parallel-deployments: true\n"
        "modules:\n"
        f"  - name: {project.lower()}-srv\n"
        "    type: nodejs\n"
        "    path: gen/srv\n"
    )


def project_root(rule):
    """样本里统一的项目根目录（相对路径）。

    使用 "." 代表当前工作目录，避免模型记住固定的绝对路径。
    CodeBuddy 的文件类工具在实际运行时会自动解析相对路径，
    使用 "." 能让模型适配任何真实工作区位置。
    """
    return "."


def to_rel_path(path_str):
    """将路径统一转换为以 .\\ 开头的相对路径，内部反斜杠统一。"""
    p = path_str.replace("/", "\\").lstrip(".\\").lstrip("\\")
    return f".\\{p}"


def locate_mta(rule):
    """返回 (匹配到的源文件相对路径, 该工作区下的全部 mta 候选文件相对路径)。

    真实项目里 mta-*.yaml 有时放在项目根目录，有时放在 mta 子目录，
    统一使用以 .\\ 开头的相对路径。
    """
    ws_mtas = sorted({r["mta"] for r in RULES if r["workspace"] == rule["workspace"]})
    mta_dir = rng.choice(["", "mta\\"])
    mta_src = to_rel_path(f"{mta_dir}{rule['mta']}")
    candidates = [to_rel_path(f"{mta_dir}{m}") for m in ws_mtas]
    if mta_src not in candidates:
        candidates.append(mta_src)
    rng.shuffle(candidates)
    return mta_src, candidates


def make_search_mta_fail(rule, candidates_rel):
    """构造真实的 search_file 未匹配到源文件的情况：
    情况 1: 工作区中没有任何 mta yaml（返回空列表 []）
    情况 2: 找到了其他 mta 文件，但唯独缺少当前项目/环境所需的 mta 文件
    """
    if rng.random() < 0.5:
        return json.dumps([], ensure_ascii=False)
    else:
        other_files = [c for c in candidates_rel if rule["mta"] not in c]
        if not other_files:
            other_files = [".\\mta-backup.yaml", ".\\config.yaml"]
        return json.dumps(other_files, ensure_ascii=False)



# ============================================================
# 3. 环境同义词（中英混合）
# ============================================================

ENV_SYNONYMS = {
    "dev": [
        "开发环境", "开发", "dev", "dev环境", "d环境",
        "development", "development env", "prequality", "prequality环境",
    ],
    "quality": [
        "quality环境", "quality", "qa", "qa环境", "测试环境", "q环境",
        "test", "testing", "test env",
    ],
}

ENV_DISPLAY = {"dev": "开发环境", "quality": "quality 环境"}


# ============================================================
# 4. 提示词
# ============================================================

# CodeBuddy craft-agent 系统提示里的行为约束（精简摘录，保留与工具调用强相关的几条）
_BASE_SYSTEM_PROMPT_CODEBUDDY = (
    "你是 CodeBuddy 的开发与运维助手，在用户的工作区中直接执行任务。\n\n"
    "## 沟通与行动\n"
    "- CRITICAL 简洁回复：一句话描述行动，避免冗长的计划说明\n"
    "- CRITICAL 言必行：完成行动描述后，务必执行对应的工具\n"
    "- CRITICAL 工具结果处理：禁止直接呈现工具执行结果给用户，必须理解分析结果内容，为下一步执行提供依据\n\n"
    "## Shell 执行环境规范\n"
    "- 使用系统默认的 shell 执行命令（当前环境为 Windows PowerShell）\n"
    "- 确保命令语法符合当前 shell 的要求，不要使用其他 shell 特有的语法特性\n"
    "- 在使用 execute_command 时，确保命令能在当前 shell 中正确执行\n"
    "- 对于需要用户交互的命令，假定用户不可用，必须传入非交互参数，并等待命令完整返回后再继续"
)

# 非 CodeBuddy 档位使用的中性提示词：不提具体产品名，也不写死 shell
_BASE_SYSTEM_PROMPT_NEUTRAL = (
    "你是一个专业的开发与运维助手，在用户的项目工作区中直接执行任务。\n\n"
    "## 沟通与行动\n"
    "- 简洁回复：一句话描述行动，避免冗长的计划说明\n"
    "- 言必行：完成行动描述后，务必执行对应的工具\n"
    "- 工具结果处理：不要直接复述工具输出，必须理解结果内容，为下一步执行提供依据\n\n"
    "## Shell 执行环境规范\n"
    "- 使用系统默认的 shell 执行命令，确保命令语法符合当前 shell 的要求\n"
    "- 对于需要用户交互的命令，假定用户不可用，必须传入非交互参数\n"
    "- 长时间命令要设置足够长的超时，并等待命令完整返回后再继续"
)

_BASE_SYSTEM_PROMPT_NEUTRAL_EN = (
    "You are a professional DevOps assistant that executes tasks directly in the user's project workspace.\n\n"
    "## Communication and action\n"
    "- Keep replies short: one sentence describing the action, no long plans\n"
    "- Always follow through: after describing the action, actually call the tool\n"
    "- Never dump raw tool output; interpret it and use it to decide the next step\n\n"
    "## Shell environment\n"
    "- Use the system's default shell and make sure the command syntax fits it\n"
    "- For interactive commands, assume the user is unavailable and pass non-interactive flags\n"
    "- Set a generous timeout for long-running commands and wait for them to finish"
)


def pick_system_prompt(profile):
    """system prompt 也要跟着 harness 走：CodeBuddy 档位用 CodeBuddy 风格，其余用中性风格。"""
    if profile["id"] == "codebuddy":
        return rng.choice([
            _BASE_SYSTEM_PROMPT_CODEBUDDY,
            _BASE_SYSTEM_PROMPT_CODEBUDDY,
            _BASE_SYSTEM_PROMPT_CODEBUDDY,
        ])
    return rng.choice([
        _BASE_SYSTEM_PROMPT_NEUTRAL,
        _BASE_SYSTEM_PROMPT_NEUTRAL,
        _BASE_SYSTEM_PROMPT_NEUTRAL_EN,
    ])

# btp-deploy agent 的指令正文（= .codebuddy/agents/btp-deploy.md 的正文）。
# 与线上提示词逐字一致，唯一新增的是一张「项目/环境 <-> 登录 code <-> MTA 文件」对照表。
# 原则：方法层面的东西（命令具体怎么写、失败怎么处理）一律不进提示词，全部交给样本去教会模型。
DEPLOY_AGENT_INSTRUCTION = (
    "请根据以下步骤帮我完成 SAP BTP 项目的构建与部署，注意：不要对项目的源代码做任何检查、审查或分析，"
    "也不要尝试理解或修改代码内容，严格按照下面的步骤直接执行部署操作即可。\n\n"
    "## 工作区 / 项目 / 环境 / 登录 code / MTA 文件 对照表（MPB 工作区）\n\n"
    "| 项目 | 环境 | 登录 code | MTA 文件 |\n"
    "| --- | --- | --- | --- |\n"
    "| HC | 开发环境 | `163-d-hc` | `mta-develop-hc.yaml` |\n"
    "| CPT | 开发环境 | `162-d-cpt` | `mta-develop.yaml` |\n"
    "| PT | 开发环境 | `162-d-pt` | `mta-develop-pt.yaml` |\n\n"
    "登录 code 必须从上表原样复制，不要自行推测或改写格式。\n\n"
    "首先根据目标环境匹配对应的登录 code，调用本地登录接口完成 BTP 空间登录。\n\n"
    "接着根据部署环境，在当前工作目录（.）中查找对应的 MTA 配置文件，找到匹配的文件后，"
    "直接使用 Copy-Item 命令将其复制覆盖为根目录下的 .\\mta.yaml 文件；如果在搜索结果中未能匹配到对应的 MTA 配置文件，则立即中止后续流程并报错。\n\n"
    "接着，在项目根目录执行 mbt build 命令进行项目打包，这个构建过程大约需要 2 分钟甚至更久，"
    "所以必须特别注意 timeout 设置，请将命令超时时间设置得足够长（建议至少 10 分钟 / 600 秒），"
    "使用非交互模式运行，绝对不要因为 timeout 而中途终止命令，必须等待命令完整返回退出码后再进行下一步。\n\n"
    "构建完成后，获取构建日志的最后 10 行，从中匹配类似 the MTA archive generated at: 的行，"
    "提取出 .mtar 文件的完整路径。\n\n"
    "接着，使用提取到的文件路径执行 cf deploy <构建文件路径> 命令进行部署，这个部署过程同样大约需要 2 分钟甚至更久，"
    "所以也必须特别注意 timeout 设置，请将命令超时时间设置得足够长（建议至少 10 分钟 / 600 秒），"
    "使用非交互模式运行，绝对不要因为 timeout 而中途终止命令，必须等待命令完整返回退出码。\n\n"
    "最后，部署完成后，请进行清理工作：删除之前在根目录下创建的临时 .\\mta.yaml 文件，"
    "同时删除构建和部署过程中产生的所有多余文件（例如构建日志文件、mta_archives 目录下生成的 .mtar 归档文件、"
    "以及 .mta_build_tmp 等临时目录和文件），确保项目目录恢复到部署前的干净状态。\n\n"
    "如果任何步骤失败则中止后续流程并输出错误信息。"
)

# 用户消息里携带 SOP 的兜底形态（不依赖 agent 配置）
USER_MESSAGE_SOP = DEPLOY_AGENT_INSTRUCTION + "\n\n当前目标：{target_goal}。"

TARGET_GOAL_TEMPLATES = [
    "部署到 {ws} 工作区 {proj} 项目的 {env}",
    "帮我把 {ws} 的 {proj} 项目发布到 {env}",
    "部署项目：工作区 {ws}，项目 {proj}，环境 {env}",
    "发布 {ws} 下面的 {proj} {env}",
    "将 {proj} 部署至 {ws} 工作区的 {env}",
    "把 {proj} 项目的 {env} 部署到 {ws}",
]


# ============================================================
# 5. 构建与部署流水线（前缀展开 + 失败分支）
# ============================================================

MTAR_VERSION_POOL = ["1.0.0", "1.2.0", "2.0.1", "0.9.7", "3.4.0"]
MTAR_SUFFIX_POOL = ["", "", "", "-srv", "-app", "-db"]


def make_mtar_name(project):
    """随机生成一个真实感的归档文件名（带后缀/版本号的变化，避免模型背公式）。"""
    base = project.lower()
    return f"{base}{rng.choice(MTAR_SUFFIX_POOL)}_{rng.choice(MTAR_VERSION_POOL)}.mtar"


def make_mtar_path(project):
    """相对项目根目录的归档路径（用于 devops 单步样本）。"""
    return f"mta_archives\\{make_mtar_name(project)}"


def make_chain_mtar(rule):
    r"""返回 (构建日志里出现的路径, cf deploy 使用的路径, 项目根目录, 是否需要拼接)。

    使用相对路径后，日志里打印的路径和 cf deploy 用的路径都是相对路径。
    随机两种格式让模型学会处理不同的日志输出：
      A. 日志打印 mta_archives\xxx.mtar（无 .\前缀）-> 直接使用；
      B. 日志打印 .\mta_archives\xxx.mtar（带 .\前缀）-> 直接使用。
    """
    name = make_mtar_name(rule["project"])
    rel = f"mta_archives\\{name}"
    root = project_root(rule)
    if rng.random() < 0.5:
        return rel, rel, root, False
    dotrel = f".\\{rel}"
    return dotrel, dotrel, root, False


def make_build_log(archive_path):
    """构造 `mbt build` 尾部输出（截取最后 10 行）。

    归档行落在「最后 10 行」之内，配合 Select-Object -Last 10，
    确保返回给模型的内容只有尾部 10 行，防止撑爆上下文。
    """
    lines = [
        "[10:12:01] INFO validating the MTA project",
        "[10:12:02] INFO module ui: validating module content",
        "[10:12:04] INFO module srv: validating module content",
        "[10:12:05] INFO building the MTA project",
        "[10:12:06] INFO building the module ui",
        "[10:12:31] INFO module ui: running npm install",
        "[10:13:18] INFO module ui: build succeeded",
        "[10:13:19] INFO building the module srv",
        "[10:13:44] INFO module srv: running npm install",
        "[10:14:20] INFO module srv: build succeeded",
        "[10:14:21] INFO assembling the MTA archive",
        "[10:14:22] INFO reading the mta.yaml descriptor",
        "[10:14:22] INFO resolving module dependencies",
        "[10:14:24] INFO collecting the build artifacts",
        "[10:14:25] INFO generating the MTA archive manifest",
        "[10:14:26] INFO compressing the archive content",
        "[10:14:28] INFO archive content compressed",
        "[10:14:29] INFO writing the MTA archive",
        f"[10:14:30] INFO the MTA archive generated at: {archive_path}",
        "[10:14:30] INFO cleaning temporary files",
        "[10:14:31] INFO build succeeded",
    ]
    return "\n".join(lines[-10:])


STEP_ORDER = ["login", "search_mta", "copy_mta", "build", "deploy", "clean"]
# 可失败步骤包含：登录失败、MTA 文件未查找到、MTA 复制失败、构建失败、部署失败。
FAILABLE_STEPS = ["login", "search_mta", "copy_mta", "build", "deploy"]



def build_trajectory_group(rule, env_str, system_prompt, user_prompt, group_id, fail_points, profile):
    """针对一条 (规则 x 用户指令) 生成一个完整样本组。

    组内样本 = 全部前缀（冷启动 + 中途推进）+ 成功收尾 + 若干失败中止分支。
    组内样本共享同一个上下文前缀与同一套工具签名，切分时必须整组进 train 或整组进 val。
    """
    ws = rule["workspace"]
    proj = rule["project"]
    code = rule["code"]
    mta = rule["mta"]
    env_disp = ENV_DISPLAY[rule["env"]]

    tools_list = make_tools_list(profile)
    tool_name = shell_tool_name(profile)
    mtar_log, mtar_full, mta_root, need_join = make_chain_mtar(rule)

    mta_src, mta_candidates = locate_mta(rule)

    def shell_call(content, command, requires_approval, short_desc, is_long_running=False):
        args = build_arguments(profile, command, short_desc, requires_approval, is_long_running)
        return make_call(content, tool_name, args)

    # 丰富且真实的报错候选（覆盖真实环境中的各类异常：HTTP、PowerShell 语法错误、CLI 退出码异常等）
    login_fail_choice = rng.choice([
        ("Invoke-RestMethod : The remote server returned an error: (401) Unauthorized.\r\n"
         f"login failed for space '{code}'"),
        (f"Invoke-RestMethod : {{\"code\":500,\"data\":\"error\",\"message\":\"Internal Server Error: space '{code}' not registered\"}}\r\n"
         "At line:1 char:1\r\n"
         "+ Invoke-RestMethod -Uri http://localhost:3000/space/login -Method Post ...\r\n"
         "+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n"
         "    + CategoryInfo          : InvalidOperation: (System.Net.HttpWebRequest:HttpWebRequest) [Invoke-RestMethod], WebException\r\n"
         "    + FullyQualifiedErrorId : WebCmdletWebResponseException,Microsoft.PowerShell.Commands.InvokeRestMethodCommand"),
        ("Invoke-RestMethod : 无法连接到远程服务器 http://localhost:3000/space/login\r\n"
         "Connect-Failure: No connection could be made because the target machine actively refused it 127.0.0.1:3000"),
    ])

    build_fail_choice = rng.choice([
        # 1. 真实用户遇到的 PowerShell 管道与重定向语法解析错误（ParserError）
        ("At C:\\Users\\azt1szh\\AppData\\Local\\Temp\\genie-ps-1789353580462-mxbplr.ps1:813 char:13\r\n"
         "+ mbt build 2>&| > tmp.log\r\n"
         "+             ~\r\n"
         "Missing file specification after redirection operator.\r\n"
         "The ampersand (&) character is not allowed. The & operator is reserved for future use; wrap an ampersand in double quotation marks (\"&\") to pass it as part of a string.\r\n"
         "    + CategoryInfo          : ParserError: (:) [], ParentContainsErrorRecordException\r\n"
         "    + FullyQualifiedErrorId : MissingFileSpecification"),
        # 2. mbt build 校验与构建失败
        ("[INFO] validating the MTA project\r\n"
         "[ERROR] the MTA project is not valid\r\n"
         "[ERROR] could not find mta.yaml in current directory or syntax error\r\n"
         "[ERROR] build failed, no archive was generated"),
        # 3. 超时或异常终止
        ("mbt build : The process was terminated because the timeout period elapsed.\r\n"
         "[ERROR] execution timed out (600s), build aborted\r\n"
         "Process exited with code -1"),
    ])

    deploy_fail_choice = rng.choice([
        # 1. 真实用户遇到的 MTA 文件未找到（模型若幻觉文件名，真实环境直接报此错）
        (f"FAILED\r\n"
         f"Error retrieving MTA: Could not find MTA {mtar_full}\r\n"
         f"Tip: use 'cf deploy -h' for more info."),
        # 2. 应用启动失败
        (f"Deploying multi-target app archive {mtar_full} in org my-org / space {code}...\r\n"
         "Uploading application...\r\n"
         "[ERROR] Deployment failed: application crashed or failed to start (exit code 1)"),
        # 3. 502 / 网关错误
        (f"Deploying multi-target app archive {mtar_full}...\r\n"
         "Error: Service broker failed to provision service: 502 Bad Gateway\r\n"
         "CF-ServiceBrokerBadResponse(10001): Service broker failed to provision\r\n"
         "[ERROR] Deployment failed"),
    ])

    # 每个 step 形如 dict(key, pre, ok, fail, label)。
    # 中间轮措辞统一为「状态 + 下一步动作 + 冒号」，禁止完成态措辞，
    # 把「全部完成 / 已就绪」这类收尾语气全部留给 success summary。
    steps = [
        {
            "key": "login",
            "pre": shell_call(
                f"开始执行部署流程。先完成 {ws} 工作区 {proj} 项目（{env_str}）的空间登录，"
                f"对应的登录 code 为 `{code}`：",
                ("Invoke-RestMethod -Uri http://localhost:3000/space/login -Method Post "
                 f"-ContentType application/json -Body (@{{space='{code}'}}|ConvertTo-Json)"),
                True,
                "登录 BTP 空间",
            ),
            "ok": wrap_tool_result(profile, tool_name, json.dumps({"code": 200, "data": "success", "message": "success"}, ensure_ascii=False), False),
            "fail": wrap_tool_result(profile, tool_name, login_fail_choice, True),
            "label": "空间登录",
        },
    ]

    # ---- 「找到对应 mta 文件并直接复制为根目录 .\\mta.yaml」（方案 A：两步走，相对路径）----
    mta_rel = to_rel_path(mta_src)
    mta_yaml_rel = ".\\mta.yaml"
    cand_rel_list = [to_rel_path(c) for c in mta_candidates]

    if profile["id"] == "codebuddy":
        steps.append({
            "key": "search_mta",
            "pre": make_call(
                f"空间 `{code}` 已登录。接下来在当前工作目录（`.`）中查找与 {env_disp} 匹配的 MTA 配置文件：",
                "search_file",
                {"target_directory": ".", "pattern": "mta*.yaml", "recursive": True}),
            "ok": json.dumps(cand_rel_list, ensure_ascii=False),
            "fail": make_search_mta_fail(rule, cand_rel_list),
            "label": "查找 MTA 配置文件",
        })
        steps.append({
            "key": "copy_mta",
            "pre": shell_call(
                f"在搜索结果中已匹配到目标源文件 `{mta_rel}`。直接使用 Copy-Item 将其复制覆盖为根目录 `{mta_yaml_rel}`：",
                f"Copy-Item {mta_rel} -Destination {mta_yaml_rel} -Force",
                False,
                "复制 MTA 配置为 mta.yaml"),
            "ok": wrap_tool_result(profile, tool_name, "", False),
            "fail": wrap_tool_result(profile, tool_name,
                                     f"Copy-Item : Cannot find path '{mta_rel}' because it does not exist.\r\n"
                                     f"At line:1 char:1\r\n"
                                     f"+ Copy-Item {mta_rel} -Destination {mta_yaml_rel} -Force\r\n"
                                     f"+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n"
                                     f"    + CategoryInfo          : ObjectNotFound: ({mta_rel}:String) [Copy-Item], ItemNotFoundException\r\n"
                                     f"    + FullyQualifiedErrorId : PathNotFound,Microsoft.PowerShell.Commands.CopyItemCommand",
                                     True),
            "label": "复制 MTA 配置文件",
        })
    else:
        copy_cmd = (f"$f=(Get-ChildItem -Path . -Recurse -File -Filter {mta} "
                    f"| Select-Object -First 1).FullName; Copy-Item $f -Destination {mta_yaml_rel} -Force")
        steps.append({
            "key": "copy_mta",
            "pre": shell_call(
                f"空间 `{code}` 已登录。接下来在项目中查找与 {env_disp} 匹配的 MTA 配置文件 `{mta}`，"
                f"并把它复制为根目录下的 `{mta_yaml_rel}`（不存在就新建，已存在就覆盖）：",
                copy_cmd, False, "查找并复制 MTA 配置为 mta.yaml"),
            "ok": wrap_tool_result(profile, tool_name, "", False),
            "fail": wrap_tool_result(profile, tool_name, f"Copy-Item : Cannot find path '{mta}' because it does not exist.", True),
            "label": "MTA 文件复制",
        })

    steps.append({
        "key": "build",
        "pre": shell_call(
            "`mta.yaml` 已就绪。接下来在项目根目录执行 `mbt build` 打包构建"
            "（超时 600 秒，非交互模式，需等待命令完整返回）：",
            "mbt build 2>&1 | Select-Object -Last 10", False, "构建 MTA 项目", True),
        "ok": wrap_tool_result(profile, tool_name, make_build_log(mtar_log), False),
        "fail": wrap_tool_result(profile, tool_name, build_fail_choice, True),
        "label": "项目构建（mbt build）",
    })

    deploy_intro = (
        "构建完成。取构建日志最后 10 行，匹配到 `the MTA archive generated at:` 一行，"
        f"得到归档路径 `{mtar_full}`。"
        "接下来执行 `cf deploy` 部署（超时 600 秒，非交互模式，需等待命令完整返回）："
    )

    steps.append({
        "key": "deploy",
        "pre": shell_call(
            deploy_intro,
            f"cf deploy {mtar_full} -f", True, "部署 MTAR 归档到 BTP", True),
        "ok": wrap_tool_result(profile, tool_name, (f"Deploying multi-target app archive {mtar_full} in org my-org / space {code}...\r\n"
               "Application started\r\n"
               "Process finished."), False),
        "fail": wrap_tool_result(profile, tool_name, deploy_fail_choice, True),
        "label": "CF 部署（cf deploy）",
    })

    if profile["id"] == "codebuddy":
        # 根目录 mta.yaml 是「文件」，交给 delete_file 更贴合工具语义，也让 delete_file 拿到真实演示；
        # 目录与构建产物再交给 execute_command（此时已不含 mta.yaml，避免重复删同一个东西）。
        steps.append({
            "key": "clean",
            "pre": make_call(
                "部署动作已返回成功。先删除临时创建的根目录 `.\\mta.yaml`：",
                "delete_file",
                {"target_file": ".\\mta.yaml"}),
            "ok": "",
            "fail": "",
            "label": "删除临时 mta.yaml",
        })
        clean_narration = (
            "`.\\mta.yaml` 已删除。接着清理 `mta_archives` 目录、生成的 `.mtar` 归档"
            "与 `.mta_build_tmp` 等构建产物："
        )
        clean_cmd = ("Remove-Item -Path mta_archives, .mta_build_tmp "
                     "-Recurse -Force -ErrorAction SilentlyContinue")
    else:
        clean_narration = (
            "部署动作已返回成功。接下来执行清理，删除临时 `.\\mta.yaml`、`mta_archives` 目录"
            "以及 `.mta_build_tmp` 等构建产物："
        )
        clean_cmd = ("Remove-Item -Path .\\mta.yaml, mta_archives, .mta_build_tmp "
                     "-Recurse -Force -ErrorAction SilentlyContinue")

    steps.append({
        "key": "clean",
        "pre": shell_call(clean_narration, clean_cmd, True, "清理临时构建产物"),
        "ok": wrap_tool_result(profile, tool_name, "", False),
        "fail": "",
        "label": "环境清理",
    })

    def success_summary():
        return {
            "role": "assistant",
            "content": (
                f"✅ **SAP BTP 构建与部署全流程已完成**\n\n"
                f"- **目标环境**：{ws} 工作区 / {proj} 项目 / {env_disp}\n"
                f"- **空间登录**：已登录 space code `{code}`\n"
                f"- **配置同步**：`{mta_rel}` 已直接复制为根目录 `.\\mta.yaml`\n"
                f"- **MTA 构建**：`mbt build` 已生成，归档完整路径 `{mtar_full}`\n"
                f"- **CF 部署**：`cf deploy` 执行成功，应用已启动\n"
                f"- **环境清理**：临时 `.\\mta.yaml` 与构建产物已删除，项目目录已恢复干净状态"
            ),
        }

    def error_summary(failed_index, failed_step):
        failed_label = failed_step["label"]
        done = "、".join(s["label"] for s in steps[:failed_index]) or "无"
        if failed_step["key"] == "search_mta":
            detail = (
                f"在当前工作目录（`.`）中未能找到与 {env_disp} 匹配的 MTA 配置文件（未找到 `{mta}`）。\n\n"
                f"- **目标环境**：{ws} 工作区 / {proj} 项目 / {env_disp}\n"
                f"- **已完成步骤**：{done}\n"
                f"- **失败步骤**：{failed_label}\n"
                f"- **后续步骤**：已按要求停止，未继续执行后续构建与部署\n\n"
                f"请检查当前工作区目录下是否存在与目标环境匹配的 MTA 配置文件后再试。"
            )
        else:
            detail = (
                f"第 {failed_index + 1} 步「{failed_label}」执行失败。\n\n"
                f"- **目标环境**：{ws} 工作区 / {proj} 项目 / {env_disp}\n"
                f"- **已完成步骤**：{done}\n"
                f"- **失败步骤**：{failed_label}\n"
                f"- **后续步骤**：已按要求停止，未继续执行\n\n"
                f"请根据上面的报错信息排查后重试。"
            )
        return {
            "role": "assistant",
            "content": f"❌ **流程已中止**：{detail}",
        }

    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    samples = []

    def windowed_samples(history, target):
        """同一个目标动作用多个历史窗口各生成一条样本（完整 / 近 2 步 / 近 1 步）。

        模型记住一条命令靠的是「它当过几次答案」，而不是它在上下文里出现过几次：
        只做完整前缀展开时，每条命令在整组里只有 1 次是答案，其余全是背景。
        多窗口同时顺带教会「历史被截断时也能接上下一步」。
        """
        out = []
        seen_lengths = set()
        for window in (None, 2, 1):
            hist = history if window is None else history[-2 * window:]
            if len(hist) in seen_lengths:  # 历史本来就很短时几个窗口会重合，去重
                continue
            seen_lengths.add(len(hist))
            out.append({
                "tools": tools_list,
                "messages": base_messages + list(hist) + [target],
            })
        return out

    # ---- 冷启动：目标是第 1 步 ----
    samples.append({"tools": tools_list, "messages": base_messages + [steps[0]["pre"]]})

    # ---- 成功链路的前缀展开：每个中途状态的目标都是下一次工具调用 ----
    history = []
    for index, step in enumerate(steps):
        history.extend([step["pre"], make_result(step["pre"], step["ok"])])
        if index + 1 < len(steps):
            samples.extend(windowed_samples(history, steps[index + 1]["pre"]))
    # ---- 全部步骤走完：目标是最终汇报（唯一允许出现完成态措辞的地方）----
    samples.append({
        "tools": tools_list,
        "messages": base_messages + list(history) + [success_summary()],
    })

    # ---- 失败分支：第 i 步失败 -> 立即中止并报错，不再发工具调用 ----
    for fail_key in fail_points:
        matching = [i for i, s in enumerate(steps) if s["key"] == fail_key]
        if not matching:
            continue
        fail_index = matching[0]
        fail_step = steps[fail_index]
        if not fail_step.get("fail"):
            continue
        fail_history = []
        for k in range(fail_index):
            step = steps[k]
            fail_history.extend([step["pre"], make_result(step["pre"], step["ok"])])
        fail_history.extend([fail_step["pre"], make_result(fail_step["pre"], fail_step["fail"])])
        # 「看到失败 → 中止并汇报」同样用多窗口，避免只在完整历史下才学得会
        samples.extend(windowed_samples(fail_history,
                                        error_summary(fail_index, fail_step)))

    return samples


def build_deploy_chain_samples():
    """构建完整部署流水线的前缀展开样本池。

    返回 [(key, group_id, sample), ...]

    两种形态：
      A. agent 形态（主）—— SOP 在 system prompt 里（对齐 .codebuddy/agents/btp-deploy.md），
         用户消息只给目标环境。
      B. 兜底形态 —— SOP 在用户消息里，system prompt 只给通用行为约束。
    """
    pool = []

    for rule in RULES:
        key = rule_key(rule)
        is_ambiguous = (rule["workspace"], rule["project"], rule["env"]) in AMBIGUOUS_KEYS
        repeats = AMBIGUOUS_REPEAT if is_ambiguous else BASE_REPEAT

        for rep in range(repeats):
            # ---------- A. agent 形态：SOP 在 system ----------
            for goal_index, goal_tmpl in enumerate(rng.sample(TARGET_GOAL_TEMPLATES, k=2)):
                env_str = rng.choice(ENV_SYNONYMS[rule["env"]])
                user_prompt = goal_tmpl.format(ws=rule["workspace"], proj=rule["project"], env=env_str)
                profile = pick_profile()
                failable_keys = (["login", "search_mta", "copy_mta", "build", "deploy"]
                                 if profile["id"] == "codebuddy"
                                 else ["login", "copy_mta", "build", "deploy"])
                fail_points = rng.sample(failable_keys, k=min(len(failable_keys), FAIL_BRANCHES_PER_TRAJECTORY))
                agent_system = pick_system_prompt(profile) + "\n\n" + DEPLOY_AGENT_INSTRUCTION
                group_id = f"chain-agent|{key}|{rep}|{goal_index}|{profile['id']}"
                for s in build_trajectory_group(rule, env_str, agent_system, user_prompt,
                                                group_id, fail_points, profile):
                    pool.append((key, group_id, s))

            # ---------- B. 兜底形态：SOP 在用户消息 ----------
            goal_tmpl = rng.choice(TARGET_GOAL_TEMPLATES)
            env_str = rng.choice(ENV_SYNONYMS[rule["env"]])
            target_goal = goal_tmpl.format(ws=rule["workspace"], proj=rule["project"], env=env_str)
            user_prompt = USER_MESSAGE_SOP.format(target_goal=target_goal)
            profile = pick_profile()
            failable_keys = (["login", "search_mta", "copy_mta", "build", "deploy"]
                             if profile["id"] == "codebuddy"
                             else ["login", "copy_mta", "build", "deploy"])
            fail_points = rng.sample(failable_keys, k=min(len(failable_keys), FAIL_BRANCHES_PER_TRAJECTORY))
            group_id = f"chain-user|{key}|{rep}|{profile['id']}"
            for s in build_trajectory_group(rule, env_str, pick_system_prompt(profile), user_prompt,
                                            group_id, fail_points, profile):
                pool.append((key, group_id, s))


    return pool


# ============================================================
# 6. 登录单点样本（用户只要求登录 / 只查配置）
# ============================================================

LOGIN_TEMPLATES = [
    "帮我准备 {ws} 工作区下 {proj} 项目的 {env}",
    "请登录 {ws} 的 {proj} {env}，并给我 mta 文件",
    "在 {ws} 工作区下部署 {proj} 的 {env}，帮我调登录接口并告知 mta 文件",
    "帮我调一下本地登录 API：工作区 {ws}，项目 {proj}，环境 {env}",
    "请协助登录 {ws} 的 {proj} 项目（{env}），告知对应的 mta 部署文件",
    "登录 {ws} 的 {proj}（{env}），对应的登录 code 是多少？",
    "切到 {ws} 工作区 {proj} 的 {env}，先做空间登录",
    "帮我用本地接口登录 {ws} {proj} 的 {env}",
]

# 登录失败的真实回包（{code} 会被替换成规则里的 code）。
# 后两类是线上实际遇到过的：space 未注册（500）与路由写错导致返回 404 页面而不是 JSON。
LOGIN_FAILURES = [
    "Invoke-RestMethod : The remote server returned an error: (401) Unauthorized.\r\nlogin failed for space '{code}'",
    "Invoke-RestMethod : {\"code\":500,\"data\":\"error\",\"message\":\"Internal Server Error: space '{code}' not found\"}",
    "Invoke-RestMethod : {\"code\":500,\"data\":\"error\",\"message\":\"Internal Server Error: space '{code}' not registered\"}",
    "<!DOCTYPE html>\r\n<html lang=\"en\">\r\n  <head>\r\n    <meta charset=\"utf-8\">\r\n    <title>Error</title>\r\n  </head>\r\n  <body>\r\n    <pre>Cannot POST /api/login</pre>\r\n  </body>\r\n</html>",
    "Invoke-RestMethod : 无法连接到远程服务器 http://localhost:3000/space/login\r\nConnect-Failure: No connection could be made because the target machine actively refused it 127.0.0.1:3000",
]


def build_login_action_samples():
    pool = []

    for rule in RULES:
        ws = rule["workspace"]
        proj = rule["project"]
        code = rule["code"]
        mta = rule["mta"]
        key = rule_key(rule)
        env_disp = ENV_DISPLAY[rule["env"]]

        for index, tmpl in enumerate(LOGIN_TEMPLATES):
            env_str = rng.choice(ENV_SYNONYMS[rule["env"]])
            user_text = tmpl.format(ws=ws, proj=proj, env=env_str)
            profile = pick_profile()
            tools_list = make_tools_list(profile)
            tool_name = shell_tool_name(profile)
            call_id = f"call_login_{uuid.uuid4().hex[:8]}"

            login_cmd = ("Invoke-RestMethod -Uri http://localhost:3000/space/login -Method Post "
                         f"-ContentType application/json -Body (@{{space='{code}'}}|ConvertTo-Json)")
            login_args = json.dumps(
                build_arguments(profile, login_cmd, "登录 BTP 空间", True), ensure_ascii=False)

            tool_res = json.dumps({"code": 200, "data": "success", "message": "success"}, ensure_ascii=False)

            final_success = rng.choice([
                (f"✅ 登录接口调用完成\n- **工作区**：{ws}\n- **项目**：{proj}（{env_disp}）\n"
                 f"- **登录 code**：`{code}`\n- **部署 MTA 文件**：`{mta}`\n"
                 f"空间已就绪，可以继续执行构建与部署。"),
                (f"已为您完成登录操作：\n1. **项目与环境**：{ws} 工作区 / {proj} 项目（{env_disp}）\n"
                 f"2. **登录 code**：`{code}`\n3. **MTA 配置文件**：`{mta}`\n"
                 f"接口调用状态正常。"),
                (f"登录操作完成\n- 登录 code: `{code}`\n- MTA 部署文件: `{mta}`\n"
                 f"本地 API 已返回登录成功。"),
            ])

            group_id = f"login|{key}|{index}"

            pool.append((key, group_id, {
                "tools": tools_list,
                "messages": [
                    {"role": "system", "content": pick_system_prompt(profile)},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": "", "tool_calls": [{
                        "id": call_id, "type": "function",
                        "function": {"name": tool_name, "arguments": login_args},
                    }]},
                    {"role": "tool", "name": tool_name, "tool_call_id": call_id, "content": wrap_tool_result(profile, tool_name, tool_res, False)},
                    {"role": "assistant", "content": final_success},
                ],
            }))

            # 登录失败分支：教「失败就中止并汇报」，而不是闷头继续往下走。
            # 覆盖真实回包里的各类形态：401 / 500(space 未注册或不存在) / 路由写错返回 404 页面 / 连不上本地服务
            for fail_offset, fail_template in enumerate(rng.sample(LOGIN_FAILURES, k=2)):
                login_fail_str = fail_template.replace("{code}", code)
                if "<!DOCTYPE" in login_fail_str:
                    fail_summary = ("❌ **空间登录失败**：登录接口返回的不是 JSON 而是 404 页面，"
                                    "说明请求地址不对。已停止后续流程，请确认登录接口路径后重试。")
                else:
                    fail_summary = (f"❌ **空间登录失败**：登录接口返回异常，无法登录 space `{code}`。"
                                    f"已停止后续流程，请检查本地登录服务（端口 3000）与空间授权后重试。")
                pool.append((key, group_id + f"|fail{fail_offset}", {
                    "tools": tools_list,
                    "messages": [
                        {"role": "system", "content": pick_system_prompt(profile)},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": "", "tool_calls": [{
                            "id": call_id, "type": "function",
                            "function": {"name": tool_name, "arguments": login_args},
                        }]},
                        {"role": "tool", "name": tool_name, "tool_call_id": call_id,
                         "content": wrap_tool_result(profile, tool_name, login_fail_str, True)},
                        {"role": "assistant", "content": fail_summary},
                    ],
                }))

            # code 写错的自纠样本：历史里是错的 code（不参与监督），目标是「中止 + 按对照表核对」
            # —— 线上就是这么翻车的（space='123-d-pt' 属于凭空捏造，数据里根本没有这个值）
            if index % 2 == 0:
                wrong_code = code[:-1] + rng.choice("0123456789")
                if wrong_code == code:
                    wrong_code = code + "9"
                wrong_args = json.dumps(
                    build_arguments(profile,
                                    login_cmd.replace(f"'{code}'", f"'{wrong_code}'"),
                                    "登录 BTP 空间", True), ensure_ascii=False)
                pool.append((key, group_id + "|wrong-code", {
                    "tools": tools_list,
                    "messages": [
                        {"role": "system", "content": pick_system_prompt(profile)},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": "", "tool_calls": [{
                            "id": call_id, "type": "function",
                            "function": {"name": tool_name, "arguments": wrong_args},
                        }]},
                        {"role": "tool", "name": tool_name, "tool_call_id": call_id,
                         "content": wrap_tool_result(
                             profile, tool_name,
                             LOGIN_FAILURES[2].replace("{code}", wrong_code), True)},
                        {"role": "assistant", "content": (
                            f"❌ **空间登录失败**：space `{wrong_code}` 未注册，登录 code 写错了。"
                            f"已停止后续流程；请按对照表核对目标环境对应的 code（当前应为 `{code}`）后重试。")},
                    ],
                }))

            # 单轮变体：只发出工具调用，训练「首轮直接行动」的行为。
            # 同一模板出两条（旁白措辞不同），把「登录命令当答案」的曝光翻倍 ——
            # 这一条命令是线上最容易翻车的地方（space 写成幻觉值、路由写成 /api/login）。
            for cold_index, cold_narration in enumerate([
                f"登录 code: {code}\nMTA 文件: {mta}",
                f"用对照表里 {ws} {proj}（{env_disp}）对应的 code `{code}` 完成空间登录：",
            ]):
                pool.append((key, group_id + f"|cold{cold_index}", {
                    "tools": tools_list,
                    "messages": [
                        {"role": "system", "content": pick_system_prompt(profile)},
                        {"role": "user", "content": user_text},
                        {"role": "assistant",
                         "content": cold_narration,
                         "tool_calls": [{
                             "id": f"call_login_{uuid.uuid4().hex[:8]}", "type": "function",
                             "function": {"name": tool_name, "arguments": login_args},
                         }]},
                    ],
                }))

    return pool


# ============================================================
# 7. 单步原子动作样本（复制 / 构建 / 部署 / 清理）
# ============================================================

def _copy_action_calls(profile, rule, env_str, mta_src, mta_candidates, action):
    """把「找到 mta 文件并直接复制为根目录 .\\mta.yaml」按档位展开成具体工具调用序列。

    CodeBuddy 档位：search_file 定位 -> shell 执行 Copy-Item 复制；
    只有 shell 的档位用一条「递归查找 + 复制」命令。

    返回 [(pre_content, tool, payload, tool_result), ...]；tool="shell" 时 payload 为
    (command, requires_approval, short_desc, is_long_running)。
    """
    mta_rel = to_rel_path(mta_src)
    mta_yaml_rel = ".\\mta.yaml"
    cand_rel_list = [to_rel_path(c) for c in mta_candidates]
    if profile["id"] == "codebuddy":
        return [
            (f"正在查找与 {env_str} 匹配的 MTA 配置文件：", "search_file",
             {"target_directory": ".", "pattern": "mta*.yaml", "recursive": True},
             json.dumps(cand_rel_list, ensure_ascii=False)),
            (f"已定位到 `{mta_rel}`，直接使用 Copy-Item 复制覆盖为根目录 `{mta_yaml_rel}`：", "shell",
             (f"Copy-Item {mta_rel} -Destination {mta_yaml_rel} -Force", False, "复制 MTA 配置为 mta.yaml", False),
             ""),
        ]
    copy_cmd = (f"$f=(Get-ChildItem -Path . -Recurse -File -Filter {rule['mta']} "
                f"| Select-Object -First 1).FullName; Copy-Item $f -Destination {mta_yaml_rel} -Force")
    return [(action["pre_shell"], "shell",
             (copy_cmd, False, "查找并复制 MTA 配置为 mta.yaml", False), "")]


def build_devops_action_samples():
    pool = []

    for rule in RULES:
        ws = rule["workspace"]
        proj = rule["project"]
        mta = rule["mta"]
        key = rule_key(rule)
        env_str = rng.choice(ENV_SYNONYMS[rule["env"]])
        mtar_path = make_mtar_path(proj)

        mta_src, mta_candidates = locate_mta(rule)

        actions = []

        # ---- 复制配置（CodeBuddy 档位展开为 search_file 定位 + Copy-Item 复制）----
        for q in rng.sample([
            f"把 {ws} 工作区 {proj} 项目 {env_str} 对应的 MTA 文件复制到根目录下的 mta.yaml",
            "根据规则找到匹配的 MTA 文件并覆盖复制为 mta.yaml",
            f"找到 {proj}（{env_str}）匹配的 YAML 文件并复制到根目录下的 mta.yaml",
            f"复制与 {env_str} 匹配的 MTA 文件内容到当前目录的 mta.yaml",
        ], k=2):
            actions.append({
                "kind": "copy",
                "user": q,
                "pre_shell": f"正在查找匹配的 MTA 文件 `{mta}` 并直接复制为根目录下的 `.\\mta.yaml`：",
                "final_shell": f"`{mta}` 已直接复制并覆盖为 `.\\mta.yaml`。",
            })

        # ---- mbt build（结果里带回真实日志行，路径由日志决定）----
        for q in rng.sample([
            f"在项目根目录执行 mbt build 命令对 {proj} 项目进行打包",
            "执行 mbt build 进行 MTA 项目构建",
            "运行 mbt build 打包项目，生成 mtar 文件",
        ], k=2):
            actions.append({
                "kind": "shell", "user": q,
                "pre": "正在执行 `mbt build` 进行项目构建打包：",
                "cmd": "mbt build 2>&1 | Select-Object -Last 10", "approval": False,
                "desc": "构建 MTA 项目", "long": True,
                "result": make_build_log(mtar_path),
                "final": f"MTA 构建完成，构建日志末尾显示归档包为 `{mtar_path}`。",
            })

        # ---- cf deploy（路径由用户在指令中给出）----
        for q in rng.sample([
            f"使用构建日志中的路径执行 cf deploy {mtar_path} -f 进行部署",
            f"把已构建的归档包 {mtar_path} 部署到 SAP BTP",
            f"运行 cf deploy {mtar_path} -f 完成应用发布",
        ], k=2):
            actions.append({
                "kind": "shell", "user": q,
                "pre": f"正在执行 `cf deploy` 部署 `{mtar_path}`：",
                "cmd": f"cf deploy {mtar_path} -f", "approval": True,
                "desc": "部署 MTAR 归档到 BTP", "long": True,
                "result": (f"Deploying multi-target app archive {mtar_path} in org my-org / space {ws}...\r\n"
                           "Application started\r\nProcess finished."),
                "final": f"CF 部署完成，应用已通过 `{mtar_path}` 成功启动。",
            })

        # ---- 清理 ----
        for q in rng.sample([
            "部署完成后进行清理：删除临时 mta.yaml、mta_archives 目录及临时构建文件",
            "清理 mta.yaml 和构建产物，恢复项目目录干净状态",
            "删除临时创建的 mta.yaml 和 .mta_build_tmp 临时目录",
        ], k=2):
            actions.append({
                "kind": "clean", "user": q,
                "pre": "正在清理临时 `.\\mta.yaml`、`mta_archives` 与 `.mta_build_tmp`：",
                "cmd": ("Remove-Item -Path .\\mta.yaml, mta_archives, .mta_build_tmp "
                        "-Recurse -Force -ErrorAction SilentlyContinue"),
                "approval": True, "desc": "清理临时构建产物", "long": False,
                "result": "",
                "final": "清理完成，临时文件与构建产物已删除，项目目录已恢复干净状态。",
            })

        # 重复采样：这一族是「关键命令当答案」的主力，把动作列表复制若干遍，
        # 让 mbt build / cf deploy / Remove-Item 这些命令作为监督目标出现的次数直接翻倍。
        actions = actions * STEP_REPEAT

        for index, action in enumerate(actions):
            profile = pick_profile()

            if action["kind"] == "copy":
                calls = _copy_action_calls(profile, rule, env_str, mta_src,
                                           mta_candidates, action)
                final_text = (f"`{to_rel_path(mta_src)}` 已成功直接复制为根目录 `.\\mta.yaml`。"
                              if profile["id"] == "codebuddy" else action["final_shell"])
            elif action["kind"] == "clean" and profile["id"] == "codebuddy":
                # 与完整部署链保持同一口径：文件交给 delete_file，目录交给 execute_command
                calls = [
                    ("正在删除临时创建的根目录 `.\\mta.yaml`：", "delete_file",
                     {"target_file": ".\\mta.yaml"}, ""),
                    ("`.\\mta.yaml` 已删除。接着清理 `mta_archives` 与 `.mta_build_tmp`：", "shell",
                     ("Remove-Item -Path mta_archives, .mta_build_tmp "
                      "-Recurse -Force -ErrorAction SilentlyContinue",
                      True, "清理临时构建产物", False),
                     ""),
                ]
                final_text = action["final"]
            else:
                calls = [(action["pre"], "shell",
                          (action["cmd"], action["approval"], action["desc"], action["long"]),
                          action["result"])]
                final_text = action["final"]


            messages = [
                {"role": "system", "content": pick_system_prompt(profile)},
                {"role": "user", "content": action["user"]},
            ]
            for step_idx, (pre_content, tool, payload, tool_result) in enumerate(calls):
                if tool == "shell":
                    cmd, approval, desc, is_long = payload
                    tname = shell_tool_name(profile)
                    args = build_arguments(profile, cmd, desc, approval, is_long)
                else:
                    tname, args = tool, payload
                call_message = make_call(pre_content, tname, args)

                # 将工具调用本身作为监督目标的样本（冷启动或前序工具返回后的推进动作）
                if len(calls) > 1 and step_idx == len(calls) - 1:
                    pool.append((key, f"devops|{key}|{index}|step{step_idx}", {
                        "tools": make_tools_list(profile),
                        "messages": list(messages) + [call_message],
                    }))
                elif len(calls) == 1:
                    pool.append((key, f"devops|{key}|{index}|cold", {
                        "tools": make_tools_list(profile),
                        "messages": list(messages) + [call_message],
                    }))

                messages.append(call_message)
                messages.append(make_result(call_message, tool_result, tname))
            messages.append({"role": "assistant", "content": final_text})

            pool.append((key, f"devops|{key}|{index}|flow", {
                "tools": make_tools_list(profile),
                "messages": messages,
            }))

    return pool


# ============================================================
# 8. 工程探查与文件编辑样本（补齐 agent 提示词里其余工具的示范）
# ============================================================
# agent 的工具菜单里有 9 条 schema，但部署 SOP 只会用到 5 条：
#   execute_command / search_file / read_file / write_to_file / delete_file
# 剩下 list_dir / search_content / read_lints / replace_in_file 有 schema 却从没被演示过——
# 模型只能猜参数怎么填，到了真实环境就会拿 shell 命令硬凑。所以这里单独建两族样本补齐。
# 注意：这些请求不属于 btp-deploy 的 SOP（SOP 明确要求「不要检查/审查/分析代码」），
# 因此 system prompt 只用通用行为约束、不拼 SOP，避免教模型违反自己的指令。


def make_dir_listing(mta_files):
    """构造一次 list_dir 的真实返回：工作区根目录下的条目清单。"""
    entries = sorted(mta_files) + ["mta_archives/", "node_modules/", "package.json", "srv/", "ui/"]
    return "\n".join(entries)


def call_sequence_messages(system_prompt, user_text, calls,
                           final_text=None, include_results=True):
    """把 [(旁白, 工具名, 参数, 工具返回), ...] 拼成标准消息序列。

    final_text 传 None 表示不监督收尾文本；include_results=False 表示连工具返回也省略，
    此时序列停在「assistant 发出调用」上 —— 这正是冷启动样本要监督的形态
    （最后一条必须是 assistant，否则模型没有监督目标，自检会直接拦下）。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    for narration, tool, args, result in calls:
        call = make_call(narration, tool, args)
        messages.append(call)
        if include_results:
            messages.append(make_result(call, result, tool))
    if final_text is not None:
        messages.append({"role": "assistant", "content": final_text})
    return messages


def workspaces_by_rule():
    """按工作区聚合规则：list_dir / lint 这类请求面向整个工作区，与具体项目无关。"""
    grouped = {}
    for rule in RULES:
        grouped.setdefault(rule["workspace"], []).append(rule)
    return grouped


def build_explore_samples():
    """工程探查样本：覆盖 list_dir / search_content / read_lints。"""
    pool = []
    codebuddy = next(p for p in TOOL_PROFILES if p["id"] == "codebuddy")

    for ws, rules in workspaces_by_rule().items():
        root = project_root(rules[0])
        mta_files = sorted({r["mta"] for r in rules})
        listing = make_dir_listing(mta_files)

        items = [
            {
                "users": [
                    f"{ws} 工作区项目根目录下都有哪些文件？",
                    f"列一下 {ws} 工作区的项目文件，我确认下 mta 配置在不在根目录",
                    f"{ws} 这个项目里有什么？mta 配置文件放在哪一层？",
                    "根目录下有哪些文件？",
                    f"帮我列一下 {ws} 项目根目录的清单",
                ],
                "calls": [("先看一下项目根目录的结构：", "list_dir",
                           {"target_directory": root}, listing)],
                "final": (f"根目录共 {len(listing.splitlines())} 个条目，"
                          + "、".join(f"`{m}`" for m in mta_files)
                          + " 都在根目录下；另有 `srv/`、`ui/` 两个模块目录，"
                            "以及上次构建留下的 `mta_archives/`。"),
            },
            {
                "users": [
                    f"在 {ws} 工作区里搜一下所有 mta 配置的 `_schema-version`",
                    f"看一下 {ws} 的 mta 配置 schema 版本是不是一致的",
                ],
                "calls": [("逐个确认 mta 配置的 schema 版本声明：", "search_content",
                           {"pattern": "_schema-version", "path": root, "glob": "mta*.yaml"},
                           "\n".join(f"{m}:1:_schema-version: '3.1'" for m in mta_files))],
                "final": (f"命中 {len(mta_files)} 处，{len(mta_files)} 份 mta 配置的 "
                          "`_schema-version` 都是 `3.1`，版本一致。"),
            },
            {
                "users": [
                    f"{ws} 工作区里哪些文件引用了 `mta_archives`？",
                    f"搜一下 {ws} 工作区里 `mta_archives` 的引用位置",
                ],
                "calls": [("搜一下哪些脚本引用了 mta_archives：", "search_content",
                           {"pattern": "mta_archives", "path": root},
                           'srv/package.json:9:    "deploy": "cf deploy mta_archives/*.mtar",')],
                "final": ("只有 `srv/package.json` 的 deploy 脚本引用了 `mta_archives`，"
                          "其余位置没有硬编码归档路径。"),
            },
            {
                "users": [
                    f"检查一下 {ws} 工作区现在的 lint 情况",
                    f"{ws} 工作区有 lint 报错吗？",
                ],
                "calls": [("读取当前工作区的 lint 诊断：", "read_lints",
                           {"paths": [root]}, "No lint errors")],
                "final": "当前工作区没有 lint 报错，`srv/` 和 `ui/` 两个模块都是干净的。",
            },
            {
                "users": [
                    f"{ws} 工作区的 lint 有哪些问题需要修？",
                    f"跑一遍 {ws} 的 lint，把问题列出来",
                ],
                "calls": [("读取当前工作区的 lint 诊断（只看 error / warning）：", "read_lints",
                           {"paths": [root], "severity": ["error", "warning"]},
                           "srv/service.js:42: 'deploy' is defined but never used (no-unused-vars)\n"
                           "srv/service.js:57: Missing semicolon (semi)")],
                "final": ("发现 2 处，都在 `srv/service.js`：第 42 行有个未使用的 `deploy` 变量、"
                          "第 57 行缺分号，都是 warning 级，不影响 `mbt build`。"),
            },
        ]

        for item_index, item in enumerate(items):
            for phrase_index, user_text in enumerate(item["users"]):
                system_prompt = pick_system_prompt(codebuddy)
                # flow: 工具结果 + 解读收尾；cold: 序列停在「首轮直接发出调用」上
                pool.append((f"EXPLORE|{ws}",
                             f"explore|{ws}|{item_index}|{phrase_index}|flow",
                             {"tools": make_tools_list(codebuddy),
                              "messages": call_sequence_messages(
                                  system_prompt, user_text, item["calls"], item["final"])}))
                pool.append((f"EXPLORE|{ws}",
                             f"explore|{ws}|{item_index}|{phrase_index}|cold",
                             {"tools": make_tools_list(codebuddy),
                              "messages": call_sequence_messages(
                                  system_prompt, user_text, item["calls"],
                                  final_text=None, include_results=False)}))

    return pool


def build_edit_samples():
    """文件编辑样本：覆盖 replace_in_file（先 read_file 看清现状，再做精确替换）。"""
    pool = []
    codebuddy = next(p for p in TOOL_PROFILES if p["id"] == "codebuddy")

    for ws, rules in workspaces_by_rule().items():
        root = project_root(rules[0])
        mta_yaml = f"{root}\\mta.yaml"

        for rule in rules:
            content = make_mta_content(rule["project"])
            new_version = rng.choice(["1.1.0", "2.0.0", "2.1.0"])

            items = [
                {
                    "users": [
                        f"把根目录 mta.yaml 的 version 升级到 {new_version}，其他内容别动",
                        f"mta.yaml 里的版本号改成 {new_version}",
                    ],
                    "old": "version: 1.0.0",
                    "new": f"version: {new_version}",
                    "final": f"`mta.yaml` 的 version 已从 `1.0.0` 改为 `{new_version}`，其余内容未改动。",
                },
                {
                    "users": [
                        "根目录 mta.yaml 的 _schema-version 升到 3.2",
                        "把 mta.yaml 的 _schema-version 改成 3.2",
                    ],
                    "old": "_schema-version: '3.1'",
                    "new": "_schema-version: '3.2'",
                    "final": "`mta.yaml` 的 `_schema-version` 已改为 `3.2`，描述符结构没有其它变化。",
                },
                {
                    "users": [
                        "把 mta.yaml 的 enable-parallel-deployments 关掉",
                        "mta.yaml 里把并行部署改成 false（enable-parallel-deployments）",
                    ],
                    "old": "enable-parallel-deployments: true",
                    "new": "enable-parallel-deployments: false",
                    "final": "`mta.yaml` 的 `enable-parallel-deployments` 已置为 `false`，部署会改为串行执行。",
                },
            ]

            for item_index, item in enumerate(items):
                for phrase_index, user_text in enumerate(item["users"]):
                    calls = [
                        ("先读取 `mta.yaml` 确认当前内容：", "read_file",
                         {"filePath": mta_yaml}, content),
                        (f"定位到 `{item['old']}`，做精确替换：", "replace_in_file",
                         {"filePath": mta_yaml, "old_str": item["old"], "new_str": item["new"]}, ""),
                    ]
                    pool.append((f"EDIT|{ws}",
                                 f"edit|{ws}|{rule['project']}|{item_index}|{phrase_index}",
                                 {"tools": make_tools_list(codebuddy),
                                  "messages": call_sequence_messages(
                                      pick_system_prompt(codebuddy), user_text, calls,
                                      item["final"])}))

    return pool


# ============================================================
# 8.5 恢复样本族 + 工具字段语义族
# ============================================================
# 线上最致命的两类错误都出在「出错之后」：
#   1. 把同一条错误调用原样重发 5 次（CodeBuddy 都注入 loop 提示了也不停）；
#   2. 把目录路径塞进 search_file 的 pattern，文件明明存在却 Found 0 files。
# 这一族专门教「怎么纠正」：换正确字段 / 换更精确的取法；仍然失败就按 SOP 中止。
# 监督目标只放在「正确的那一次」上，错误调用只作为历史上下文出现（不参与监督）。


def _codebuddy_profile():
    return next(p for p in TOOL_PROFILES if p["id"] == "codebuddy")


def build_write_file_samples():
    """写入文件专用样本：覆盖 write_to_file 的正确使用。
    使用相对路径（.\\...），演示创建配置文件、小脚本等场景。
    """
    pool = []
    codebuddy = _codebuddy_profile()
    system_prompt = pick_system_prompt(codebuddy)

    write_tasks = [
        {
            "users": [
                "在根目录新建一个 default-env.json 文件，写入本地测试环境变量",
                "帮我创建 .\\default-env.json，配置本地测试变量",
                "新建 default-env.json 配置文件",
            ],
            "file": ".\\default-env.json",
            "content": '{\n  "destinations": [\n    {\n      "name": "srv-api",\n      "url": "http://localhost:4004"\n    }\n  ]\n}',
            "narration": "在项目根目录下创建 `.\\default-env.json`：",
            "final": "`default-env.json` 已创建并写入本地测试环境变量配置。",
        },
        {
            "users": [
                "创建一个本地运行脚本 scripts/start-local.ps1",
                "在 scripts 目录下新建 start-local.ps1 脚本",
                "写一个 PowerShell 启动脚本 scripts/start-local.ps1",
            ],
            "file": ".\\scripts\\start-local.ps1",
            "content": "#!/usr/bin/env pwsh\nWrite-Host 'Starting local services...'\nnpm run start:srv\n",
            "narration": "创建 `.\\scripts\\start-local.ps1` 启动脚本：",
            "final": "`scripts\\start-local.ps1` 脚本已创建完成。",
        },
        {
            "users": [
                "在根目录新建一个 DEPLOY_NOTE.md 记录部署规范",
                "帮我写一个 DEPLOY_NOTE.md 说明文档",
            ],
            "file": ".\\DEPLOY_NOTE.md",
            "content": "# SAP BTP 部署说明\n\n1. 检查空间登录状态\n2. 复制对应环境的 MTA 配置文件到 mta.yaml\n3. 执行 mbt build 和 cf deploy\n",
            "narration": "创建根目录说明文档 `.\\DEPLOY_NOTE.md`：",
            "final": "`DEPLOY_NOTE.md` 文档已创建并写入部署说明。",
        },
    ]

    for item_idx, item in enumerate(write_tasks):
        for user_idx, user_text in enumerate(item["users"]):
            call_msg = make_call(
                item["narration"],
                "write_to_file",
                {"filePath": item["file"], "content": item["content"]}
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
                call_msg,
                make_result(call_msg, "", "write_to_file"),
                {"role": "assistant", "content": item["final"]},
            ]
            pool.append(("WRITE", f"write|{item_idx}|{user_idx}|flow", {
                "tools": make_tools_list(codebuddy),
                "messages": messages,
            }))
            pool.append(("WRITE", f"write|{item_idx}|{user_idx}|cold", {
                "tools": make_tools_list(codebuddy),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                    call_msg,
                ],
            }))

    return pool



def build_recovery_samples():
    """恢复样本：字段语义纠错 / 归档行取不到时改精确匹配 / 二次失败即中止。"""
    pool = []
    codebuddy = _codebuddy_profile()

    for rule in RULES:
        ws = rule["workspace"]
        proj = rule["project"]
        env_disp = ENV_DISPLAY[rule["env"]]
        root = project_root(rule)
        mta_file = rule["mta"]
        mtar_name = f"{proj.lower()}-mpb-d_1.0.0.mtar"

        # ---- A. pattern 里塞了目录 -> Found 0 files -> 换正确字段重试 ----
        wrong_call = make_call(
            f"先找一下 {proj} 在 {env_disp} 用的 MTA 文件：",
            "search_file",
            {"target_directory": root, "pattern": f"mta\\{mta_file}", "recursive": True})
        pool.append((f"RECOVER|{ws}", f"recover|{ws}|{proj}|pattern-field", {
            "tools": make_tools_list(codebuddy),
            "messages": [
                {"role": "system", "content": pick_system_prompt(codebuddy)},
                {"role": "user", "content": f"帮我找到 {ws} 工作区 {proj} 项目对应的 MTA 配置文件"},
                wrong_call,
                make_result(wrong_call, "Found 0 files"),
                make_call("pattern 里不能带目录，改成通配符、把目录放在 target_directory 里重查：",
                          "search_file",
                          {"target_directory": root, "pattern": "mta*.yaml", "recursive": True}),
            ],
        }))

        # ---- B. 构建日志尾部没有归档行（verbose 输出/警告行混入）-> 改用精确匹配 ----
        noisy_build = make_call(
            "先跑一次构建，取最后 10 行找归档路径：",
            "execute_command",
            {"command": "mbt build 2>&1 | Select-Object -Last 10", "requires_approval": False})
        noisy_tail = (
            "(node:23380) Warning: Setting the NODE_TLS_REJECT_UNAUTHORIZED environment variable to '0' "
            "makes TLS connections and HTTPS requests insecure by disabling certificate verification.\r\n"
            "info mpbfiori › Running task minify...\r\n"
            "info mpbfiori › Running task generateComponentPreload...\r\n"
            "info ProjectBuilder Executing cleanup tasks...")
        pool.append((f"RECOVER|{ws}", f"recover|{ws}|{proj}|no-archive-line", {
            "tools": make_tools_list(codebuddy),
            "messages": [
                {"role": "system", "content": pick_system_prompt(codebuddy)},
                {"role": "user", "content": f"把 {ws} 的 {proj} 构建一下，然后告诉我归档包路径"},
                noisy_build,
                make_result(noisy_build, noisy_tail),
                make_call("尾部 10 行里没有归档行（verbose 输出把尾部挤掉了）。改用精确匹配，"
                          "把构建输出落到日志文件后只取那一行：",
                          "execute_command",
                          {"command": ("mbt build *> mbt_build.log; "
                                       "Select-String -Path mbt_build.log "
                                       "-Pattern 'the MTA archive generated at:'"),
                           "requires_approval": False}),
            ],
        }))

        # ---- C. 同一类调用第二次仍然失败 -> 立即中止汇报（线上曾原样重发 5 次） ----
        first_fail = make_call(
            "执行构建：",
            "execute_command",
            {"command": "mbt build -l mta.yaml -d build", "requires_approval": False})
        second_fail = make_call(
            "换个参数再试一次：",
            "execute_command",
            {"command": "mbt build --timeout 600", "requires_approval": False})
        pool.append((f"RECOVER|{ws}", f"recover|{ws}|{proj}|retry-then-abort", {
            "tools": make_tools_list(codebuddy),
            "messages": [
                {"role": "system", "content": pick_system_prompt(codebuddy)},
                {"role": "user", "content": f"把 {proj} 项目构建一下"},
                first_fail,
                make_result(first_fail, "Error: unknown shorthand flag: 'l' in -l"),
                second_fail,
                make_result(second_fail, "Error: unknown flag: --timeout"),
                {"role": "assistant", "content": (
                    "❌ **构建中止**：`mbt build` 连续两次因命令参数不合法失败（`-l` / `--timeout` 都不是有效参数）。"
                    "按流程要求已停止，不再重复执行同一命令；请确认后在项目根目录直接用 `mbt build` 重试。")},
            ],
        }))

        # ---- D. 修好之后继续：从正确返回里取出归档路径并部署 ----
        fixed_call = make_call(
            "改用精确匹配从构建日志里取归档行：",
            "execute_command",
            {"command": "mbt build 2>&1 | Select-String -Pattern 'the MTA archive generated at:'",
             "requires_approval": False})
        pool.append((f"RECOVER|{ws}", f"recover|{ws}|{proj}|extract-then-deploy", {
            "tools": make_tools_list(codebuddy),
            "messages": [
                {"role": "system", "content": pick_system_prompt(codebuddy)},
                {"role": "user", "content": f"构建完成后继续把 {proj} 部署到 BTP"},
                fixed_call,
                make_result(fixed_call, f"[10:14:30] INFO the MTA archive generated at: mta_archives\\{mtar_name}"),
                make_call(f"日志输出了归档路径 `mta_archives\\{mtar_name}`，直接用这个相对路径执行部署：",
                          "execute_command",
                          {"command": f"cf deploy mta_archives\\{mtar_name} -f",
                           "requires_approval": True}),
            ],
        }))

    return pool


def build_field_semantics_samples():
    """工具字段语义正例：pattern 只写文件名通配符，目录一律放 target_directory。"""
    pool = []
    codebuddy = _codebuddy_profile()

    for rule in RULES:
        ws = rule["workspace"]
        proj = rule["project"]
        root = project_root(rule)
        mta_file = rule["mta"]
        archives_dir = f"{root}\\mta_archives"

        items = [
            {
                "users": [
                    f"在 {ws} 工作区里找一下所有 mta 配置文件",
                    "列一下工作区里有哪些 mta yaml",
                ],
                "args": {"target_directory": root, "pattern": "mta*.yaml", "recursive": True},
                "result": json.dumps([f"{root}\\{mta_file}", f"{root}\\mta\\{mta_file}"], ensure_ascii=False),
                "final": (f"工作区里共有 2 个候选：`{root}\\{mta_file}` 与 `{root}\\mta\\{mta_file}`，"
                          f"{proj} 用的是 `{mta_file}`。"),
            },
            {
                "users": [
                    "构建产物归档包在哪？去 mta_archives 里找一下",
                    "在 mta_archives 目录里找 .mtar 归档",
                ],
                "args": {"target_directory": archives_dir, "pattern": "*.mtar", "recursive": False},
                "result": f"{archives_dir}\\{proj.lower()}-mpb-d_1.0.0.mtar",
                "final": f"`mta_archives` 下的归档是 `{proj.lower()}-mpb-d_1.0.0.mtar`。",
            },
            {
                "users": [
                    f"确认一下 {mta_file} 在项目哪一层",
                    f"{proj} 的 MTA 文件放在根目录还是 mta 子目录？",
                ],
                "args": {"target_directory": root, "pattern": mta_file, "recursive": True},
                "result": f"{root}\\mta\\{mta_file}",
                "final": f"`{mta_file}` 在 `mta` 子目录下（完整路径 `{root}\\mta\\{mta_file}`）。",
            },
        ]

        for item_index, item in enumerate(items):
            for phrase_index, user_text in enumerate(item["users"]):
                calls = [(f"用 `pattern` 写通配符、目录放 `target_directory` 来查：",
                          "search_file", item["args"], item["result"])]
                system_prompt = pick_system_prompt(codebuddy)
                pool.append((f"FIELDS|{ws}", f"fields|{ws}|{proj}|{item_index}|{phrase_index}|flow",
                             {"tools": make_tools_list(codebuddy),
                              "messages": call_sequence_messages(system_prompt, user_text, calls,
                                                                  item["final"])}))
                pool.append((f"FIELDS|{ws}", f"fields|{ws}|{proj}|{item_index}|{phrase_index}|cold",
                             {"tools": make_tools_list(codebuddy),
                              "messages": call_sequence_messages(system_prompt, user_text, calls,
                                                                  final_text=None,
                                                                  include_results=False)}))

    return pool


# ============================================================
# 9. 知识库问答样本（不触发 tool_call）
# ============================================================

def build_qa_samples():
    pool = []

    code_templates = [
        "{ws} 工作区下 {proj} 项目的 {env} 登录 code 是多少？",
        "查一下 {ws} 的 {proj} {env} 对应的登录 code",
        "请问在 {ws} 里的 {proj}（{env}）登录 code 是什么",
        "{ws} {proj} {env} 的登陆 code 是什么？",
    ]
    mta_templates = [
        "{ws} 工作区下 {proj} 项目的 {env} 部署 mta 文件是什么？",
        "请问 {ws} 的 {proj} {env} 用的 mta 文件叫什么？",
        "{ws} 工作区中 {proj} 的 {env} 对应哪个 mta yaml？",
        "{ws} {proj} {env} 的 mta 部署配置文件名？",
    ]

    for rule in RULES:
        ws = rule["workspace"]
        proj = rule["project"]
        code = rule["code"]
        mta = rule["mta"]
        key = rule_key(rule)
        env_names = ENV_SYNONYMS[rule["env"]]

        for index, tmpl in enumerate(code_templates):
            profile = pick_profile()
            pool.append((key, f"qa-code|{key}|{index}", {
                "tools": make_tools_list(profile),
                "messages": [
                    {"role": "system", "content": pick_system_prompt(profile)},
                    {"role": "user", "content": tmpl.format(ws=ws, proj=proj, env=rng.choice(env_names))},
                    {"role": "assistant",
                     "content": f"{ws} 工作区下 {proj} 项目 {env_names[0]}的登录 code 是 `{code}`。"},
                ],
            }))

        for index, tmpl in enumerate(mta_templates):
            profile = pick_profile()
            pool.append((key, f"qa-mta|{key}|{index}", {
                "tools": make_tools_list(profile),
                "messages": [
                    {"role": "system", "content": pick_system_prompt(profile)},
                    {"role": "user", "content": tmpl.format(ws=ws, proj=proj, env=rng.choice(env_names))},
                    {"role": "assistant",
                     "content": f"{ws} 工作区下 {proj} 项目 {env_names[0]}的部署 MTA 文件是 `{mta}`。"},
                ],
            }))

    # 反向查询：code -> 归属
    code_to_rules = defaultdict(list)
    for rule in RULES:
        code_to_rules[rule["code"]].append(rule)

    for code, matched in code_to_rules.items():
        info_lines = []
        for rule in matched:
            info_lines.append(
                f"- **{rule['workspace']} 工作区** 的 **{rule['project']} 项目**"
                f"（{ENV_DISPLAY[rule['env']]}，MTA: `{rule['mta']}`）"
            )
        response = f"登录 code `{code}` 对应的项目与环境如下：\n" + "\n".join(info_lines)

        for index, query in enumerate([
            f"登录 code `{code}` 对应的是哪个项目和环境？",
            f"code 是 `{code}` 的环境有哪些？",
            f"谁使用登录 code `{code}`？",
        ]):
            profile = pick_profile()
            pool.append((f"CODEquery|{code}", f"qa-rev|{code}|{index}", {
                "tools": make_tools_list(profile),
                "messages": [
                    {"role": "system", "content": pick_system_prompt(profile)},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response},
                ],
            }))

    # 全局列表（跨工作区对比样本已随 CMP 工作区一并移除）
    global_answers = {
        "列出 MPB 工作区下的所有项目和环境配置": (
            "MPB 工作区包含以下 3 个项目的开发环境配置：\n"
            "1. **HC 项目**：登录 code 为 `163-d-hc`，MTA 文件为 `mta-develop-hc.yaml`\n"
            "2. **CPT 项目**：登录 code 为 `162-d-cpt`，MTA 文件为 `mta-develop.yaml`\n"
            "3. **PT 项目**：登录 code 为 `162-d-pt`，MTA 文件为 `mta-develop-pt.yaml`"
        ),
    }
    for index, (query, answer) in enumerate(global_answers.items()):
        profile = pick_profile()
        pool.append(("GLOBAL", f"qa-list|{index}", {
            "tools": make_tools_list(profile),
            "messages": [
                {"role": "system", "content": pick_system_prompt(profile)},
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer},
            ],
        }))

    return pool


# ============================================================
# 10. 按分组切分（杜绝同一条轨迹跨 train / val）
# ============================================================

def split_by_group(pool, val_ratio=0.15):
    """先按业务 key 分层，再在层内按 group_id 整组切分。

    这样既保证同一条轨迹（含它的所有前缀与失败分支）不会同时出现在
    train 和 val，也保证每个业务规则都会出现在验证集里。

    返回 (train, val, train_group_ids, val_group_ids)
    """
    buckets = defaultdict(lambda: defaultdict(list))
    for key, group_id, sample in pool:
        buckets[key][group_id].append(sample)

    train, val = [], []
    train_gids, val_gids = set(), set()

    for key, groups in buckets.items():
        group_ids = list(groups)
        rng.shuffle(group_ids)
        if len(group_ids) <= 1:
            n_val = 0
        else:
            n_val = max(1, int(round(len(group_ids) * val_ratio)))
        for gid in group_ids[:n_val]:
            val.extend(groups[gid])
            val_gids.add(gid)
        for gid in group_ids[n_val:]:
            train.extend(groups[gid])
            train_gids.add(gid)

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val, train_gids, val_gids


# ============================================================
# 11. 生成后自检
# ============================================================
# 这一节检查的都是「会静默毁掉训练」的坑：工具名不在菜单里、必填参数缺失、
# 参数未在 schema 里声明、tool 结果与调用配对错位、同组内工具签名不一致。
# 这些错误不会让脚本崩，只会让模型训出来是废的，所以必须在生成完立刻拦住。

_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_structure(rows, label):
    """消息序列的结构检查：role 合法、有 system 和 user、最后一条必须是 assistant。"""
    issues = []
    for i, row in enumerate(rows):
        messages = row.get("messages") or []
        if not messages:
            issues.append(f"{label}#{i} 没有 messages")
            continue
        bad_roles = [m.get("role") for m in messages if m.get("role") not in _ALLOWED_ROLES]
        if bad_roles:
            issues.append(f"{label}#{i} 出现非法 role: {bad_roles}")
        if messages[0].get("role") != "system":
            issues.append(f"{label}#{i} 第一条不是 system")
        if not any(m.get("role") == "user" for m in messages):
            issues.append(f"{label}#{i} 没有 user 消息")
        if messages[-1].get("role") != "assistant":
            issues.append(f"{label}#{i} 最后一条不是 assistant（模型没有监督目标）")
    return issues


def validate_tool_calls(rows, label):
    """tool_call 必须与样本自带的工具菜单自洽。"""
    issues = []
    for i, row in enumerate(rows):
        schemas = {t["function"]["name"]: t["function"] for t in row.get("tools") or []}
        calls = {}
        for m in row["messages"]:
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                if fn["name"] not in schemas:
                    issues.append(f"{label}#{i} 工具名不在菜单中: {fn['name']}")
                    continue
                calls[tc["id"]] = fn["name"]
                spec = schemas[fn["name"]]["parameters"]
                try:
                    args = json.loads(fn["arguments"])
                except Exception as exc:
                    issues.append(f"{label}#{i} arguments 不是合法 JSON: {exc}")
                    continue
                for req in spec.get("required", []):
                    if req not in args:
                        issues.append(f"{label}#{i} {fn['name']} 缺少必填参数 {req}，"
                                      f"实参 {sorted(args)}")
                for key in args:
                    if key not in spec.get("properties", {}):
                        issues.append(f"{label}#{i} {fn['name']} 参数未在 schema 声明: {key}")
            if m.get("role") == "tool":
                if m.get("name") not in schemas:
                    issues.append(f"{label}#{i} tool 结果的 name 不在菜单中: {m.get('name')}")
                if m.get("tool_call_id") not in calls:
                    issues.append(f"{label}#{i} tool 结果找不到对应调用: {m.get('tool_call_id')}")
                elif m["name"] != calls[m["tool_call_id"]]:
                    issues.append(f"{label}#{i} tool 结果 name 与调用不一致: "
                                  f"{m['name']} vs {calls[m['tool_call_id']]}")
    return issues


def validate_group_consistency(pool):
    """同一个轨迹组内必须共用同一套工具菜单。

    前缀展开出来的样本共享同一段上下文，如果组内工具签名不同，
    同一条轨迹的历史会自相矛盾（上一轮叫 execute_command，下一轮叫 Bash）。
    """
    issues = []
    seen = {}
    for _key, group_id, row in pool:
        fingerprint = tuple(sorted(t["function"]["name"] for t in row.get("tools") or []))
        if group_id in seen and seen[group_id] != fingerprint:
            issues.append(f"轨迹组 {group_id} 内工具菜单不一致")
        seen[group_id] = fingerprint
    return issues


def count_used_tools(rows):
    """按样本统计「真正发出过调用」的工具（同一条样本里同一工具多次只计一次）。"""
    counts = defaultdict(int)
    for row in rows:
        used = {tc["function"]["name"]
                for m in row["messages"]
                for tc in (m.get("tool_calls") or [])}
        for name in used:
            counts[name] += 1
    return counts


def validate_tool_coverage(rows, label, min_samples=MIN_TOOL_COVERAGE):
    """agent 提示词里声明的每条工具，都必须有足够多的真实调用样本。

    这里以 CodeBuddy 档位的工具菜单为准 —— 那正是线上 agent 注入的提示词。
    覆盖不足说明「有 schema、没示范」，模型只能猜参数怎么填，
    到了真实环境就只会硬凑 shell 命令（这正是 read_file/replace_in_file 的线上表现）。
    """
    menu = [t["function"]["name"] for t in CODEBUDDY_TOOL_MENU]
    counts = count_used_tools(rows)

    issues = []
    print(f"\n[{label}] 工具覆盖自检（菜单 {len(menu)} 条，阈值 {min_samples} 条样本）:")
    for name in menu:
        got = counts.get(name, 0)
        mark = "OK" if got >= min_samples else "!!"
        print(f"    [{mark}] {name:<20} {got:5d} 条样本发出过调用")
        if got < min_samples:
            issues.append(f"[{label}] 工具 {name} 只有 {got} 条调用样本（要求 ≥ {min_samples}）")

    others = {k: v for k, v in sorted(counts.items(), key=lambda kv: -kv[1]) if k not in set(menu)}
    if others:
        print(f"    其它档位工具: {others}")
    return issues


def count_command_targets(rows, fragment):
    """统计 fragment 作为「监督目标」出现的样本数（= 最后一条 assistant 的工具调用里含它）。

    注意只统计监督位置：历史上下文里出现过多少次不算数 —— 那教的是「复述」，
    不是「接到指令后生成这条命令」。
    """
    hits = 0
    for row in rows:
        for tc in row["messages"][-1].get("tool_calls") or []:
            if fragment in tc["function"]["arguments"]:
                hits += 1
                break
    return hits


def validate_command_exposure(rows, label, min_samples=MIN_KEY_COMMAND_TARGET):
    """关键命令必须以「监督目标」的身份出现足够多次。

    实测教训：旧数据里 `mbt build` 在 159 处出现过，但真正要求模型生成的只有 23 条，
    结果模型学会了动作序列却记不住命令字面量（线上表现：瞎试 -l/-f/-d、登录 code 幻觉）。
    """
    issues = []
    print(f"\n[{label}] 关键命令目标曝光（阈值 {min_samples} 条）：")
    for tag, fragment in KEY_COMMANDS:
        got = count_command_targets(rows, fragment)
        mark = "OK" if got >= min_samples else "!!"
        print(f"    [{mark}] {tag:<6} `{fragment}`  {got:5d} 条样本把它当监督目标")
        if got < min_samples:
            issues.append(f"[{label}] 关键命令 {fragment} 作为监督目标只有 {got} 条"
                          f"（要求 ≥ {min_samples}）")
    return issues


def estimate_tokens(text):
    """粗略估算 token 数（中文 1 token/字，其余 0.3 token/字符）。
    真实值以 main.py 里 tokenizer 实测为准，这里只用来快速发现异常长的样本。"""
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", text))
    return int(cjk + (len(text) - cjk) * 0.30)


def estimate_sample_tokens(row):
    parts = [json.dumps(t, ensure_ascii=False) for t in row.get("tools") or []]
    for m in row["messages"]:
        parts.append(str(m.get("content") or ""))
        for tc in m.get("tool_calls") or []:
            parts.append(json.dumps(tc, ensure_ascii=False))
    return estimate_tokens("\n".join(parts))


MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "4352"))


def load_tokenizer_for_check():
    """尝试加载基座分词器以实测 Tokenizer 长度。"""
    try:
        import config
        from transformers import AutoTokenizer
        model_name = getattr(config, "BASE_MODEL_NAME", "Qwen/Qwen3-8B")
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"[提示] 未能加载 Tokenizer 实测精确 Token 长度 ({e})，保留启发式估算。")
        return None


def check_max_length(rows, label, tokenizer, max_length=MAX_LENGTH):
    """用 tokenizer 实测每条样本真实长度，确认没有任何一条会被静默截断。

    按「工具菜单」分组报告最长值——不同 harness 档位的工具定义大小差别很大
    （9 条工具 ≈1800 token，单工具档位 ≈120 token），分组看才知道是谁把样本撑长的。
    超限直接计入 issues 拦截：截断会把尾部的最终汇报砍掉，训出来的模型是废的。
    """
    issues = []
    if tokenizer is None:
        return issues

    def render(row):
        return tokenizer.apply_chat_template(
            row["messages"],
            tools=row.get("tools"),
            tokenize=False,
            add_generation_prompt=False,
        )

    print(f"\n[{label}] 真实 Token 长度自检（按 Tokenizer 实测，阈值 {max_length}）:")
    longest_by_menu = {}
    for row in rows:
        length = len(tokenizer(render(row))["input_ids"])
        fingerprint = tuple(sorted(t["function"]["name"] for t in (row.get("tools") or [])))
        if fingerprint not in longest_by_menu or length > longest_by_menu[fingerprint][0]:
            longest_by_menu[fingerprint] = (length, row)

    if not longest_by_menu:
        print(f"[{label}] 数据集为空，跳过长度检查")
        return issues

    overall_length, longest_row = max(longest_by_menu.values(), key=lambda item: item[0])
    for fingerprint, (length, _row) in sorted(longest_by_menu.items(), key=lambda kv: -kv[1][0]):
        tag = fingerprint[0] if len(fingerprint) == 1 else f"{len(fingerprint)} 条工具"
        print(f"    {length:6d} tokens   工具档位: {tag}")

    last_message = longest_row["messages"][-1]
    last_kind = "工具调用" if last_message.get("tool_calls") else "文本收尾"
    print(f"[{label}] 全量最长 {overall_length} tokens / max_length {max_length}"
          f"（该样本最后一回合: {last_kind}）")

    if overall_length > max_length:
        issues.append(
            f"[{label}] 最长样本 {overall_length} tokens 超过 max_length {max_length}，"
            f"尾部会被静默截断（很可能正好砍掉最终汇报），请调大 MAX_LENGTH。"
        )
    else:
        print(f"[{label}] 长度检查通过，余量 {max_length - overall_length} tokens")

    return issues


def summarize(rows, label):
    """打印工具档位分布 / 监督目标分布 / 长度估算"""
    menu_counter = defaultdict(int)
    for row in rows:
        menu_counter[tuple(sorted(t["function"]["name"] for t in row.get("tools") or []))] += 1

    print(f"\n[{label}] 工具档位分布:")
    for menu, count in sorted(menu_counter.items(), key=lambda kv: -kv[1]):
        if len(menu) == 1:
            tag = menu[0]
        else:
            tag = f"{len(menu)} 条工具（{menu[0] if menu else '?'} 等）"
        print(f"    {count:5d} 条  {tag}")

    behavior = defaultdict(int)
    for row in rows:
        messages = row["messages"]
        last = messages[-1]
        has_prior_tool = any(m.get("role") == "tool" for m in messages[:-1])
        if has_prior_tool and last.get("tool_calls"):
            behavior["拿到工具结果后继续调下一个工具"] += 1
        elif has_prior_tool:
            behavior["拿到工具结果后输出文本收尾"] += 1
        elif last.get("tool_calls"):
            behavior["冷启动首轮直接发工具调用"] += 1
        else:
            behavior["纯文本问答（不调工具）"] += 1

    print(f"[{label}] 监督目标分布:")
    for name, count in sorted(behavior.items(), key=lambda kv: -kv[1]):
        print(f"    {count:5d} 条  {name}")

    sizes = [estimate_sample_tokens(r) for r in rows]
    if sizes:
        print(f"[{label}] 启发式估算长度: 最长 {max(sizes)} / 平均 {sum(sizes) / len(sizes):.0f} tokens")


def run_post_generation_checks(train_data, val_data, train_file, val_file, pool):
    """生成后的端到端自检，返回问题列表（空 = 通过）。

    刻意把刚写盘的文件**重新读回来**再校验一遍：这样能顺带发现序列化往返的问题
    （编码、非法 JSON、字段丢失），而不只是检查内存里的对象。
    """
    print("=" * 64)
    print("生成后自检（回读 train.jsonl / val.jsonl）")
    print("=" * 64)

    issues = []

    reloaded_train = read_jsonl(train_file)
    reloaded_val = read_jsonl(val_file)
    if len(reloaded_train) != len(train_data):
        issues.append(f"train.jsonl 回读条数不符: 内存 {len(train_data)} / 磁盘 {len(reloaded_train)}")
    if len(reloaded_val) != len(val_data):
        issues.append(f"val.jsonl 回读条数不符: 内存 {len(val_data)} / 磁盘 {len(reloaded_val)}")

    issues += validate_structure(reloaded_train, "train")
    issues += validate_structure(reloaded_val, "val")
    issues += validate_tool_calls(reloaded_train, "train")
    issues += validate_tool_calls(reloaded_val, "val")
    issues += validate_group_consistency(pool)
    # agent 提示词里声明的每条工具都必须有示范：train 用阈值卡死，val 只打印（组切分后条数天然少）
    issues += validate_tool_coverage(reloaded_train, "train")
    validate_tool_coverage(reloaded_val, "val", min_samples=0)
    # 关键命令必须作为「监督目标」出现足够多次，避免「见过很多次但没当过答案」
    issues += validate_command_exposure(reloaded_train, "train")
    validate_command_exposure(reloaded_val, "val", min_samples=0)

    # 精确 Tokenizer 长度自检（原在 main.py，现移至此处）
    tokenizer = load_tokenizer_for_check()
    if tokenizer:
        issues += check_max_length(reloaded_train, "train", tokenizer, MAX_LENGTH)
        check_max_length(reloaded_val, "val", tokenizer, MAX_LENGTH)

    rows = reloaded_train + reloaded_val
    print(f"\n回读样本总数: {len(rows)}  (train {len(reloaded_train)} / val {len(reloaded_val)})")
    summarize(reloaded_train, "train")
    summarize(reloaded_val, "val")


    print("\n" + "-" * 64)
    if issues:
        print(f"自检结果: 不通过，共 {len(issues)} 个问题")
        for item in issues[:30]:
            print("  -", item)
        if len(issues) > 30:
            print(f"  ... 另有 {len(issues) - 30} 个问题未展示")
    else:
        print("自检结果: 通过")
    print("-" * 64)
    return issues


# ============================================================
# 12. 入口
# ============================================================

def generate_dataset():
    chain_pool = build_deploy_chain_samples()
    login_pool = build_login_action_samples()
    devops_pool = build_devops_action_samples()
    recovery_pool = build_recovery_samples()
    fields_pool = build_field_semantics_samples()
    explore_pool = build_explore_samples()
    edit_pool = build_edit_samples()
    write_pool = build_write_file_samples()
    qa_pool = build_qa_samples()

    print("原始生成:")
    print(f"  - 部署流水线前缀展开（多历史窗口）+ 失败分支: {len(chain_pool)} 条 "
          f"({len({gid for _, gid, _ in chain_pool})} 个轨迹组)")
    print(f"  - 登录 Action 样本（含失败与 code 写错自纠）: {len(login_pool)} 条")
    print(f"  - DevOps 单步动作样本（×{STEP_REPEAT} 重复采样）: {len(devops_pool)} 条")
    print(f"  - 恢复样本（字段纠错 / 归档行取不到 / 二次失败即中止）: {len(recovery_pool)} 条")
    print(f"  - 字段语义正例（search_file 的 pattern/target_directory）: {len(fields_pool)} 条")
    print(f"  - 工程探查样本（list_dir/search_content/read_lints）: {len(explore_pool)} 条")
    print(f"  - 文件编辑样本（read_file/replace_in_file）: {len(edit_pool)} 条")
    print(f"  - 文件写入样本（write_to_file 规范调用）: {len(write_pool)} 条")
    print(f"  - 知识库问答样本: {len(qa_pool)} 条")

    all_pool = (chain_pool + login_pool + devops_pool + recovery_pool + fields_pool
                + explore_pool + edit_pool + write_pool + qa_pool)
    all_gids = {gid for _, gid, _ in all_pool}
    train_data, val_data, train_gids, val_gids = split_by_group(all_pool)

    # 泄漏自检：同一个轨迹组绝不能同时落到 train 和 val
    leak = train_gids & val_gids
    if leak:
        raise RuntimeError(f"检测到轨迹组跨集合泄漏，共 {len(leak)} 组：{sorted(leak)[:5]} ...")

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

    total = len(train_data) + len(val_data)
    chain_ratio = len(chain_pool) / total * 100 if total else 0

    print("=" * 64)
    print("SAP BTP Multi-Turn Agent 前缀展开数据集生成完成")
    print(f"总样本数: {total} 条 / 轨迹组: {len(all_gids)} 个")
    print(f"  - 链式与失败样本: {len(chain_pool)} 条 (占比 {chain_ratio:.1f}%)")
    print("划分结果（按轨迹组切分，已验证无泄漏）:")
    print(f"  - 训练集 (train.jsonl): {len(train_data)} 条 / {len(train_gids)} 组")
    print(f"  - 验证集 (val.jsonl)  : {len(val_data)} 条 / {len(val_gids)} 组")
    print(f"输出目录: {data_dir}")
    print("=" * 64)

    print("\n[提醒] 工具层采用「固定任务 + 随机化外壳」策略：")
    for p in TOOL_PROFILES:
        print(f"        {p['id']:>18s}  权重 {p['weight']:<2d}  工具 {len(p['tools'])} 条"
              f"  命令工具名 `{shell_tool_name(p)}`")
    print("        改权重只需调整 TOOL_PROFILES 里的 weight。")
    print("[提醒] 每步要执行的命令（Invoke-RestMethod / 查找并复制 mta / mbt build / "
          "cf deploy / Remove-Item）在所有档位里是恒定的 —— 那才是要学的技能。")
    print("[提醒] 「复制 mta」按方案 A 展开：CodeBuddy 走两步（search_file 查找 . 相对路径 -> "
          "Copy-Item 直接覆盖复制为 .\\mta.yaml；未匹配到则报错中止）；"
          "其余档位用一条 Get-ChildItem 递归查找 + Copy-Item 命令。")

    print("[提醒] 清理同样按档位展开：CodeBuddy 用 delete_file 删 mta.yaml + "
          "execute_command 删目录与构建产物。")
    print(f"[提醒] 覆盖自检要求 agent 菜单里 9 条工具每条至少有 {MIN_TOOL_COVERAGE} 条调用样本；"
          "不足会被自检拦下（见 validate_tool_coverage）。")
    print("[提醒] CodeBuddy 的 execute_command 没有 timeout 参数，"
          "超时要求靠助手旁白表达；Bash 档位则会带 timeout=600000（毫秒）。")

    # ---- 生成完立刻自检一遍 ----
    issues = run_post_generation_checks(train_data, val_data, train_file, val_file, all_pool)
    return len(issues)


if __name__ == "__main__":
    problem_count = generate_dataset()
    if problem_count:
        raise SystemExit(f"生成后自检发现 {problem_count} 个问题，先修数据再拿去训练。")
    print("\n下一步: python main.py  （数据生成阶段已完成分词器精确长度实测，训练可直接启动）")
