import os
import re
import uuid
import json
import time
import argparse
import asyncio
from pathlib import Path
from threading import Thread
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer

app = FastAPI(title="Local OpenVINO LLM Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None
current_device = "GPU"
# 默认指向微调好的 4B OpenVINO 模型目录，支持环境变量 OV_MODEL_PATH 覆盖
MODEL_PATH = os.environ.get("OV_MODEL_PATH", r"C:\Users\azt1szh\Desktop\set\checkpoints\ov_model")

import config

# 工具调用本地智能纠错开关（仅在 config.py 中配置）
ENABLE_TOOL_CALL_CORRECTION = getattr(config, "ENABLE_TOOL_CALL_CORRECTION", False)


def load_train_tool_names() -> set:
    """动态获取训练集中的所有合法工具名称"""
    train_file = Path(__file__).resolve().parent / "data" / "train.jsonl"
    tools = set()
    if train_file.exists():
        try:
            with open(train_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    for t in item.get("tools", []):
                        if isinstance(t, dict):
                            fn = t.get("function", {})
                            name = fn.get("name") or t.get("name")
                            if name:
                                tools.add(name)
        except Exception as e:
            print(f"[警告] 动态读取训练集工具失败: {e}")
    if not tools:
        # 兜底：训练集标准 12 个工具
        tools = {
            "list_dir", "search_file", "search_content", "read_file",
            "read_lints", "replace_in_file", "write_to_file",
            "execute_command", "delete_file",
            "Bash", "terminal", "run_terminal_cmd"
        }
    return tools


TRAIN_TOOL_NAMES = load_train_tool_names()
print(f"🛠️ [已加载训练集支持工具 ({len(TRAIN_TOOL_NAMES)} 个)]: {sorted(list(TRAIN_TOOL_NAMES))}")


def load_model(device: str = "GPU"):
    global model, tokenizer, current_device
    current_device = device.upper()
    print("\n" + "=" * 60)
    print(f"正在将模型载入硬件设备: [{current_device}] (Intel Arc / NPU / CPU)...")
    print(f"模型路径: {MODEL_PATH}")
    print("=" * 60 + "\n")

    ov_config = {
        "PERFORMANCE_HINT": "LATENCY",
        "GPU_ENABLE_LARGE_ALLOCATIONS": "YES",
        "KV_CACHE_PRECISION": "u8",
        "CACHE_DIR": os.path.join(MODEL_PATH, "ov_cache"),
    } if current_device == "GPU" else {"PERFORMANCE_HINT": "LATENCY"}

    try:
        model = OVModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device=current_device,
            ov_config=ov_config
        )
    except Exception as e:
        print(f"无法在 [{current_device}] 上加载模型: {e}")
        print("自动回退到 [CPU] 加载...")
        current_device = "CPU"
        model = OVModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device="CPU",
            ov_config={"PERFORMANCE_HINT": "LATENCY"}
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"\n>>> 模型已就绪！成功运行在: [{current_device}] <<<\n")


def sanitize_command(cmd: str) -> str:
    """对模型生成的终端/PowerShell命令行进行深度容错纠错与格式规范化"""
    if not cmd or not isinstance(cmd, str):
        return cmd

    repaired = cmd.strip()

    # 1. 修复 PowerShell Cmdlet 与参数黏连（如 Select-Object-Last 10 -> Select-Object -Last 10）
    # 匹配非空格、非横杠后紧跟的常见 PowerShell 参数名
    ps_param_pattern = (
        r'(?<=[^\s\-])-(last|first|destination|path|force|filter|recurse|file|'
        r'uri|method|contenttype|body|pattern|erroraction|compress|depth|'
        r'property|expandproperty|passthru|confirm|whatif)\b'
    )
    repaired = re.sub(ps_param_pattern, r' -\1', repaired, flags=re.IGNORECASE)

    # 2. 修复 Windows PowerShell 5.1 不支持的 && 语法，替换为 ;
    if "&&" in repaired and "http" not in repaired:
        repaired = re.sub(r'\s*&&\s*', ' ; ', repaired)

    # 3. 修复目录 / 路径拼写幻觉（实测多次出现 mta_archures 导致 cf deploy 找不到文件）
    repaired = re.sub(r'\bmta_arch[a-z0-9_]*\b', 'mta_archives', repaired, flags=re.IGNORECASE)

    # 4. 修复管道符或重定向符附近的常见语法错误
    repaired = re.sub(r'2>&\|', '2>&1 |', repaired)
    repaired = re.sub(r'2>&1\|', '2>&1 |', repaired)
    repaired = re.sub(r'\|\s*tail(?:\s+-n|\s+)?\s*(\d+)', r'| Select-Object -Last \1', repaired, flags=re.IGNORECASE)

    # 5. 针对 mbt build 的专属优化
    # 真实演示中，mbt build 是核心关键步。模型若带上非法参数（如 -l, --timeout, --local 等），
    # 或者缺少截断管道导致日志过大，在此统一样式保底。
    if "mbt build" in repaired:
        bad_flags = [
            r'--timeout\s+\d+', r'-l\s+\S+', r'--local', r'-f\b', r'--manifest\b',
            r'--mta\s+\S+', r'--requirement\b', r'--build\b', r'\bmta\.yaml\b',
            r'-d\s+\S+'
        ]
        mbt_clean = repaired
        for bf in bad_flags:
            mbt_clean = re.sub(bf, '', mbt_clean, flags=re.IGNORECASE)
        mbt_clean = re.sub(r'\s+', ' ', mbt_clean).strip()

        if "Select-String" in mbt_clean or "the MTA archive generated at:" in mbt_clean:
            repaired = "mbt build 2>&1 | Select-String -Pattern 'the MTA archive generated at:'"
        else:
            repaired = "mbt build 2>&1 | Select-Object -Last 10"

    # 6. 针对 cf deploy 的优化（确保带有 -f，防止交互式确认导致超时）
    if "cf deploy" in repaired:
        if not re.search(r'\s-f\b', repaired, flags=re.IGNORECASE):
            repaired = repaired.rstrip(" ;") + " -f"

    return repaired.strip()


def sanitize_arguments(tool_name: str, args: dict):
    """对工具调用的入参做统一清洗校验与智能纠错"""
    if not isinstance(args, dict):
        return

    # 1. 修复命令类字段
    for key in ["command", "cmd", "script"]:
        if key in args and isinstance(args[key], str):
            orig_cmd = args[key]
            cleaned_cmd = sanitize_command(orig_cmd)
            if cleaned_cmd != orig_cmd:
                print(f"🔧 [命令自动纠错]: '{orig_cmd}' -> '{cleaned_cmd}'")
            args[key] = cleaned_cmd

    # 2. 补全或规范化 execute_command 的 requires_approval
    if tool_name == "execute_command":
        if "requires_approval" in args:
            val = args["requires_approval"]
            if isinstance(val, str):
                args["requires_approval"] = val.lower() in ("true", "1")
        else:
            cmd = args.get("command", "")
            if any(k in cmd for k in ["Invoke-RestMethod", "cf deploy", "Remove-Item"]):
                args["requires_approval"] = True
            else:
                args["requires_approval"] = False

    # 3. 修复文件路径类字段
    for path_key in ["filePath", "target_file", "path"]:
        if path_key in args and isinstance(args[path_key], str):
            orig_path = args[path_key]
            cleaned_path = re.sub(r'\bmta_arch[a-z0-9_]*\b', 'mta_archives', orig_path, flags=re.IGNORECASE)
            cleaned_path = re.sub(r'^[.]+[/\\]+', r'.\\', cleaned_path)
            if cleaned_path != orig_path:
                print(f"🔧 [路径自动纠错]: '{orig_path}' -> '{cleaned_path}'")
            args[path_key] = cleaned_path


def sanitize_command_in_text(text: str) -> str:
    """修复输出给用户的纯文本/Markdown中包含的命令笔误"""
    if not text:
        return text
    text = re.sub(
        r'(?<=[^\s\-])-(last|first|destination|path|force|filter|recurse|file|'
        r'uri|method|contenttype|body|pattern|erroraction|compress|depth|'
        r'property|expandproperty|passthru|confirm|whatif)\b',
        r' -\1',
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r'\bmta_arch[a-z0-9_]*\b', 'mta_archives', text, flags=re.IGNORECASE)
    return text


def robust_parse_tool_call(m: str, enable_correction: bool = ENABLE_TOOL_CALL_CORRECTION):
    """容错解析可能带有未转义引号的脏 JSON 工具调用，并按需注入智能容错修复"""
    tool_name = ""
    arguments = None

    try:
        call_json = json.loads(m.strip())
        tool_name = call_json.get("name", "")
        # 如果未开启本地纠错，直接原样使用模型给出的 name 和 arguments，不做任何本地修改
        if not enable_correction:
            return tool_name, call_json.get("arguments", {})
        if tool_name != "Invoke-RestMethod":
            arguments = call_json.get("arguments", {})
    except Exception:
        pass

    # 正则提取 name
    if not tool_name:
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', m)
        tool_name = name_match.group(1) if name_match else ""

    # 针对模型幻觉：将命令行直接当成 tool_name 的情况自动修复为 execute_command（仅在开启纠错时）
    if enable_correction and tool_name == "Invoke-RestMethod":
        code_match = re.search(r"['\"]?16[23]-d-[a-z]+['\"]?", m)
        space_code = code_match.group(0).strip("'\"") if code_match else "162-d-pt"
        repaired_cmd = (
            f"Invoke-RestMethod -Uri http://localhost:3000/space/login -Method Post "
            f"-ContentType application/json -Body (@{{space='{space_code}'}}|ConvertTo-Json)"
        )
        return "execute_command", {"command": repaired_cmd, "requires_approval": True}

    # 特别针对 execute_command 的命令字符串修复（解决 powershell 引号嵌套冲突）
    if tool_name == "execute_command" and arguments is None:
        cmd_match = re.search(r'"command"\s*:\s*"(.*?)"\s*,\s*"requires_approval"', m, re.DOTALL)
        if not cmd_match:
            cmd_match = re.search(r'"command"\s*:\s*"(.*)"\s*\}', m, re.DOTALL)
        appr_match = re.search(r'"requires_approval"\s*:\s*(true|false)', m, re.IGNORECASE)
        requires_approval = (appr_match.group(1).lower() == 'true') if appr_match else (True if enable_correction else None)
        if cmd_match:
            cmd_text = cmd_match.group(1)
            arguments = {"command": cmd_text}
            if requires_approval is not None:
                arguments["requires_approval"] = requires_approval

    # 特别针对 write_to_file 的长文本内容提取
    if tool_name == "write_to_file" and arguments is None:
        fp_match = re.search(r'"filePath"\s*:\s*"([^"]+)"', m)
        content_match = re.search(r'"content"\s*:\s*"(.*?)"\s*(?:\}\s*\}|\}\s*$|$)', m, re.DOTALL)
        if fp_match:
            c_val = content_match.group(1) if content_match else ""
            arguments = {"filePath": fp_match.group(1), "content": c_val}

    # 通用 arguments 提取
    if arguments is None:
        args_match = re.search(r'"arguments"\s*:\s*(\{.*\})', m, re.DOTALL)
        if args_match:
            try:
                arguments = json.loads(args_match.group(1))
            except Exception:
                arguments = {"raw_arguments": args_match.group(1)}
        else:
            arguments = {}

    # 如果 arguments 是字符串（例如被 json 双重转义），尝试再次 loads
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            pass

    # 深度纠错与优化：对 arguments 进行命令级与路径级智能修正（仅在开启纠错时执行）
    if enable_correction and isinstance(arguments, dict):
        sanitize_arguments(tool_name, arguments)

    return tool_name, arguments


def parse_tool_calls(text: str, enable_correction: bool = ENABLE_TOOL_CALL_CORRECTION):
    """解析 Qwen 文本中的 <tool_call> 标签并转化为 OpenAI 标准 tool_calls 结构"""
    tool_calls = []
    # 移除 think 标签
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 兼容闭合或未完全闭合的 <tool_call>
    pattern = r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)"
    matches = re.findall(pattern, text_clean, re.DOTALL)
    clean_content = re.sub(r"<tool_call>.*", "", text_clean, flags=re.DOTALL).strip()

    # 兼容兜底：如果模型未输出 <tool_call> 标签而是直接输出了合法 JSON 结构（仅在开启纠错时兜底）
    if enable_correction and not matches:
        raw_json_match = re.search(r'(\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:.*?\})', text_clean, re.DOTALL)
        if raw_json_match:
            matches = [raw_json_match.group(1)]
            clean_content = text_clean[:raw_json_match.start()].strip()

    for idx, m in enumerate(matches):
        m_str = m.strip()
        if not m_str:
            continue
        tool_name, args = robust_parse_tool_call(m_str, enable_correction=enable_correction)
        if tool_name:
            if tool_name not in TRAIN_TOOL_NAMES:
                print(f"[警告] 模型生成的工具调用 '{tool_name}' 不在训练集范围内，已过滤")
                continue
            args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
            tool_calls.append({
                "index": len(tool_calls),
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": args_str
                }
            })
        else:
            print(f"[ToolCall Parse Failed]: {m_str[:120]}...")

    for stop in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        clean_content = clean_content.replace(stop, "").strip()

    # 纯文本内容同步纠错规范化（仅在开启纠错时执行）
    if enable_correction:
        clean_content = sanitize_command_in_text(clean_content)

    return clean_content, tool_calls if tool_calls else None


class ChatMessage(BaseModel):
    role: str
    content: Optional[Any] = ""
    tool_calls: Optional[List[Any]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    class Config:
        extra = "allow"


class ChatRequest(BaseModel):
    model: Optional[str] = "qwen"
    messages: List[ChatMessage]
    tools: Optional[List[Any]] = None
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

    class Config:
        extra = "allow"


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-openvino",
                "device": current_device
            }
        ]
    }


generate_lock = asyncio.Lock()


def _sync_generate(inputs, max_new_tokens: int):
    stop_ids = [tokenizer.eos_token_id]
    for token_str in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.append(tid)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=stop_ids,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )
    gen_ids = outputs[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(gen_ids, skip_special_tokens=False), len(inputs.input_ids[0]), len(gen_ids)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    tool_count = len(req.tools) if req.tools else 0
    print(f"\n[收到请求] 消息数: {len(req.messages)}, 工具数: {tool_count}, 流式模式: {req.stream}, 本地纠错模式: {'开启' if ENABLE_TOOL_CALL_CORRECTION else '关闭(纯模型)'}")

    filtered_tools = None
    if req.tools:
        tool_names = []
        cleaned = []
        discarded = []
        for t in req.tools:
            name = ""
            if isinstance(t, dict):
                fn = t.get("function", {})
                name = fn.get("name") or t.get("name") or ""
            else:
                name = str(t)
            tool_names.append(name)

            if name in TRAIN_TOOL_NAMES:
                cleaned.append(t)
            else:
                discarded.append(name)

        print(f"👉 客户端请求工具列表 ({len(req.tools)} 个): {tool_names}")
        if discarded:
            print(f"🚫 [已过滤丢弃非训练集工具 ({len(discarded)} 个)]: {discarded}")
        print(f"✅ [最终送入模型的合法工具 ({len(cleaned)} 个)]: {[t.get('function', {}).get('name') or t.get('name') for t in cleaned]}")
        filtered_tools = cleaned if cleaned else None

    # 写入诊断日志，方便排查 CodeBuddy 请求的完整结构
    debug_log = Path(__file__).resolve().parent / "api_server_debug.log"
    try:
        with open(debug_log, "a", encoding="utf-8") as f_log:
            last_msg = req.messages[-1].content if req.messages else ""
            f_log.write(f"\n{'='*50}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 收到请求: 消息数={len(req.messages)}, 传入工具数={tool_count}, 过滤后工具数={len(filtered_tools) if filtered_tools else 0}\n")
            if req.tools:
                f_log.write(f"原始工具名: {tool_names}\n")
                if discarded:
                    f_log.write(f"被过滤工具: {discarded}\n")
            else:
                f_log.write("⚠️ 警告: 客户端没有传入任何 tools！\n")
            f_log.write(f"最后一条用户消息: {str(last_msg)[:150]}\n")
    except Exception:
        pass

    msgs = []
    for m in req.messages:
        item = {"role": m.role, "content": m.content if m.content is not None else ""}
        if m.tool_calls:
            item["tool_calls"] = m.tool_calls
        if m.tool_call_id:
            item["tool_call_id"] = m.tool_call_id
        if m.name:
            item["name"] = m.name
        msgs.append(item)

    # 渲染模版 (添加 enable_thinking=False 对齐训练集)
    try:
        text = tokenizer.apply_chat_template(
            conversation=msgs,
            tools=filtered_tools if filtered_tools else None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except Exception as e:
        print(f"[apply_chat_template Error]: {e}, fallback without tools template")
        text = tokenizer.apply_chat_template(
            conversation=msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

    inputs = tokenizer(text, return_tensors="pt")
    created_time = int(time.time())
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    max_tokens = max(req.max_tokens or 512, 2048)

    # 执行推理 (加锁保证线程安全)
    async with generate_lock:
        raw_reply, prompt_tokens, completion_tokens = await asyncio.to_thread(_sync_generate, inputs, max_tokens)

    # 如果文本中包含 <|im_end|>，只截取第一个 <|im_end|> 之前的内容
    if "<|im_end|>" in raw_reply:
        raw_reply = raw_reply.split("<|im_end|>")[0]

    clean_content, tool_calls = parse_tool_calls(raw_reply, enable_correction=ENABLE_TOOL_CALL_CORRECTION)

    if tool_calls:
        print(f"[ToolCalls 解析成功]: {json.dumps(tool_calls, ensure_ascii=False)}")
    else:
        print(f"[回复生成]: {clean_content}")

    try:
        with open(debug_log, "a", encoding="utf-8") as f_log:
            f_log.write(f"Prompt 包含 <tools>: {'<tools>' in text}\n")
            f_log.write(f"模型原始生成 (前 300 字): {raw_reply[:300]}...\n")
            f_log.write(f"解析到 tool_calls: {bool(tool_calls)}\n{'='*50}\n")
    except Exception:
        pass

    # 1. 流式响应 (Stream = True)
    if req.stream:
        async def event_generator():
            if tool_calls:
                # 针对 CodeBuddy: 若包含说明文字，先流式推送文本
                if clean_content:
                    txt_chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": clean_content
                                },
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(txt_chunk, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)

                # 再推送标准 OpenAI tool_calls chunk
                tc_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant" if not clean_content else None,
                                "tool_calls": tool_calls
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

                final_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls"
                        }
                    ]
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            else:
                # 普通文本流式分块输出
                init_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(init_chunk, ensure_ascii=False)}\n\n"

                chunk_size = 4
                for i in range(0, len(clean_content), chunk_size):
                    part = clean_content[i:i + chunk_size]
                    chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": part},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.005)

                final_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # 2. 非流式响应 (Stream = False)
    message_obj = {
        "role": "assistant",
        "content": clean_content if clean_content else None
    }
    if tool_calls:
        clean_tool_calls = [
            {
                "id": tc["id"],
                "type": tc["type"],
                "function": tc["function"]
            }
            for tc in tool_calls
        ]
        message_obj["tool_calls"] = clean_tool_calls

    return JSONResponse(content={
        "id": req_id,
        "object": "chat.completion",
        "created": created_time,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": message_obj,
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVINO Local OpenAI API Server")
    parser.add_argument("--device", type=str, default="GPU", help="Device to run on: GPU, NPU, CPU")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    print(f"🔧 [本地纠错状态 (config.ENABLE_TOOL_CALL_CORRECTION)]: {'已开启 (ENABLED)' if ENABLE_TOOL_CALL_CORRECTION else '已关闭 (DISABLED - 纯模型原生处理)'}")

    load_model(device=args.device)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
