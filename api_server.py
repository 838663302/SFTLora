import os
import re
import uuid
import json
import time
import argparse
import asyncio
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
MODEL_PATH = r"C:\Users\azt1szh\Desktop\set\checkpoints\ov_model"


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
        "KV_CACHE_PRECISION": "u8"
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


def parse_tool_calls(text: str):
    """解析 Qwen 文本中的 <tool_call> 标签并转化为 OpenAI 标准 tool_calls 结构"""
    tool_calls = []
    # 移除 think 标签
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text_clean, re.DOTALL)
    clean_content = re.sub(pattern, "", text_clean, flags=re.DOTALL).strip()

    for idx, m in enumerate(matches):
        try:
            call_json = json.loads(m.strip())
            tool_name = call_json.get("name", "")
            args = call_json.get("arguments", {})
            args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
            tool_calls.append({
                "index": idx,
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": args_str
                }
            })
        except Exception as e:
            print(f"[ToolCall Parse Error]: {e} on raw content: {m}")

    for stop in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        clean_content = clean_content.replace(stop, "").strip()

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
    print(f"\n[收到请求] 消息数: {len(req.messages)}, 工具数: {tool_count}, 流式模式: {req.stream}")

    if req.tools:
        tool_names = []
        for t in req.tools:
            if isinstance(t, dict):
                fn = t.get("function", {})
                name = fn.get("name") or t.get("name") or "unknown"
                tool_names.append(name)
            else:
                tool_names.append(str(t))
        print(f"👉 包含工具列表 ({tool_count} 个): {tool_names}")

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

    # 渲染模版
    try:
        text = tokenizer.apply_chat_template(
            conversation=msgs,
            tools=req.tools if req.tools else None,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        print(f"[apply_chat_template Error]: {e}, fallback without tools template")
        text = tokenizer.apply_chat_template(
            conversation=msgs,
            tokenize=False,
            add_generation_prompt=True
        )

    inputs = tokenizer(text, return_tensors="pt")
    created_time = int(time.time())
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    max_tokens = req.max_tokens or 512

    # 执行推理 (加锁保证线程安全)
    async with generate_lock:
        raw_reply, prompt_tokens, completion_tokens = await asyncio.to_thread(_sync_generate, inputs, max_tokens)

    # 如果文本中包含 <|im_end|>，只截取第一个 <|im_end|> 之前的内容
    if "<|im_end|>" in raw_reply:
        raw_reply = raw_reply.split("<|im_end|>")[0]

    clean_content, tool_calls = parse_tool_calls(raw_reply)

    if tool_calls:
        print(f"[ToolCalls 解析成功]: {json.dumps(tool_calls, ensure_ascii=False)}")
    else:
        print(f"[回复生成]: {clean_content}")

    # 1. 流式响应 (Stream = True)
    if req.stream:
        async def event_generator():
            if tool_calls:
                # 针对工具调用的标准 OpenAI 流式 Chunk
                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": clean_content if clean_content else None,
                                "tool_calls": tool_calls
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
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

    load_model(device=args.device)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
