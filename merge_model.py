import os
import ssl
import urllib3
import dataclasses

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 兼容内网代理/VPN环境下的 SSL 证书拦截
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

import httpx
from huggingface_hub.utils import set_client_factory, _http
def unverified_client_factory() -> httpx.Client:
    return httpx.Client(
        verify=False,
        event_hooks={"request": [_http.hf_request_event_hook]},
        follow_redirects=True,
        timeout=None,
    )
set_client_factory(unverified_client_factory)

from huggingface_hub import file_download
orig_get_metadata = file_download.get_hf_file_metadata
def patched_get_metadata(*args, **kwargs):
    meta = orig_get_metadata(*args, **kwargs)
    if meta.commit_hash is None:
        meta = dataclasses.replace(meta, commit_hash="1cfa9a7208912126459214e8b04321603b3df60c")
    return meta
file_download.get_hf_file_metadata = patched_get_metadata

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import config
import torch

def main():
    model_name = getattr(config, "BASE_MODEL_NAME", "Qwen/Qwen3-4B")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
        )
    except Exception as exc:
        print(f"[提示] AutoModelForCausalLM 加载异常，尝试 AutoModelForImageTextToText: {exc}")
        from transformers import AutoModelForImageTextToText
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lora_model = PeftModel.from_pretrained(
        base_model,
        str(config.MODEL_PATH),
    )

    merged_model = lora_model.merge_and_unload()  # type: ignore

    merged_model.save_pretrained(str(config.MERGED_PATH))
    tokenizer.save_pretrained(str(config.MERGED_PATH))
    print(f"完整合并模型已保存至: {config.MERGED_PATH}")


if __name__ == "__main__":
    main()
