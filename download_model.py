import os
import sys
import ssl
import urllib3

# 1. 环境变量配置
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# 2. 彻底解决公司网络/代理/VPN 下的 SSL 证书拦截报错 (CERTIFICATE_VERIFY_FAILED)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# 2.1 全局拦截 httpx（transformers 与 huggingface_hub 新版核心底层网络库）
import httpx
_orig_httpx_init = httpx.Client.__init__
def _patched_httpx_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _orig_httpx_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_httpx_init

# 2.2 配置 huggingface_hub 客户端工厂
from huggingface_hub.utils import set_client_factory, _http
def unverified_client_factory() -> httpx.Client:
    return httpx.Client(
        verify=False,
        event_hooks={"request": [_http.hf_request_event_hook]},
        follow_redirects=True,
        timeout=None,
    )
set_client_factory(unverified_client_factory)

# 2.3 兼容 requests（旧版接口或依赖库）
import requests
_orig_requests_merge = requests.Session.merge_environment_settings
def _patched_requests_merge(self, url, proxies, stream, verify, cert):
    settings = _orig_requests_merge(self, url, proxies, stream, verify, cert)
    settings["verify"] = False
    return settings
requests.Session.merge_environment_settings = _patched_requests_merge

import config
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = getattr(config, "BASE_MODEL_NAME", "Qwen/Qwen3-4B")
    print("=" * 64)
    print(f"正在预下载基座模型: {model_name}")
    print("已启用全局 SSL 证书免校验（彻底兼容企业代理、VPN 与安全拦截网关）")
    print("支持自动断点续传，文件将保存至本地 HuggingFace 缓存")
    print("=" * 64)

    try:
        # 1. 优先使用 snapshot_download 进行多线程断点续传下载
        local_path = snapshot_download(
            repo_id=model_name,
            max_workers=4,
        )
        print("\n" + "=" * 64)
        print(f"✅ 模型权重文件预下载完成！本地缓存路径:")
        print(f"{local_path}")

        # 2. 验证加载分词器
        tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        print(f"✅ 分词器自检通过！词表大小: {len(tokenizer)}")
        print("后续运行训练脚本或 merge_model.py 将直接从磁盘秒级加载，无需再次联网。")
        print("=" * 64)
    except Exception as e:
        print(f"\n[提示] snapshot_download 模式遇到问题: {e}")
        print("正在尝试直接调用 AutoModelForCausalLM.from_pretrained 进行拉取...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("✅ 模型与分词器下载并缓存成功！")
        except Exception as exc2:
            print(f"\n❌ 下载失败: {exc2}")
            print("提示：请确认本地网络可连通外部网络，如遇断网重新执行即可断点续传。")
            sys.exit(1)


if __name__ == "__main__":
    main()
