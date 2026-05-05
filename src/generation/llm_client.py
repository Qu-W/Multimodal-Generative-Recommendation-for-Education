"""
LLMClient: OpenAI-compatible 接口封装（支持 Qwen / OpenAI / 本地 vLLM）
"""
import json
import os
from openai import OpenAI


class LLMClient:
    def __init__(self, cfg: dict):
        self.model       = cfg.get("model_name", "qwen-plus")
        self.temperature = cfg.get("temperature", 0.7)
        self.max_tokens  = cfg.get("max_tokens", 1024)
        api_base = cfg.get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        api_key  = cfg.get("api_key") or os.environ.get("DASHSCOPE_API_KEY", "")
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def chat(self, system: str, user: str) -> str:
        """返回原始文本响应"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def chat_json(self, system: str, user: str) -> dict:
        """调用并解析 JSON 响应，失败时返回空 dict"""
        raw = self.chat(system, user)
        # 提取 JSON 块（模型可能包裹在 ```json ... ``` 中）
        text = raw
        if "```" in text:
            start = text.find("{")
            end   = text.rfind("}") + 1
            text  = text[start:end] if start != -1 else text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": raw}
