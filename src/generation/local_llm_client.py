"""
LocalLLMClient: 加载本地 LLM + LoRA adapter，接口与 LLMClient 完全相同。

配置 configs/config.yaml:
  generation:
    provider: "local"
    base_model: "E:/3-Models/Qwen2.5-7B-Instruct"
    lora_adapter: "models/lora_reranker"   # LoRA 训练产物
    temperature: 0.3
    max_new_tokens: 512
"""

import json
import re
from pathlib import Path


class LocalLLMClient:
    def __init__(self, cfg: dict):
        self.temperature    = cfg.get("temperature", 0.3)
        self.max_new_tokens = cfg.get("max_tokens", cfg.get("max_new_tokens", 512))
        base_model   = cfg.get("base_model", "")
        lora_adapter = cfg.get("lora_adapter", "")

        if not base_model:
            raise ValueError("generation.base_model must be set for provider=local")

        print(f"[LocalLLMClient] Loading base model: {base_model}")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        # 加载 LoRA adapter（如果存在）
        adapter_path = Path(lora_adapter)
        if lora_adapter and adapter_path.exists():
            print(f"[LocalLLMClient] Loading LoRA adapter: {lora_adapter}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
            self.model = self.model.merge_and_unload()   # 合并权重，推理更快
            print("[LocalLLMClient] LoRA merged.")
        else:
            if lora_adapter:
                print(f"[LocalLLMClient] WARNING: adapter path not found: {lora_adapter}, using base model only.")

        self.model.eval()
        print("[LocalLLMClient] Ready.")

    def chat(self, system: str, user: str) -> str:
        """返回原始文本响应"""
        import torch

        messages = [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=(self.temperature > 0),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 只取新生成的 token
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def chat_json(self, system: str, user: str) -> dict:
        """调用并解析 JSON 响应，失败时返回空 dict"""
        raw = self.chat(system, user)
        # 提取 JSON 块
        text = raw
        if "```" in text:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            text = m.group(0) if m else text
        else:
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start != -1:
                text = text[start:end]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": raw}
