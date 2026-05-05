"""
scripts/train_lora.py
使用 PEFT + LoRA 对本地 LLM 做 SFT，用于推荐重排任务。

依赖:
  pip install peft trl datasets

读取:
  data/sft/train.jsonl
  data/sft/valid.jsonl

输出:
  models/lora_reranker/  -- LoRA adapter

用法:
  python scripts/train_lora.py \
      --base_model "E:/3-Models/Qwen/Qwen2.5-1.5B-Instruct" \
      --output_dir models/lora_reranker \
      --epochs 3 --batch 2 --grad_accum 8
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def apply_chat_template(sample, tokenizer):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    parts = []
    for m in sample["messages"]:
        role, content = m["role"], m["content"]
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
    return "\n".join(parts) + tokenizer.eos_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",   type=str, required=True,
                        help="本地基座模型路径，例如 E:/3-Models/Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir",   type=str, default="models/lora_reranker")
    parser.add_argument("--sft_dir",      type=str, default="data/sft")
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--batch",        type=int,   default=2,
                        help="per_device_train_batch_size")
    parser.add_argument("--grad_accum",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--max_len",      type=int,   default=2048)
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install peft trl datasets")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Base model: {args.base_model}")

    # ── tokenizer ────────────────────────────────────────────────────────────
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 模型（fp16 + gradient checkpointing 节省显存）────────────────────────
    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()   # 用时间换显存，10GB 显存必备
    model.enable_input_require_grads()

    # ── LoRA 配置（适配 Qwen2 / Qwen2.5）────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── 数据集 ────────────────────────────────────────────────────────────────
    sft_dir = ROOT / args.sft_dir
    print("Loading SFT data ...")
    train_raw = load_jsonl(sft_dir / "train.jsonl")
    valid_raw = load_jsonl(sft_dir / "valid.jsonl")

    train_ds = Dataset.from_dict({"text": [apply_chat_template(s, tokenizer) for s in train_raw]})
    valid_ds = Dataset.from_dict({"text": [apply_chat_template(s, tokenizer) for s in valid_raw]})
    print(f"  train: {len(train_ds)}  valid: {len(valid_ds)}")

    # ── 训练参数 ──────────────────────────────────────────────────────────────
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=(device == "cuda"),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        dataset_text_field="text",
        max_seq_length=args.max_len,
        tokenizer=tokenizer,
    )

    print("Training ...")
    trainer.train()

    # ── 保存 LoRA adapter ─────────────────────────────────────────────────────
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nLoRA adapter saved: {output_dir}")
    print("Next: set provider=local and base_model in configs/config.yaml")


if __name__ == "__main__":
    main()