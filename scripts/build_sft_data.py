"""
scripts/build_sft_data.py
从用户交互序列构建 LLM 重排的 SFT 训练数据。

策略（leave-one-out 风格）：
  对每个用户的训练序列，以末尾一项为正样本，
  随机采样负样本，构建 (history, candidates, ground_truth_ranking) 三元组。

输出:
  data/sft/train.jsonl  —  chat format，每行一个样本
  data/sft/valid.jsonl

用法:
  python scripts/build_sft_data.py [--max_users 5000] [--neg_k 7]
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.generation.prompt_builder import PromptBuilder, SYSTEM_PROMPT

PROCESSED = ROOT / "data" / "processed"
SFT_DIR   = ROOT / "data" / "sft"
SFT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    with open(PROCESSED / "courses.json", encoding="utf-8") as f:
        courses = json.load(f)
    for c in courses.values():
        if "about" not in c:
            c["about"] = c.get("description", "") or " ".join(c.get("concepts", [])[:8])
    with open(PROCESSED / "user_sequences.json", encoding="utf-8") as f:
        user_seq = json.load(f)
    return courses, user_seq


def make_reason(course: dict) -> str:
    """基于课程元数据生成模板化推荐理由（作为 SFT 目标输出）"""
    name     = course.get("name", "该课程")
    concepts = course.get("concepts", [])
    if concepts:
        top = "、".join(concepts[:3])
        return f"{name} 涵盖 {top} 等核心知识点，与用户兴趣高度相关。"
    about = course.get("about", "")[:40]
    suffix = "——" + about if about else ""
    return f"{name}{suffix}，适合用户当前学习阶段。"


def build_sample(history_courses, candidate_courses, positive_id, builder):
    """构建一个 chat 格式样本"""
    system, user_msg = builder.build(history_courses, candidate_courses)

    # ground truth：正样本排第一，其余按原顺序
    ranked_ids = [positive_id] + [
        c["course_id"] for c in candidate_courses if c["course_id"] != positive_id
    ]
    reasons = {c["course_id"]: make_reason(c) for c in candidate_courses}

    assistant_msg = json.dumps(
        {"ranked_ids": ranked_ids, "reasons": reasons},
        ensure_ascii=False
    )
    return {
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_users",   type=int,   default=5000,
                        help="最多使用多少用户构建训练数据")
    parser.add_argument("--neg_k",       type=int,   default=7,
                        help="每个正样本配 neg_k 个负样本（候选总数 = neg_k+1）")
    parser.add_argument("--min_hist",    type=int,   default=3,
                        help="用户至少有多少历史条目才参与采样")
    parser.add_argument("--valid_ratio", type=float, default=0.05)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    print("Loading data ...")
    courses, user_seq = load_data()
    all_course_ids = list(courses.keys())

    builder  = PromptBuilder(max_history=5, max_candidates=args.neg_k + 1)
    samples  = []
    skipped  = 0
    users    = list(user_seq.items())
    random.shuffle(users)

    for user_id, seq in users:
        if len(samples) >= args.max_users:
            break

        # 训练序列 = 去掉末尾2条（valid + test）
        train_seq = seq[:-2] if len(seq) > 2 else seq[:-1] if len(seq) > 1 else []
        if len(train_seq) < args.min_hist + 1:
            skipped += 1
            continue

        positive_id = train_seq[-1]
        history_ids = train_seq[-(args.min_hist + 1):-1]

        if positive_id not in courses:
            skipped += 1
            continue

        history_courses = [courses[c] for c in history_ids if c in courses]
        if not history_courses:
            skipped += 1
            continue

        # 负样本：从全集随机采样，排除用户已学过的
        user_set = set(seq)
        negatives = [c for c in all_course_ids if c not in user_set]
        if len(negatives) < args.neg_k:
            skipped += 1
            continue
        neg_ids = random.sample(negatives, args.neg_k)

        # 候选集随机打乱（避免位置偏差）
        cand_ids = [positive_id] + neg_ids
        random.shuffle(cand_ids)
        candidate_courses = [courses[c] for c in cand_ids if c in courses]

        samples.append(build_sample(history_courses, candidate_courses, positive_id, builder))

    print(f"Built {len(samples)} samples (skipped {skipped})")

    random.shuffle(samples)
    n_valid      = max(1, int(len(samples) * args.valid_ratio))
    valid_samples = samples[:n_valid]
    train_samples = samples[n_valid:]

    def write_jsonl(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(SFT_DIR / "train.jsonl", train_samples)
    write_jsonl(SFT_DIR / "valid.jsonl", valid_samples)
    print(f"train: {len(train_samples)} samples → {SFT_DIR}/train.jsonl")
    print(f"valid: {len(valid_samples)} samples → {SFT_DIR}/valid.jsonl")
    print("\nSample preview (first train item):")
    s = train_samples[0]
    print("  USER:", s["messages"][1]["content"][:300], "...")
    print("  ASST:", s["messages"][2]["content"][:200])


if __name__ == "__main__":
    main()