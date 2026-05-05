"""
cli.py  —  EduRec 命令行交互式推荐
运行: python cli.py
"""
import json
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval.faiss_retriever import FaissRetriever
from src.retrieval.bpr_retriever import BPRRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_client import LLMClient
from src.generation.generator import Generator
from src.reranking.reranker import LLMReranker


def build_llm_client(cfg: dict):
    """根据 generation.provider 选择 API 客户端或本地 LoRA 客户端"""
    provider = cfg.get("provider", "qwen")
    if provider == "local":
        from src.generation.local_llm_client import LocalLLMClient
        return LocalLLMClient(cfg)
    return LLMClient(cfg)

# ── 初始化 ─────────────────────────────────────────────────────────────────────

with open("configs/config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

PROCESSED_DIR = CFG["data"]["processed_dir"]
INDEX_DIR     = CFG["data"]["index_dir"]

print("Loading data...")
with open(Path(PROCESSED_DIR) / "courses.json", encoding="utf-8") as f:
    COURSES = json.load(f)
for c in COURSES.values():
    if "about" not in c:
        c["about"] = c.get("description", "") or " ".join(c.get("concepts", [])[:10])

with open(Path(PROCESSED_DIR) / "user_sequences.json", encoding="utf-8") as f:
    USER_SEQ = json.load(f)

with open(Path(PROCESSED_DIR) / "test.json", encoding="utf-8") as f:
    TEST_DATA = json.load(f)

print("Initializing retrievers...")
faiss_ret = FaissRetriever(INDEX_DIR, {**CFG["embedding"], **CFG["retrieval"]})
bpr_ret   = BPRRetriever(PROCESSED_DIR, CFG["retrieval"])

if Path(INDEX_DIR, "course.faiss").exists():
    faiss_ret.load(COURSES)
    bpr_ret.load(COURSES)

hybrid   = HybridRetriever(faiss_ret, bpr_ret, alpha=0.7)
llm      = build_llm_client(CFG["generation"])
gen      = Generator(llm, {"max_history": 10, "max_candidates": 8})
reranker = LLMReranker(gen, top_k=CFG["retrieval"]["top_k_rerank"])

SAMPLE_USERS = list(TEST_DATA.keys())[:20]

# ── 工具函数 ───────────────────────────────────────────────────────────────────

def get_history(user_id):
    seq = USER_SEQ.get(user_id, [])
    ids = seq[:-2] if len(seq) > 2 else seq[:-1] if len(seq) > 1 else []
    return [COURSES[c] for c in ids if c in COURSES]

def show_history(history):
    if not history:
        print("  （无历史记录）")
        return
    for i, c in enumerate(history[-5:], 1):
        name  = c.get("name", c["course_id"])
        about = c.get("about", "")[:60]
        print(f"  {i}. {name}")
        if about:
            print(f"     {about}")

def show_results(results, with_reason=False):
    for i, c in enumerate(results, 1):
        name  = c.get("name", c["course_id"])
        about = c.get("about", c.get("description", ""))[:80]
        score = c.get("hybrid_score", c.get("score", 0))
        print(f"\n  {i}. {name}  [score={score:.4f}]")
        if about:
            print(f"     {about}")
        if with_reason and c.get("reason"):
            print(f"     💡 {c['reason']}")

# ── 主循环 ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  EduRec 教育课程推荐系统（命令行版）")
    print("="*60)
    print(f"\n示例用户 ID（共 {len(TEST_DATA)} 个）：")
    for u in SAMPLE_USERS:
        print(f"  {u}")

    while True:
        print("\n" + "-"*60)
        user_id = input("输入用户 ID（回车跳过用查询词推荐，q 退出）: ").strip()

        if user_id.lower() == "q":
            print("退出。")
            break

        query = input("补充需求（可选，直接回车跳过）: ").strip()

        use_llm_input = input("启用 LLM 重排？(y/n，默认 n): ").strip().lower()
        use_llm = use_llm_input == "y"

        # 获取用户历史
        if user_id:
            history = get_history(user_id)
            print(f"\n📚 用户 [{user_id}] 的学习历史（最近 5 门）：")
            show_history(history)
        else:
            history = []

        # 构建查询
        effective_query = query or " ".join(c.get("name","") for c in history[-3:]) or "课程推荐"
        print(f"\n🔍 查询词：{effective_query}")

        # 召回
        print("召回中...", end="", flush=True)
        candidates = hybrid.retrieve(
            query=effective_query,
            user_id=user_id or None,
            exclude_ids=[c["course_id"] for c in history],
            top_k=CFG["retrieval"]["top_k_recall"],
        )
        print(f" 找到 {len(candidates)} 个候选")

        if not candidates:
            print("未找到相关课程。")
            continue

        # 重排
        if use_llm:
            print("LLM 重排中（首次需加载模型，约 30-60s）...")
            results = reranker.rerank(history, candidates, user_query=query)
        else:
            results = candidates[:CFG["retrieval"]["top_k_rerank"]]

        print(f"\n✨ 推荐结果（Top {len(results)}）：")
        show_results(results, with_reason=use_llm)

if __name__ == "__main__":
    main()
