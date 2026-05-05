"""
app.py  —  EduRec Gradio 演示界面
运行: python app.py
"""
import json
import os
import sys
import yaml
import gradio as gr
from pathlib import Path

# 必须在 gradio import 之后、任何 gr.* 调用之前设置，禁止联网（网络受限时会卡住）
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval.faiss_retriever import FaissRetriever
from src.retrieval.bpr_retriever import BPRRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_client import LLMClient
from src.generation.generator import Generator
from src.reranking.reranker import LLMReranker

# ── 全局初始化 ────────────────────────────────────────────────────────────────

with open("configs/config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f.read())

PROCESSED_DIR = CFG["data"]["processed_dir"]
INDEX_DIR     = CFG["data"]["index_dir"]

# 加载课程数据库（courses.json 字段: course_id, name, description, concepts, ...）
_course_db_path = Path(PROCESSED_DIR) / "courses.json"
COURSES: dict = {}
if _course_db_path.exists():
    with open(_course_db_path, encoding="utf-8") as f:
        COURSES = json.load(f)
    # 统一 about 字段，兼容 PromptBuilder / 格式化函数
    for c in COURSES.values():
        if "about" not in c:
            c["about"] = c.get("description", "") or " ".join(c.get("concepts", [])[:10])

# 加载用户完整交互序列（用于提取历史，格式: {user_id: [course_id, ...]}）
_seq_path = Path(PROCESSED_DIR) / "user_sequences.json"
USER_SEQ: dict = {}
if _seq_path.exists():
    with open(_seq_path, encoding="utf-8") as f:
        USER_SEQ = json.load(f)

# 加载 test split（仅用于前端下拉框展示样本用户）
_test_path = Path(PROCESSED_DIR) / "test.json"
TEST_DATA: dict = {}
if _test_path.exists():
    with open(_test_path, encoding="utf-8") as f:
        TEST_DATA = json.load(f)

# 初始化检索器
faiss_ret = FaissRetriever(INDEX_DIR, {**CFG["embedding"], **CFG["retrieval"]})
bpr_ret   = BPRRetriever(PROCESSED_DIR, CFG["retrieval"])

_index_ready = Path(INDEX_DIR, "course.faiss").exists()
if _index_ready:
    faiss_ret.load(COURSES)
    bpr_ret.load(COURSES)

hybrid = HybridRetriever(faiss_ret, bpr_ret, alpha=0.7)

# 初始化生成器
llm     = LLMClient(CFG["generation"])
gen     = Generator(llm, {"max_history": 10, "max_candidates": 8})
reranker = LLMReranker(gen, top_k=CFG["retrieval"]["top_k_rerank"])

# ── 核心推荐函数 ──────────────────────────────────────────────────────────────

def get_user_history(user_id: str):
    """从 user_sequences.json 取训练历史（去掉末尾 valid+test 各 1 条）"""
    seq = USER_SEQ.get(user_id, [])
    # leave-one-out: 倒数第1=test, 倒数第2=valid, 其余=训练历史
    history_ids = seq[:-2] if len(seq) > 2 else seq[:-1] if len(seq) > 1 else []
    return [COURSES[cid] for cid in history_ids if cid in COURSES]


def recommend(user_id: str, user_query: str, use_llm_rerank: bool):
    if not _index_ready:
        return "⚠️ 索引未构建，请先运行 `python scripts/build_index.py`", ""

    user_history = get_user_history(user_id.strip())
    query = user_query.strip() or (
        " ".join(c.get("name", "") for c in user_history[-3:])
    )

    # 召回
    candidates = hybrid.retrieve(
        query=query,
        user_id=user_id.strip() or None,
        exclude_ids=[c["course_id"] for c in user_history],
        top_k=CFG["retrieval"]["top_k_recall"],
    )

    if not candidates:
        return "未找到相关课程，请检查索引或调整查询。", ""

    # 重排
    if use_llm_rerank:
        results = reranker.rerank(user_history, candidates, user_query=user_query)
    else:
        results = candidates[:CFG["retrieval"]["top_k_rerank"]]

    # 格式化输出
    history_md = _format_history_md(user_history)
    results_md = _format_results_md(results, use_llm_rerank)
    return history_md, results_md


def _format_history_md(history):
    if not history:
        return "（无历史记录）"
    lines = [f"**{i+1}. {c.get('name', c['course_id'])}**  \n{c.get('about', c.get('description', ''))[:100]}"
             for i, c in enumerate(history[-5:])]
    return "\n\n".join(lines)


def _format_results_md(results, with_reason):
    lines = []
    for i, c in enumerate(results, 1):
        name   = c.get("name", c["course_id"])
        about  = c.get("about", c.get("description", ""))[:120]
        reason = c.get("reason", "")
        score  = c.get("hybrid_score", c.get("score", ""))
        score_str = f"  `score={score:.3f}`" if isinstance(score, float) else ""
        block = f"**{i}. {name}**{score_str}  \n{about}"
        if with_reason and reason:
            block += f"\n\n> 推荐理由：{reason}"
        lines.append(block)
    return "\n\n---\n\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

sample_users = list(TEST_DATA.keys())[:20] if TEST_DATA else []

with gr.Blocks(title="EduRec — 教育课程推荐演示") as demo:
    gr.Markdown("# EduRec 教育课程推荐系统\n基于 MOOCCube 数据集，融合语义检索 + LLM 重排")

    with gr.Row():
        with gr.Column(scale=1):
            user_id_input = gr.Dropdown(
                choices=sample_users,
                label="选择用户 ID（来自 test split）",
                allow_custom_value=True,
            )
            query_input = gr.Textbox(
                label="补充需求（可选）",
                placeholder="例如：我想学深度学习相关课程",
                lines=2,
            )
            use_llm = gr.Checkbox(label="启用 LLM 重排（需要 API Key）", value=True)
            btn = gr.Button("推荐", variant="primary")

        with gr.Column(scale=2):
            history_out = gr.Markdown(label="用户历史（最近 5 门）")
            results_out = gr.Markdown(label="推荐结果")

    btn.click(
        fn=recommend,
        inputs=[user_id_input, query_input, use_llm],
        outputs=[history_out, results_out],
    )

if __name__ == "__main__":
    import os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # 禁用联网统计（避免网络卡住）
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
    )
