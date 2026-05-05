"""
scripts/build_index.py  —  构建 FAISS 向量索引
运行: python scripts/build_index.py
"""
import sys, json, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.faiss_retriever import FaissRetriever


def main():
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["data"]["processed_dir"]
    index_dir     = cfg["data"]["index_dir"]
    emb_cfg       = cfg["embedding"]
    ret_cfg       = cfg["retrieval"]

    course_db_path = Path(processed_dir) / "courses.json"
    with open(course_db_path, encoding="utf-8") as f:
        courses = json.load(f)

    # 构建 text_for_embed 字段
    for c in courses.values():
        concepts = "、".join(c.get("concepts", [])[:20])
        c["text_for_embed"] = f"{c.get('name', '')} {concepts}".strip()

    retriever = FaissRetriever(
        index_dir=index_dir,
        cfg={**emb_cfg, **ret_cfg},
    )
    retriever.build_index(courses, batch_size=emb_cfg.get("batch_size", 64))
    print("索引构建完成。")


if __name__ == "__main__":
    main()
