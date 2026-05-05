"""
FaissRetriever: 基于 BGE-M3 语义嵌入的向量召回
"""
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer


class FaissRetriever:
    def __init__(self, index_dir: str, cfg: dict):
        self.index_dir  = Path(index_dir)
        self.model_name = cfg.get("model_name", "BAAI/bge-m3")
        self.top_k      = cfg.get("top_k_recall", 20)
        self.embed_model = None   # 懒加载
        self.index       = None
        self.course_ids  = None
        self.courses     = None

    def build_index(self, courses: dict, batch_size: int = 64):
        """离线构建 FAISS 索引，保存到 index_dir"""
        self._load_embed_model()
        self.index_dir.mkdir(parents=True, exist_ok=True)

        ids   = list(courses.keys())
        texts = [c["text_for_embed"] for c in courses.values()]

        print(f"编码 {len(texts)} 门课程（模型: {self.model_name}）...")
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            e = self.embed_model.encode(
                batch, normalize_embeddings=True, show_progress_bar=False)
            embs.append(e)
            if (i // batch_size) % 10 == 0:
                print(f"  {i}/{len(texts)}")

        embs = np.vstack(embs).astype("float32")
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)

        faiss.write_index(index, str(self.index_dir / "course.faiss"))
        with open(self.index_dir / "course_ids.json", "w") as f:
            json.dump(ids, f)
        print(f"索引构建完成，维度={embs.shape[1]}, 条目={embs.shape[0]}")

    def load(self, courses: dict):
        """推理时加载 FAISS 索引（不加载嵌入模型，等第一次 retrieve 时再懒加载）"""
        self.index = faiss.read_index(str(self.index_dir / "course.faiss"))
        with open(self.index_dir / "course_ids.json") as f:
            self.course_ids = json.load(f)
        self.courses = courses
        # 注意：embed_model 保持 None，retrieve() 首次调用时才加载（节省启动时间）

    def retrieve(self, query: str, exclude_ids: list = None, top_k: int = None):
        self._load_embed_model()   # 懒加载：首次调用时才加载 2GB 模型
        top_k = top_k or self.top_k
        exclude = set(exclude_ids or [])
        q_emb = self.embed_model.encode(
            [query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q_emb, top_k + len(exclude) + 10)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.course_ids):
                continue
            cid = self.course_ids[idx]
            if cid in exclude:
                continue
            results.append({**self.courses[cid], "score": float(score)})
            if len(results) >= top_k:
                break
        return results

    def _load_embed_model(self):
        if self.embed_model is None:
            import warnings
            print(f"加载嵌入模型: {self.model_name}")
            # 屏蔽 BGE-M3 tokenizer regex 无害警告（fix_mistral_regex 在旧版 tokenizers 会崩溃）
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
                self.embed_model = SentenceTransformer(self.model_name)
