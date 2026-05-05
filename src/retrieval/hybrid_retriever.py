"""
HybridRetriever: 融合语义召回 + 协同过滤召回
"""
from .faiss_retriever import FaissRetriever
from .bpr_retriever import BPRRetriever


class HybridRetriever:
    def __init__(self, faiss: FaissRetriever, bpr: BPRRetriever,
                 alpha: float = 0.7):
        """
        alpha: 语义召回权重（1-alpha 为协同过滤权重）
        """
        self.faiss = faiss
        self.bpr   = bpr
        self.alpha = alpha

    def retrieve(self, query: str, user_id: str = None,
                 exclude_ids: list = None, top_k: int = 20):
        # 语义召回
        semantic = self.faiss.retrieve(query, exclude_ids, top_k=top_k)
        sem_map  = {r["course_id"]: r["score"] for r in semantic}

        # 协同过滤召回（可选）
        cf_map = {}
        if user_id:
            cf_results = self.bpr.retrieve(user_id, exclude_ids, top_k=top_k)
            if cf_results:
                max_cf = max(r["cf_score"] for r in cf_results) + 1e-8
                cf_map = {r["course_id"]: r["cf_score"] / max_cf
                          for r in cf_results}

        # 合并打分
        all_ids = set(sem_map) | set(cf_map)
        fused = []
        for cid in all_ids:
            score = (self.alpha * sem_map.get(cid, 0.0) +
                     (1 - self.alpha) * cf_map.get(cid, 0.0))
            fused.append((cid, score))

        fused.sort(key=lambda x: -x[1])
        courses = self.faiss.courses
        return [
            {**courses[cid], "hybrid_score": s}
            for cid, s in fused[:top_k]
            if cid in courses
        ]
