"""
BPRRetriever: 协同过滤召回（后期可替换为 MMRec 张量模型）
当前为占位实现，接口与 FaissRetriever 保持一致
"""
import json
import numpy as np
from pathlib import Path


class BPRRetriever:
    """
    接口占位符。
    后期替换为：
      from src.models.mmrec_bridge import MMRecBridge
      self.model = MMRecBridge(model_name="TMRec", ckpt_path=...)
    """
    def __init__(self, processed_dir: str, cfg: dict):
        self.processed_dir = Path(processed_dir)
        self.top_k = cfg.get("top_k_recall", 20)
        self.user_emb   = None   # [n_users, d]
        self.item_emb   = None   # [n_items, d]
        self.course_ids = None
        self._uid_map   = None   # loaded lazily in load()

    def train(self, interactions_csv: str):
        """训练 BPR，保存嵌入矩阵（简化版，后期接 RecBole）"""
        raise NotImplementedError(
            "请使用 scripts/train_bpr.py 训练后加载嵌入矩阵")

    def load(self, courses: dict):
        emb_path = self.processed_dir / "bpr_embeddings.npz"
        if not emb_path.exists():
            print("[BPRRetriever] 未找到嵌入文件，跳过协同过滤召回")
            return False
        data = np.load(emb_path)
        self.user_emb = data["user_emb"]
        self.item_emb = data["item_emb"]

        # 优先使用 item_idx_list.json（由 convert_to_mmrec.py 按整数索引顺序写入）
        idx_list_path = self.processed_dir / "item_idx_list.json"
        if idx_list_path.exists():
            self.course_ids = json.load(open(idx_list_path))
        else:
            self.course_ids = list(courses.keys())

        # 加载用户 ID 映射（string → int）
        uid_map_path = self.processed_dir / "user_id_map.json"
        if uid_map_path.exists():
            self._uid_map = json.load(open(uid_map_path))
        return True

    def retrieve(self, user_id: str, exclude_ids: list = None, top_k: int = None):
        if self.user_emb is None:
            return []
        top_k   = top_k or self.top_k
        exclude = set(exclude_ids or [])
        uid_map = self._uid_map
        if uid_map is None:
            uid_map = json.load(open(self.processed_dir / "user_id_map.json"))
        if user_id not in uid_map:
            return []
        u_vec  = self.user_emb[uid_map[user_id]]
        scores = self.item_emb @ u_vec
        ranked = np.argsort(-scores)
        results = []
        for idx in ranked:
            cid = self.course_ids[idx]
            if cid not in exclude:
                results.append({"course_id": cid, "cf_score": float(scores[idx])})
            if len(results) >= top_k:
                break
        return results
