"""
Generator: 端到端生成推荐结果
  输入: 用户历史 + 候选课程列表
  输出: 重排后的课程列表 + 推荐理由
"""
from typing import List, Dict
from .prompt_builder import PromptBuilder
from .llm_client import LLMClient


class Generator:
    def __init__(self, llm: LLMClient, cfg: dict = None):
        cfg = cfg or {}
        self.llm     = llm
        self.builder = PromptBuilder(
            max_history=cfg.get("max_history", 10),
            max_candidates=cfg.get("max_candidates", 8),
        )

    def generate(self, user_history: List[Dict], candidates: List[Dict],
                 user_query: str = "") -> List[Dict]:
        """
        Returns list of course dicts with added field 'reason'.
        Order follows LLM's ranked_ids; unranked candidates appended at end.
        """
        if not candidates:
            return []

        system, user_msg = self.builder.build(user_history, candidates, user_query)
        result = self.llm.chat_json(system, user_msg)

        ranked_ids = result.get("ranked_ids", [])
        reasons    = result.get("reasons", {})

        # 构建 course_id → candidate 映射
        cand_map = {c["course_id"]: c for c in candidates}

        ordered = []
        seen    = set()
        for cid in ranked_ids:
            if cid in cand_map and cid not in seen:
                item = dict(cand_map[cid])
                item["reason"] = reasons.get(cid, "")
                ordered.append(item)
                seen.add(cid)

        # 追加 LLM 未提及的候选（保底）
        for c in candidates:
            if c["course_id"] not in seen:
                item = dict(c)
                item["reason"] = ""
                ordered.append(item)

        return ordered
