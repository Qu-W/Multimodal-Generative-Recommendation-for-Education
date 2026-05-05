"""
LLMReranker: 用 LLM 对召回结果重排，返回 top_k 条带理由的课程
"""
from typing import List, Dict
from src.generation.generator import Generator


class LLMReranker:
    def __init__(self, generator: Generator, top_k: int = 8):
        self.generator = generator
        self.top_k     = top_k

    def rerank(self, user_history: List[Dict], candidates: List[Dict],
               user_query: str = "", top_k: int = None) -> List[Dict]:
        """
        candidates: 召回阶段返回的课程列表（已含 course_id, name, about 等字段）
        Returns top_k courses with 'reason' field added.
        """
        top_k = top_k or self.top_k
        ranked = self.generator.generate(user_history, candidates, user_query)
        return ranked[:top_k]
