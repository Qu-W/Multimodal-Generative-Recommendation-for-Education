"""
PromptBuilder: 将用户历史 + 候选课程 → LLM prompt
"""
from typing import List, Dict


SYSTEM_PROMPT = """你是一位专业的在线教育课程推荐助手。
根据用户的学习历史和兴趣，从候选课程中选出最适合的课程，并给出简洁的推荐理由。
输出格式为 JSON，字段：ranked_ids（课程ID列表，按推荐优先级排序）、reasons（每门课的一句话理由，dict）。"""


class PromptBuilder:
    def __init__(self, max_history: int = 10, max_candidates: int = 8):
        self.max_history    = max_history
        self.max_candidates = max_candidates

    def build(self, user_history: List[Dict], candidates: List[Dict],
              user_query: str = "") -> tuple[str, str]:
        """
        Returns (system_prompt, user_message)
        user_history: list of {"course_id", "name", "about"}
        candidates:   list of {"course_id", "name", "about", ...}
        """
        history_text = self._format_history(user_history)
        cand_text    = self._format_candidates(candidates)

        user_msg = f"""## 用户学习历史（最近 {len(user_history)} 门）
{history_text}

## 用户当前需求
{user_query if user_query else "（无额外说明，请根据历史推断兴趣）"}

## 候选课程（共 {len(candidates)} 门）
{cand_text}

请从候选课程中选出最适合该用户的课程，按推荐优先级排序，并给出每门课的推荐理由（一句话）。
输出严格 JSON，示例：
{{"ranked_ids": ["C_001", "C_002"], "reasons": {{"C_001": "...", "C_002": "..."}}}}"""

        return SYSTEM_PROMPT, user_msg

    def _format_history(self, history: List[Dict]) -> str:
        items = history[-self.max_history:]
        if not items:
            return "（暂无学习记录）"
        lines = []
        for i, c in enumerate(items, 1):
            name  = c.get("name", c.get("course_id", "?"))
            about = c.get("about", "")[:80]
            lines.append(f"{i}. [{c.get('course_id','')}] {name} — {about}")
        return "\n".join(lines)

    def _format_candidates(self, candidates: List[Dict]) -> str:
        items = candidates[:self.max_candidates]
        lines = []
        for c in items:
            name  = c.get("name", c.get("course_id", "?"))
            about = c.get("about", "")[:120]
            lines.append(f"- [{c['course_id']}] {name}\n  {about}")
        return "\n".join(lines)
