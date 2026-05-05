"""
DataExplorer: 快速了解 MOOCCube 数据集的字段结构和统计信息
运行: python -m src.dataprocess.explorer
"""
import json
import os
from collections import defaultdict
from pathlib import Path


class DataExplorer:
    def __init__(self, raw_dir: str):
        self.raw_dir = Path(raw_dir)

    def _load(self, rel_path: str):
        path = self.raw_dir / rel_path
        if not path.exists():
            print(f"  [MISS] {path}")
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def explore_all(self):
        print("=" * 60)
        print("MOOCCube 数据集探索报告")
        print("=" * 60)
        self._explore_entity("entities/course.json",  "课程")
        self._explore_entity("entities/concept.json", "知识概念")
        self._explore_entity("entities/video.json",   "视频")
        self._explore_entity("entities/user.json",    "用户")
        self._explore_relation("relations/user-course.json",          "用户-课程注册")
        self._explore_relation("relations/user-video.json",           "用户-视频观看")
        self._explore_relation("relations/course-concept.json",       "课程-概念")
        self._explore_relation("relations/concept_prerequisite.json", "概念前置关系")

    def _explore_entity(self, rel_path: str, label: str):
        data = self._load(rel_path)
        if data is None:
            return
        items = list(data.values()) if isinstance(data, dict) else data
        print(f"\n[{label}] 共 {len(items)} 条")
        if items:
            print(f"  字段: {list(items[0].keys())}")
            print(f"  示例: {items[0]}")

    def _explore_relation(self, rel_path: str, label: str):
        data = self._load(rel_path)
        if data is None:
            return
        items = list(data.values()) if isinstance(data, dict) else data
        print(f"\n[{label}] 共 {len(items)} 条")
        if items:
            print(f"  字段: {list(items[0].keys())}")
            # 统计用户/课程分布
            if "user_id" in items[0]:
                uid_counts = defaultdict(int)
                for r in items:
                    uid_counts[r.get("user_id")] += 1
                vals = sorted(uid_counts.values())
                print(f"  用户数: {len(vals)}, 中位数交互: {vals[len(vals)//2]}")


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    explorer = DataExplorer(cfg["data"]["raw_dir"])
    explorer.explore_all()
