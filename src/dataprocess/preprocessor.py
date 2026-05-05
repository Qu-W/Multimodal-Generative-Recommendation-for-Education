"""
MOOCCubePreprocessor: 清洗、过滤、构建课程数据库和交互序列
"""
import json
import os
import pandas as pd
from collections import defaultdict
from pathlib import Path


class MOOCCubePreprocessor:
    def __init__(self, raw_dir: str, output_dir: str, cfg: dict):
        self.raw_dir    = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_user_inter   = cfg.get("min_user_interactions", 5)
        self.min_course_inter = cfg.get("min_course_interactions", 10)

    def _load(self, rel_path):
        path = self.raw_dir / rel_path
        with open(path, encoding="utf-8") as f:
            first = f.read(1)
            f.seek(0)
            if first == "[":
                return json.load(f)
            return [json.loads(line) for line in f if line.strip()]

    def _load_tsv(self, rel_path, cols):
        path = self.raw_dir / rel_path
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                rows.append({cols[i]: parts[i] for i in range(min(len(cols), len(parts)))})
        return rows

    def _save(self, obj, filename):
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"  保存: {path}")

    # ── 公开接口 ──────────────────────────────────────
    def run(self):
        print("Step 1: 构建课程数据库...")
        course_db = self.build_course_db()

        print("Step 2: 构建用户交互序列...")
        df, user_seqs = self.build_interactions()

        print("Step 3: 构建知识图谱...")
        self.build_knowledge_graph()

        print("Step 4: 统计活跃度分布...")
        self.compute_user_activity(user_seqs)

        print("\n预处理完成！")
        return course_db, user_seqs

    def build_course_db(self):
        courses_raw  = self._load("entities/course.json")
        concepts_raw = self._load("entities/concept.json")
        cc_rel       = self._load_tsv("relations/course-concept.json", ["course_id", "concept_id"])

        # 支持 list 和 dict 两种格式
        if isinstance(courses_raw, dict):
            courses_raw = list(courses_raw.values())
        if isinstance(concepts_raw, dict):
            concepts_raw = list(concepts_raw.values())

        concept_id2name = {}
        for c in concepts_raw:
            cid = c.get("id") or c.get("concept_id")
            concept_id2name[cid] = c.get("name", "")

        course2concepts = defaultdict(list)
        for rel in cc_rel:
            cid  = rel.get("course_id")
            conc = concept_id2name.get(rel.get("concept_id"), "")
            if conc:
                course2concepts[cid].append(conc)

        course_db = {}
        for c in courses_raw:
            cid = c.get("id") or c.get("course_id")
            concepts = course2concepts.get(cid, [])
            course_db[cid] = {
                "course_id":      cid,
                "name":           c.get("name", ""),
                "description":    c.get("description", ""),
                "school":         c.get("school", ""),
                "teacher":        c.get("teacher", ""),
                "concepts":       concepts,
                "text_for_embed": (
                    f"{c.get('name','')} "
                    f"{c.get('description','')} "
                    + " ".join(concepts[:10])
                ).strip(),
            }

        self._save(course_db, "courses.json")
        print(f"  课程总数: {len(course_db)}")
        return course_db

    def build_interactions(self):
        raw = self._load_tsv("relations/user-course.json", ["user_id", "course_id"])
        if isinstance(raw, dict):
            raw = list(raw.values())

        df = pd.DataFrame(raw)
        print(f"  原始交互: {len(df)} 条, 字段: {df.columns.tolist()}")

        # 时间排序（字段名可能不同，自动检测）
        time_col = next((c for c in df.columns if "time" in c.lower()), None)
        if time_col:
            df = df.sort_values(time_col)

        # 过滤冷启动
        for _ in range(3):   # 迭代过滤直到稳定
            prev = len(df)
            course_cnt = df["course_id"].value_counts()
            df = df[df["course_id"].isin(
                course_cnt[course_cnt >= self.min_course_inter].index)]
            user_cnt = df["user_id"].value_counts()
            df = df[df["user_id"].isin(
                user_cnt[user_cnt >= self.min_user_inter].index)]
            if len(df) == prev:
                break

        print(f"  过滤后: {len(df)} 条, "
              f"{df['user_id'].nunique()} 用户, "
              f"{df['course_id'].nunique()} 课程")

        df.to_csv(self.output_dir / "interactions.csv", index=False)

        user_seqs = df.groupby("user_id")["course_id"].apply(list).to_dict()
        self._save(user_seqs, "user_sequences.json")
        return df, user_seqs

    def build_knowledge_graph(self):
        prereq = self._load_tsv("relations/prerequisite-dependency.json", ["head", "tail"])
        if isinstance(prereq, dict):
            prereq = list(prereq.values())
        graph = defaultdict(list)
        for rel in prereq:
            graph[rel.get("head", rel.get("source"))].append(
                rel.get("tail", rel.get("target")))
        self._save(dict(graph), "prereq_graph.json")
        print(f"  知识图谱节点数: {len(graph)}")

    def compute_user_activity(self, user_seqs):
        lengths = sorted(len(v) for v in user_seqs.values())
        n = len(lengths)
        stats = {
            "total_users":  n,
            "median_len":   lengths[n // 2],
            "p25_len":      lengths[n // 4],
            "p75_len":      lengths[3 * n // 4],
            "max_len":      lengths[-1],
        }
        self._save(stats, "activity_stats.json")
        print(f"  活跃度统计: {stats}")
