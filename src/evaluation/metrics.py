"""
离线评测指标：Recall@K, NDCG@K, Hit@K
支持批量评测（test split JSON）
"""
import json
import math
from typing import List, Dict


def recall_at_k(ranked: List[str], ground_truth: List[str], k: int) -> float:
    hits = len(set(ranked[:k]) & set(ground_truth))
    return hits / len(ground_truth) if ground_truth else 0.0


def hit_at_k(ranked: List[str], ground_truth: List[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & set(ground_truth) else 0.0


def ndcg_at_k(ranked: List[str], ground_truth: List[str], k: int) -> float:
    gt_set = set(ground_truth)
    dcg = 0.0
    for i, cid in enumerate(ranked[:k]):
        if cid in gt_set:
            dcg += 1.0 / math.log2(i + 2)
    # ideal DCG: all ground truth items at top positions
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(predictions: Dict[str, List[str]],
             ground_truths: Dict[str, List[str]],
             ks: List[int] = None) -> Dict:
    """
    predictions:   {user_id: [ranked course_id list]}
    ground_truths: {user_id: [true course_id list]}
    Returns averaged metrics dict.
    """
    ks = ks or [5, 10, 20]
    results = {f"{m}@{k}": [] for m in ["Recall", "NDCG", "Hit"] for k in ks}

    for uid, ranked in predictions.items():
        gt = ground_truths.get(uid, [])
        if not gt:
            continue
        for k in ks:
            results[f"Recall@{k}"].append(recall_at_k(ranked, gt, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(ranked, gt, k))
            results[f"Hit@{k}"].append(hit_at_k(ranked, gt, k))

    return {key: sum(vals) / len(vals) if vals else 0.0
            for key, vals in results.items()}


def evaluate_from_files(pred_path: str, gt_path: str,
                        ks: List[int] = None) -> Dict:
    """从 JSON 文件加载并评测"""
    with open(pred_path) as f:
        predictions = json.load(f)
    with open(gt_path) as f:
        ground_truths = json.load(f)
    return evaluate(predictions, ground_truths, ks)
