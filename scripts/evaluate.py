"""
scripts/evaluate.py  —  离线批量评测
运行: python scripts/evaluate.py --pred data/processed/predictions.json
"""
import sys, json, yaml, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="预测结果 JSON {user_id: [course_id, ...]}")
    parser.add_argument("--gt",   default=None,  help="ground truth JSON（默认用 test split）")
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    gt_path = args.gt or str(Path(cfg["data"]["processed_dir"]) / "test.json")
    with open(args.pred) as f:
        predictions = json.load(f)
    with open(gt_path) as f:
        raw_gt = json.load(f)

    # test.json 格式: {user_id: {"history": [...], "target": "course_id"}}
    ground_truths = {uid: [v["target"]] for uid, v in raw_gt.items()}

    ks = cfg["evaluation"]["topk"]
    results = evaluate(predictions, ground_truths, ks)

    print("\n=== 评测结果 ===")
    for metric, val in sorted(results.items()):
        print(f"  {metric}: {val:.4f}")


if __name__ == "__main__":
    main()
