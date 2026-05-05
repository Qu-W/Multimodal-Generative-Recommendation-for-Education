"""
scripts/convert_to_mmrec.py
将 MoocCube 处理后的数据转换为 MMRec 可直接使用的格式。

输出:
  data/mmrec/mooccube/mooccube.inter  — tab 分隔: userID itemID ratings timestamp x_label
  data/mmrec/mooccube/text_feat.npy   — shape (n_items, 1024), BGE-M3 embedding
  data/processed/user_id_map.json     — string user_id → int index
  data/processed/item_id_map.json     — string course_id → int index

用法（需要 faiss-cpu，在 minicpm 环境中运行）:
  conda run -n minicpm python scripts/convert_to_mmrec.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "data" / "index"
MMREC_DIR = ROOT / "data" / "mmrec" / "mooccube"
MMREC_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # ── 1. 加载交互数据 ────────────────────────────────────────────────────────
    print("Loading interactions.csv ...")
    rows = []
    with open(PROCESSED / "interactions.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["user_id"], r["course_id"]))
    print(f"  {len(rows)} interactions")

    # ── 2. 加载 valid/test 划分 ────────────────────────────────────────────────
    print("Loading valid/test splits ...")
    with open(PROCESSED / "valid.json", encoding="utf-8") as f:
        valid_data = json.load(f)
    with open(PROCESSED / "test.json", encoding="utf-8") as f:
        test_data = json.load(f)

    valid_set = {(u, c) for u, cs in valid_data.items() for c in cs}
    test_set  = {(u, c) for u, cs in test_data.items()  for c in cs}
    print(f"  valid pairs: {len(valid_set)}  test pairs: {len(test_set)}")

    # ── 3. 建立整数 ID 映射（按首次出现顺序） ─────────────────────────────────
    user_id_map: dict[str, int] = {}
    item_id_map: dict[str, int] = {}
    for user, course in rows:
        if user not in user_id_map:
            user_id_map[user] = len(user_id_map)
        if course not in item_id_map:
            item_id_map[course] = len(item_id_map)

    n_users = len(user_id_map)
    n_items = len(item_id_map)
    print(f"  {n_users} users  {n_items} items")

    # ── 4. 写 mooccube.inter ──────────────────────────────────────────────────
    print("Writing mooccube.inter ...")
    inter_path = MMREC_DIR / "mooccube.inter"
    with open(inter_path, "w", newline="", encoding="utf-8") as f:
        f.write("userID\titemID\tratings\ttimestamp\tx_label\n")
        for ts, (user, course) in enumerate(rows):
            uid = user_id_map[user]
            iid = item_id_map[course]
            if (user, course) in test_set:
                x_label = 2
            elif (user, course) in valid_set:
                x_label = 1
            else:
                x_label = 0
            f.write(f"{uid}\t{iid}\t1\t{ts}\t{x_label}\n")
    print(f"  Written: {inter_path}")

    # ── 5. 提取 FAISS 嵌入 → text_feat.npy ───────────────────────────────────
    print("Loading FAISS index ...")
    try:
        import faiss
    except ImportError:
        print("  ERROR: faiss not available. Run with: conda run -n minicpm python scripts/convert_to_mmrec.py")
        sys.exit(1)

    index = faiss.read_index(str(INDEX_DIR / "course.faiss"))
    faiss_ids: list[str] = json.load(open(INDEX_DIR / "course_ids.json"))
    faiss_id_to_idx = {cid: i for i, cid in enumerate(faiss_ids)}
    dim = index.d  # 1024

    print(f"  FAISS: {index.ntotal} vectors × {dim} dims")
    print("Building text_feat.npy ...")

    text_feat = np.zeros((n_items, dim), dtype=np.float32)
    missing = 0
    for course_id, item_idx in item_id_map.items():
        if course_id in faiss_id_to_idx:
            vec = index.reconstruct(faiss_id_to_idx[course_id])
            text_feat[item_idx] = vec
        else:
            missing += 1

    if missing:
        print(f"  WARNING: {missing} items not in FAISS index (zero vector used)")

    feat_path = MMREC_DIR / "text_feat.npy"
    np.save(feat_path, text_feat)
    print(f"  text_feat.npy shape: {text_feat.shape}  → {feat_path}")

    # ── 6. 保存 ID 映射（供 BPRRetriever / train_bpr.py 使用） ────────────────
    with open(PROCESSED / "user_id_map.json", "w", encoding="utf-8") as f:
        json.dump(user_id_map, f)
    with open(PROCESSED / "item_id_map.json", "w", encoding="utf-8") as f:
        json.dump(item_id_map, f)

    # item_id 反向映射（int index → course_id），用于 BPRRetriever.retrieve
    idx_to_item = {v: k for k, v in item_id_map.items()}
    with open(PROCESSED / "item_idx_list.json", "w", encoding="utf-8") as f:
        # 按整数索引顺序保存为列表
        json.dump([idx_to_item[i] for i in range(n_items)], f)

    print(f"  ID maps saved to {PROCESSED}/")
    print("\nDone! Summary:")
    print(f"  mooccube.inter  : {inter_path}")
    print(f"  text_feat.npy   : {feat_path}  shape={text_feat.shape}")
    print(f"  user_id_map.json: {n_users} entries")
    print(f"  item_id_map.json: {n_items} entries")
    print(f"  item_idx_list.json: {n_items} entries")


if __name__ == "__main__":
    main()
