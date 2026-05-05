"""
scripts/train_bpr.py
轻量级 BPR（Bayesian Personalized Ranking）协同过滤训练。

读取（由 convert_to_mmrec.py 生成）:
  data/processed/interactions.csv
  data/processed/valid.json
  data/processed/test.json
  data/processed/user_id_map.json
  data/processed/item_id_map.json

输出:
  data/processed/bpr_embeddings.npz  — {"user_emb": (n_users, dim), "item_emb": (n_items, dim)}

用法:
  python scripts/train_bpr.py [--dim 64] [--epochs 50] [--lr 0.001] [--batch 2048]

训练后通过 BPRRetriever.load() 加载嵌入用于推荐。
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_data():
    map_path = PROCESSED / "user_id_map.json"
    if not map_path.exists():
        print("ERROR: user_id_map.json not found. Run convert_to_mmrec.py first.")
        sys.exit(1)

    with open(map_path) as f:
        u_map: dict[str, int] = json.load(f)
    with open(PROCESSED / "item_id_map.json") as f:
        i_map: dict[str, int] = json.load(f)
    with open(PROCESSED / "valid.json") as f:
        valid_data = json.load(f)
    with open(PROCESSED / "test.json") as f:
        test_data = json.load(f)

    # held-out pairs (int ids)
    valid_set = {
        (u_map[u], i_map[c])
        for u, cs in valid_data.items()
        for c in cs
        if u in u_map and c in i_map
    }
    test_set = {
        (u_map[u], i_map[c])
        for u, cs in test_data.items()
        for c in cs
        if u in u_map and c in i_map
    }
    held_out = valid_set | test_set

    # train interactions + per-user item sets for negative sampling
    train_pairs = []
    user_items: dict[int, set] = {}
    with open(PROCESSED / "interactions.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u, c = row["user_id"], row["course_id"]
            if u not in u_map or c not in i_map:
                continue
            uid, iid = u_map[u], i_map[c]
            user_items.setdefault(uid, set()).add(iid)
            if (uid, iid) not in held_out:
                train_pairs.append((uid, iid))

    n_users = len(u_map)
    n_items = len(i_map)
    return train_pairs, user_items, n_users, n_items, valid_set, test_set


# ── BPR 模型 ──────────────────────────────────────────────────────────────────

class BPRModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, users, pos_items, neg_items):
        u  = self.user_emb(users)
        pi = self.item_emb(pos_items)
        ni = self.item_emb(neg_items)
        pos_score = (u * pi).sum(-1)
        neg_score = (u * ni).sum(-1)
        bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
        reg_loss  = (u.norm(2).pow(2) + pi.norm(2).pow(2) + ni.norm(2).pow(2)) / len(users)
        return bpr_loss + 1e-4 * reg_loss

    def score_all_items(self, user_ids):
        """[B, n_items] 分数矩阵，用于全量排序评测"""
        u = self.user_emb(user_ids)
        return u @ self.item_emb.weight.T


# ── 评测（在 valid/test 集上采样 1000 用户） ──────────────────────────────────

def evaluate(model, eval_pairs, user_items, n_items, device, k=10, n_eval=1000):
    model.eval()
    # 按 user 分组
    user_gt: dict[int, set] = {}
    for uid, iid in eval_pairs:
        user_gt.setdefault(uid, set()).add(iid)

    eval_users = list(user_gt.keys())
    if len(eval_users) > n_eval:
        eval_users = random.sample(eval_users, n_eval)

    hits, ndcgs = [], []
    u_tensor = torch.tensor(eval_users, device=device)
    with torch.no_grad():
        scores_mat = model.score_all_items(u_tensor).cpu().numpy()  # [B, n_items]

    for idx, uid in enumerate(eval_users):
        gt = user_gt[uid]
        scores = scores_mat[idx].copy()
        # 排除训练集已见商品（保留 gt）
        exclude = user_items.get(uid, set()) - gt
        if exclude:
            scores[list(exclude)] = -1e9
        top_k = np.argsort(-scores)[:k]
        hit = len(gt & set(top_k))
        hits.append(hit / min(len(gt), k))
        if hit > 0:
            ranks = [r + 1 for r, x in enumerate(top_k) if x in gt]
            ndcgs.append(
                sum(1 / np.log2(r + 1) for r in ranks) /
                sum(1 / np.log2(i + 2) for i in range(min(len(gt), k)))
            )
        else:
            ndcgs.append(0.0)

    return float(np.mean(hits)), float(np.mean(ndcgs))


# ── 主训练循环 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train BPR for EduRec CF retrieval")
    parser.add_argument("--dim",    type=int,   default=64,    help="Embedding dimension")
    parser.add_argument("--epochs", type=int,   default=50,    help="Training epochs")
    parser.add_argument("--lr",     type=float, default=1e-3,  help="Learning rate")
    parser.add_argument("--batch",  type=int,   default=2048,  help="Batch size")
    parser.add_argument("--eval_k", type=int,   default=10,    help="Top-K for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data ...")
    train_pairs, user_items, n_users, n_items, valid_set, test_set = load_data()
    print(f"  {n_users} users  {n_items} items  {len(train_pairs)} train pairs")

    all_items = list(range(n_items))
    model = BPRModel(n_users, n_items, args.dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_recall = 0.0
    best_state  = None

    for ep in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_pairs)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, len(train_pairs), args.batch):
            batch = train_pairs[start : start + args.batch]
            users    = torch.tensor([p[0] for p in batch], device=device)
            pos_itms = torch.tensor([p[1] for p in batch], device=device)
            # uniform negative sampling (simple but effective for BPR)
            neg_itms = torch.tensor(
                [random.choice(all_items) for _ in batch], device=device
            )
            optimizer.zero_grad()
            loss = model(users, pos_itms, neg_itms)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)

        if ep % 5 == 0 or ep == 1:
            recall, ndcg = evaluate(
                model, valid_set, user_items, n_items, device, k=args.eval_k
            )
            marker = " ◀ best" if recall > best_recall else ""
            print(
                f"Epoch {ep:3d}  loss={avg_loss:.4f}  "
                f"Recall@{args.eval_k}={recall:.4f}  NDCG@{args.eval_k}={ndcg:.4f}{marker}"
            )
            if recall > best_recall:
                best_recall = recall
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            print(f"Epoch {ep:3d}  loss={avg_loss:.4f}")

    # ── 保存最优嵌入 ──────────────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
        print(f"\nRestored best checkpoint (val Recall@{args.eval_k}={best_recall:.4f})")

    model.eval()
    with torch.no_grad():
        user_emb = model.user_emb.weight.cpu().numpy()
        item_emb = model.item_emb.weight.cpu().numpy()

    out_path = PROCESSED / "bpr_embeddings.npz"
    np.savez(out_path, user_emb=user_emb, item_emb=item_emb)
    print(f"Saved: {out_path}")
    print(f"  user_emb: {user_emb.shape}  item_emb: {item_emb.shape}")

    # ── 最终 Test 评测 ────────────────────────────────────────────────────────
    recall_t, ndcg_t = evaluate(
        model, test_set, user_items, n_items, device, k=args.eval_k, n_eval=2000
    )
    print(f"Test  Recall@{args.eval_k}={recall_t:.4f}  NDCG@{args.eval_k}={ndcg_t:.4f}")


if __name__ == "__main__":
    main()
