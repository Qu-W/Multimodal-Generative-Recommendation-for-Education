# coding: utf-8
r"""
TMRecPlus: TFN-Enhanced Tensor Multimodal Recommendation

Improvements over TMRec:

1. TFN-style 1-augmented modal vectors:
     ũ = [1; u_v; u_t]  ∈ R^(2d+1)    (user)
     ĩ = [1; v_i; t_i]  ∈ R^(2d+1)    (item)
   The bilinear form  score_modal = ũᵀ W ĩ  (W = PᵀQ, low-rank)
   automatically captures ALL interaction types:
     ─ unimodal:    u_v·v_i,  u_t·t_i
     ─ cross-modal: u_v·t_i  (user visual interest ↔ item text)
                    u_t·v_i  (user text interest   ↔ item visual)
     ─ bias terms via the leading "1"
   TMRec only approximates this with Hadamard products (diagonal of the
   outer product) and a hand-crafted softmax attention.

2. Low-rank bilinear W = PᵀQ  (P, Q ∈ R^{rank × aug_dim})
   replaces TMRec's manual [α_v, α_t, α_cross] = softmax(W_α u_cf).
   All interaction weights are learned end-to-end; no inductive bias
   needed about which modality combinations matter.

3. Cross-modal CL (new):
   Besides intra-modal alignment (u_v ↔ visual-aggregated profile),
   TMRecPlus adds a cross-modal contrastive term that aligns u_v and u_t
   for the SAME user against other users. This enforces semantic
   compatibility between a user's visual and textual interest factors,
   acting as a soft cross-modal transfer regulariser.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss


class TMRecPlus(GeneralRecommender):
    def __init__(self, config, dataset):
        super(TMRecPlus, self).__init__(config, dataset)

        def _s(v):   # extract scalar from list (yaml may pass list before grid search)
            return float(v[0] if isinstance(v, (list, tuple)) else v)
        def _si(v):
            return int(v[0] if isinstance(v, (list, tuple)) else v)

        self.embedding_dim  = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_ui_layers    = config['n_ui_layers']
        self.n_mm_layers    = config['n_mm_layers']
        self.reg_weight     = _s(config['reg_weight'])
        self.modal_weight   = _s(config['modal_weight'])
        self.cl_weight      = _s(config['cl_weight'])
        self.cl_cross_weight = _s(config['cl_cross_weight'])
        self.dropout        = _s(config['dropout'])
        self.tau            = _s(config['tau'])
        self.rank           = _si(config['rank'])     # low-rank bilinear rank r

        self.n_nodes = self.n_users + self.n_items

        # ── Interaction graph ──────────────────────────────────────────────
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self._rows = torch.from_numpy(self.interaction_matrix.row).long()
        self._cols = torch.from_numpy(self.interaction_matrix.col).long()
        self.norm_adj = self._build_norm_adj().to(self.device)

        # ── Collaborative embeddings ───────────────────────────────────────
        self.user_embedding    = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # ── Modality feature projections (item side) ───────────────────────
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.proj_v = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.proj_v.weight)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.proj_t = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.proj_t.weight)

        # ── User modality interest projections ─────────────────────────────
        self.user_proj_v = nn.Linear(self.embedding_dim, self.feat_embed_dim)
        self.user_proj_t = nn.Linear(self.embedding_dim, self.feat_embed_dim)
        nn.init.xavier_normal_(self.user_proj_v.weight)
        nn.init.xavier_normal_(self.user_proj_t.weight)

        # ── Low-rank bilinear W = P^T Q ────────────────────────────────────
        # aug_dim = 1 (constant) + feat_embed_dim per active modality
        self.aug_dim = 1
        if self.v_feat is not None:
            self.aug_dim += self.feat_embed_dim
        if self.t_feat is not None:
            self.aug_dim += self.feat_embed_dim

        # P maps user aug vectors → R^rank
        # Q maps item aug vectors → R^rank
        self.bilinear_P = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(self.rank, self.aug_dim)))
        self.bilinear_Q = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(self.rank, self.aug_dim)))

        # ── Regularization ─────────────────────────────────────────────────
        self.reg_loss = EmbLoss()

    # ── Graph utilities ────────────────────────────────────────────────────

    def _build_norm_adj(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M   = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),   [1] * inter_M.nnz))
        data.update(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz))
        A._update(data)
        sumArr = np.array((A > 0).sum(axis=1)).flatten() + 1e-7
        diag   = np.power(sumArr, -0.5)
        D = sp.diags(diag)
        L = sp.coo_matrix(D * A * D)
        idx  = torch.LongTensor(np.array([L.row, L.col]))
        vals = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(idx, vals, torch.Size((self.n_nodes, self.n_nodes)))

    def _propagate_modal(self, item_feats):
        """1-hop UI-graph propagation to enrich item modal features."""
        rows = self._rows.to(self.device)
        cols = self._cols.to(self.device)

        u_feats = torch.zeros(self.n_users, item_feats.shape[1], device=self.device)
        u_feats.index_add_(0, rows, item_feats[cols])
        deg_u = torch.bincount(rows, minlength=self.n_users).float().to(self.device) + 1e-7
        u_feats = u_feats / deg_u.unsqueeze(1)

        enriched = torch.zeros_like(item_feats)
        enriched.index_add_(0, cols, u_feats[rows])
        deg_i = torch.bincount(cols, minlength=self.n_items).float().to(self.device) + 1e-7
        enriched = enriched / deg_i.unsqueeze(1)
        return F.normalize(item_feats + enriched)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self):
        """
        Returns
        -------
        u_cf : [n_users, d]
        i_cf : [n_items, d]
        v_i  : [n_items, feat_embed_dim] or None
        t_i  : [n_items, feat_embed_dim] or None
        """
        # LightGCN
        ego = torch.cat([self.user_embedding.weight,
                         self.item_id_embedding.weight], dim=0)
        all_embs = [ego]
        x = ego
        for _ in range(self.n_ui_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            all_embs.append(x)
        all_embs = torch.stack(all_embs, dim=1).mean(dim=1)
        u_cf, i_cf = torch.split(all_embs, [self.n_users, self.n_items], dim=0)

        # Item modal features
        v_i, t_i = None, None
        if self.v_feat is not None:
            v_i = F.normalize(self.proj_v(self.image_embedding.weight))
            if self.n_mm_layers > 0:
                v_i = self._propagate_modal(v_i)
        if self.t_feat is not None:
            t_i = F.normalize(self.proj_t(self.text_embedding.weight))
            if self.n_mm_layers > 0:
                t_i = self._propagate_modal(t_i)

        return u_cf, i_cf, v_i, t_i

    # ── Augmented vectors & bilinear scoring ───────────────────────────────

    def _user_aug(self, u_cf_b):
        """Build TFN-style augmented user vector: [1; u_v; u_t]."""
        B = u_cf_b.shape[0]
        parts = [torch.ones(B, 1, device=self.device)]
        if self.v_feat is not None:
            parts.append(F.normalize(self.user_proj_v(u_cf_b)))
        if self.t_feat is not None:
            parts.append(F.normalize(self.user_proj_t(u_cf_b)))
        return torch.cat(parts, dim=-1)   # [B, aug_dim]

    def _item_aug(self, v_feats, t_feats):
        """Build TFN-style augmented item vector: [1; v_i; t_i]."""
        # v_feats / t_feats: [N, d] or None
        ref = v_feats if v_feats is not None else t_feats
        N = ref.shape[0]
        parts = [torch.ones(N, 1, device=self.device)]
        if v_feats is not None:
            parts.append(v_feats)
        if t_feats is not None:
            parts.append(t_feats)
        return torch.cat(parts, dim=-1)   # [N, aug_dim]

    def _bilinear_score_batch(self, u_aug, i_aug):
        """
        Low-rank bilinear score for a matched batch.
        u_aug, i_aug: [B, aug_dim]  → [B]
        """
        u_proj = u_aug @ self.bilinear_P.T    # [B, rank]
        i_proj = i_aug @ self.bilinear_Q.T    # [B, rank]
        return (u_proj * i_proj).sum(dim=-1)   # [B]

    def _bilinear_score_full(self, u_aug, i_aug_all):
        """
        Low-rank bilinear score for all items.
        u_aug: [B, aug_dim], i_aug_all: [N, aug_dim] → [B, N]
        """
        u_proj = u_aug       @ self.bilinear_P.T   # [B, rank]
        i_proj = i_aug_all   @ self.bilinear_Q.T   # [N, rank]
        return u_proj @ i_proj.T                    # [B, N]

    # ── CL losses ──────────────────────────────────────────────────────────

    def _agg_modal_user(self, item_feats):
        """Aggregate item modal features to build user-side modal profiles."""
        rows = self._rows.to(self.device)
        cols = self._cols.to(self.device)
        u_agg = torch.zeros(self.n_users, item_feats.shape[1], device=self.device)
        u_agg.index_add_(0, rows, item_feats[cols])
        cnt = torch.bincount(rows, minlength=self.n_users).float().to(self.device) + 1e-7
        return F.normalize(u_agg / cnt.unsqueeze(1))

    def _infonce(self, q, k_pos, k_all):
        """InfoNCE: q, k_pos [B,d]; k_all [N,d]."""
        pos = torch.exp((q * k_pos).sum(-1) / self.tau)
        all_ = torch.exp(q @ k_all.T / self.tau).sum(-1)
        return -torch.log(pos / (all_ + 1e-8)).mean()

    def _intra_modal_cl(self, u_cf, v_i, t_i, users):
        """
        Intra-modal: align user projected interest with aggregated item modal profile.
        Same as TMRec baseline CL.
        """
        loss = 0.0
        u_v_pred = F.normalize(self.user_proj_v(u_cf))
        u_t_pred = F.normalize(self.user_proj_t(u_cf))

        if v_i is not None:
            u_v_agg = self._agg_modal_user(v_i)
            loss = loss + self._infonce(u_v_pred[users], u_v_agg[users], u_v_agg)
        if t_i is not None:
            u_t_agg = self._agg_modal_user(t_i)
            loss = loss + self._infonce(u_t_pred[users], u_t_agg[users], u_t_agg)
        return loss

    def _cross_modal_cl(self, u_cf, users):
        """
        Cross-modal (NEW vs TMRec): for the same user, u_v and u_t should be
        semantically compatible. Contrast against other users in the batch.

        Intuition: a user interested in sports shoes (visual) should also have
        textual interest in athletic descriptions — the two factors co-vary.
        """
        if self.v_feat is None or self.t_feat is None:
            return 0.0

        u_v_all = F.normalize(self.user_proj_v(u_cf))   # [n_users, d]
        u_t_all = F.normalize(self.user_proj_t(u_cf))

        u_v_b = u_v_all[users]   # [B, d]
        u_t_b = u_t_all[users]   # [B, d]

        # Bidirectional: v→t and t→v
        loss_vt = self._infonce(u_v_b, u_t_b, u_t_all)
        loss_tv = self._infonce(u_t_b, u_v_b, u_v_all)
        return loss_vt + loss_tv

    # ── BPR ────────────────────────────────────────────────────────────────

    def _bpr_loss(self, pos, neg):
        return -F.logsigmoid(pos - neg).mean()

    # ── Training / inference ───────────────────────────────────────────────

    def calculate_loss(self, interaction):
        users     = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_cf, i_cf, v_i, t_i = self.forward()

        u_cf_d = F.dropout(u_cf, p=self.dropout, training=self.training)
        i_cf_d = F.dropout(i_cf, p=self.dropout, training=self.training)

        u_b       = u_cf_d[users]
        pos_i_b   = i_cf_d[pos_items]
        neg_i_b   = i_cf_d[neg_items]

        # ── CF score ───────────────────────────────────────────────────────
        pos_cf = (u_b * pos_i_b).sum(-1)
        neg_cf = (u_b * neg_i_b).sum(-1)

        # ── Bilinear modal score ───────────────────────────────────────────
        u_aug   = self._user_aug(u_b)
        v_pos   = v_i[pos_items] if v_i is not None else None
        t_pos   = t_i[pos_items] if t_i is not None else None
        v_neg   = v_i[neg_items] if v_i is not None else None
        t_neg   = t_i[neg_items] if t_i is not None else None

        pos_aug = self._item_aug(v_pos, t_pos)
        neg_aug = self._item_aug(v_neg, t_neg)

        pos_modal = self._bilinear_score_batch(u_aug, pos_aug)
        neg_modal = self._bilinear_score_batch(u_aug, neg_aug)

        pos_scores = pos_cf + self.modal_weight * pos_modal
        neg_scores = neg_cf + self.modal_weight * neg_modal

        bpr = self._bpr_loss(pos_scores, neg_scores)
        reg = self.reg_loss(u_b, pos_i_b, neg_i_b)

        # ── CL losses ──────────────────────────────────────────────────────
        cl_intra = self._intra_modal_cl(u_cf, v_i, t_i, users)
        cl_cross = self._cross_modal_cl(u_cf, users)

        return (bpr
                + self.reg_weight   * reg
                + self.cl_weight    * cl_intra
                + self.cl_cross_weight * cl_cross)

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_cf, i_cf, v_i, t_i = self.forward()

        u_b   = u_cf[user]                      # [B, d]
        u_aug = self._user_aug(u_b)              # [B, aug_dim]
        i_aug = self._item_aug(v_i, t_i)         # [n_items, aug_dim]

        # CF scores
        scores = u_b @ i_cf.T                    # [B, n_items]

        # Low-rank bilinear modal scores
        scores = scores + self.modal_weight * self._bilinear_score_full(u_aug, i_aug)

        return scores
