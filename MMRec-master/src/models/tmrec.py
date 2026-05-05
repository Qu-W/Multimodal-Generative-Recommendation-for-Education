# coding: utf-8
r"""
TMRec: Tensor-factorized Multimodal Recommendation

Core idea: model user multimodal interests via low-rank tensor decomposition.
A user's preference is disentangled into:
  - u_cf    : collaborative filtering embedding (LightGCN)
  - u_v     : visual interest factor  (projected from u_cf)
  - u_t     : textual interest factor (projected from u_cf)
  - u_cross : cross-modal interest   (u_v ⊙ u_t, Hadamard product)

Scoring function:
  s(u, i) = u_cf · e_i
           + α_v    * (u_v    · proj_v(v_i))
           + α_t    * (u_t    · proj_t(t_i))
           + α_cross * (u_cross · proj_vt(v_i ⊙ t_i))

where α = softmax(W_α u_cf) is an adaptive modality attention weight.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss


class TMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(TMRec, self).__init__(config, dataset)

        def _s(v):
            return float(v[0] if isinstance(v, (list, tuple)) else v)

        self.embedding_dim   = config['embedding_size']
        self.feat_embed_dim  = config['feat_embed_dim']
        self.n_ui_layers     = config['n_ui_layers']
        self.n_mm_layers     = config['n_mm_layers']
        self.reg_weight      = _s(config['reg_weight'])
        self.modal_weight    = _s(config['modal_weight'])
        self.cl_weight       = _s(config['cl_weight'])
        self.dropout         = _s(config['dropout'])
        self.tau             = _s(config['tau'])

        self.n_nodes = self.n_users + self.n_items

        # ── Interaction graph ──────────────────────────────────────────────
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
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

        if self.v_feat is not None and self.t_feat is not None:
            # cross-modal item feature: projects (v_i ⊙ t_i) → R^d
            self.proj_vt = nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            nn.init.xavier_normal_(self.proj_vt.weight)

        # ── User modality interest factors (user side) ─────────────────────
        # Projects collaborative embedding into modality-specific interest spaces
        self.user_proj_v = nn.Linear(self.embedding_dim, self.feat_embed_dim)
        self.user_proj_t = nn.Linear(self.embedding_dim, self.feat_embed_dim)
        nn.init.xavier_normal_(self.user_proj_v.weight)
        nn.init.xavier_normal_(self.user_proj_t.weight)

        # ── Adaptive modality attention ────────────────────────────────────
        # Outputs [α_v, α_t, α_cross] for each user
        n_modal_scores = self._count_active_modals()
        self.modal_attn = nn.Linear(self.embedding_dim, n_modal_scores)
        nn.init.xavier_normal_(self.modal_attn.weight)

        # ── Regularization ─────────────────────────────────────────────────
        self.reg_loss = EmbLoss()

    # ── Graph utilities ────────────────────────────────────────────────────

    def _count_active_modals(self):
        """Number of active scoring heads: CF always on; v, t, cross optional."""
        n = 0
        if self.v_feat is not None:
            n += 1
        if self.t_feat is not None:
            n += 1
        if self.v_feat is not None and self.t_feat is not None:
            n += 1  # cross-modal head
        return max(n, 1)

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
        i    = torch.LongTensor(np.array([L.row, L.col]))
        vals = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, vals, torch.Size((self.n_nodes, self.n_nodes)))

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self):
        """
        Returns
        -------
        u_cf : [n_users, d]   collaborative user embeddings
        i_cf : [n_items, d]   collaborative item embeddings
        v_i  : [n_items, d]   projected visual item features  (or None)
        t_i  : [n_items, d]   projected textual item features (or None)
        """
        # ── LightGCN propagation ───────────────────────────────────────────
        ego = torch.cat([self.user_embedding.weight,
                         self.item_id_embedding.weight], dim=0)
        all_embs = [ego]
        x = ego
        for _ in range(self.n_ui_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            all_embs.append(x)
        all_embs = torch.stack(all_embs, dim=1).mean(dim=1)
        u_cf, i_cf = torch.split(all_embs, [self.n_users, self.n_items], dim=0)

        # ── Item modality features ─────────────────────────────────────────
        v_i, t_i = None, None
        if self.v_feat is not None:
            v_raw = self.image_embedding.weight       # [n_items, raw_v_dim]
            v_i   = F.normalize(self.proj_v(v_raw))  # [n_items, d]
            if self.n_mm_layers > 0:
                v_i = self._propagate_on_graph(v_i)

        if self.t_feat is not None:
            t_raw = self.text_embedding.weight
            t_i   = F.normalize(self.proj_t(t_raw))
            if self.n_mm_layers > 0:
                t_i = self._propagate_on_graph(t_i)

        return u_cf, i_cf, v_i, t_i

    def _propagate_on_graph(self, item_feats):
        """
        Simple 1-hop propagation of item features to users on the UI graph,
        then back to items. Returns enriched item features.
        """
        # Build user features by aggregating interacted items
        rows = torch.from_numpy(self.interaction_matrix.row).long().to(self.device)
        cols = torch.from_numpy(self.interaction_matrix.col).long().to(self.device)
        user_feats = torch.zeros(self.n_users, item_feats.shape[1], device=self.device)
        user_feats.index_add_(0, rows, item_feats[cols])
        deg = torch.bincount(rows, minlength=self.n_users).float().to(self.device) + 1e-7
        user_feats = user_feats / deg.unsqueeze(1)

        # Propagate back to items
        enriched = torch.zeros_like(item_feats)
        enriched.index_add_(0, cols, user_feats[rows])
        deg_i = torch.bincount(cols, minlength=self.n_items).float().to(self.device) + 1e-7
        enriched = enriched / deg_i.unsqueeze(1)
        return F.normalize(item_feats + enriched)

    # ── Scoring ────────────────────────────────────────────────────────────

    def _tensor_score(self, u_cf, i_cf, v_i, t_i, users, items):
        """
        Compute the decomposed tensor-factorized score for a batch.

        Parameters (all already indexed to the batch)
        """
        # Base collaborative score
        s_cf = (u_cf * i_cf).sum(dim=-1)   # [B]

        # Per-user modality interest factors
        u_v = F.normalize(self.user_proj_v(u_cf))  # [B, d]
        u_t = F.normalize(self.user_proj_t(u_cf))  # [B, d]

        # Adaptive attention over modality heads
        attn_logits = self.modal_attn(u_cf)         # [B, n_heads]
        attn = F.softmax(attn_logits, dim=-1)        # [B, n_heads]

        modal_score = torch.zeros(u_cf.shape[0], device=u_cf.device)
        head = 0

        if v_i is not None:
            s_v = (u_v * v_i).sum(dim=-1)
            modal_score = modal_score + attn[:, head] * s_v
            head += 1

        if t_i is not None:
            s_t = (u_t * t_i).sum(dim=-1)
            modal_score = modal_score + attn[:, head] * s_t
            head += 1

        if v_i is not None and t_i is not None:
            # Cross-modal: (u_v ⊙ u_t) · proj_vt(v_i ⊙ t_i)
            u_cross  = u_v * u_t                          # [B, d]
            vt_item  = F.normalize(self.proj_vt(v_i * t_i))  # [B, d]
            s_cross  = (u_cross * vt_item).sum(dim=-1)
            modal_score = modal_score + attn[:, head] * s_cross

        return s_cf + self.modal_weight * modal_score

    # ── Contrastive loss between modality-specific user interests ──────────

    def _cl_loss(self, u_cf, v_i, t_i, users, items):
        """
        Pull the visual-side user representation close to the image-aggregated
        signal, and similarly for text. This encourages u_v / u_t to encode
        genuine modal interests.
        """
        if v_i is None or t_i is None:
            return 0.0

        # Modal-aggregated user signal (mean of interacted items' modal feats)
        rows = torch.from_numpy(self.interaction_matrix.row).long().to(self.device)
        cols = torch.from_numpy(self.interaction_matrix.col).long().to(self.device)

        u_v_agg = torch.zeros(self.n_users, v_i.shape[1], device=self.device)
        u_v_agg.index_add_(0, rows, v_i[cols])
        cnt = torch.bincount(rows, minlength=self.n_users).float().to(self.device) + 1e-7
        u_v_agg = F.normalize(u_v_agg / cnt.unsqueeze(1))

        u_t_agg = torch.zeros(self.n_users, t_i.shape[1], device=self.device)
        u_t_agg.index_add_(0, rows, t_i[cols])
        u_t_agg = F.normalize(u_t_agg / cnt.unsqueeze(1))

        # Projected user interest factors
        u_v_pred = F.normalize(self.user_proj_v(u_cf))  # [n_users, d]
        u_t_pred = F.normalize(self.user_proj_t(u_cf))

        # InfoNCE-style contrastive (batch of users)
        u_v_pred_b = u_v_pred[users]       # [B, d]
        u_v_agg_b  = u_v_agg[users]
        u_t_pred_b = u_t_pred[users]
        u_t_agg_b  = u_t_agg[users]

        loss_v = self._infonce(u_v_pred_b, u_v_agg_b, u_v_agg)
        loss_t = self._infonce(u_t_pred_b, u_t_agg_b, u_t_agg)
        return loss_v + loss_t

    def _infonce(self, q, k_pos, k_all):
        """q, k_pos: [B, d];  k_all: [N, d]"""
        pos_score = torch.exp((q * k_pos).sum(dim=-1) / self.tau)
        all_score = torch.exp(q @ k_all.T / self.tau).sum(dim=-1)
        return -torch.log(pos_score / (all_score + 1e-8)).mean()

    # ── BPR loss ───────────────────────────────────────────────────────────

    def _bpr_loss(self, pos_scores, neg_scores):
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    # ── Training / inference ───────────────────────────────────────────────

    def calculate_loss(self, interaction):
        users     = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_cf, i_cf, v_i, t_i = self.forward()

        # Apply dropout to embeddings during training
        u_cf_d = F.dropout(u_cf, p=self.dropout, training=self.training)
        i_cf_d = F.dropout(i_cf, p=self.dropout, training=self.training)

        # Batch slices
        u_b       = u_cf_d[users]
        pos_i_b   = i_cf_d[pos_items]
        neg_i_b   = i_cf_d[neg_items]
        v_pos     = v_i[pos_items] if v_i is not None else None
        v_neg     = v_i[neg_items] if v_i is not None else None
        t_pos     = t_i[pos_items] if t_i is not None else None
        t_neg     = t_i[neg_items] if t_i is not None else None

        pos_scores = self._tensor_score(u_b, pos_i_b, v_pos, t_pos, users, pos_items)
        neg_scores = self._tensor_score(u_b, neg_i_b, v_neg, t_neg, users, neg_items)

        bpr  = self._bpr_loss(pos_scores, neg_scores)
        reg  = self.reg_loss(u_b, pos_i_b, neg_i_b)
        cl   = self._cl_loss(u_cf, v_i, t_i, users, pos_items)

        return bpr + self.reg_weight * reg + self.cl_weight * cl

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_cf, i_cf, v_i, t_i = self.forward()

        u_b = u_cf[user]           # [B, d]
        u_v = F.normalize(self.user_proj_v(u_b))   # [B, d]
        u_t = F.normalize(self.user_proj_t(u_b))

        attn = F.softmax(self.modal_attn(u_b), dim=-1)  # [B, n_heads]

        # CF score: [B, n_items]
        scores = torch.matmul(u_b, i_cf.T)

        head = 0
        if v_i is not None:
            scores = scores + self.modal_weight * attn[:, head:head+1] * torch.matmul(u_v, v_i.T)
            head += 1
        if t_i is not None:
            scores = scores + self.modal_weight * attn[:, head:head+1] * torch.matmul(u_t, t_i.T)
            head += 1
        if v_i is not None and t_i is not None:
            u_cross = u_v * u_t                              # [B, d]
            vt_all  = F.normalize(self.proj_vt(v_i * t_i))  # [n_items, d]
            scores  = scores + self.modal_weight * attn[:, head:head+1] * torch.matmul(u_cross, vt_all.T)

        return scores
