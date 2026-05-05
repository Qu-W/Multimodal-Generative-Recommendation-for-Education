# coding: utf-8
r"""
HMIF: Hierarchical Modality Interaction Factorization

核心贡献：
1. 模态交互矩阵 W_u ∈ R^{M×M}（用户专属，含跨模态项 w_{vt}, w_{tv}）
   score_modal(u,i) = Tr(M_u · W_u · F_i^T)
                    = Σ_{m1,m2} [W_u]_{m1,m2} · (u_{m1} · i_{m2})
   - 对角项：同模态匹配（视觉↔视觉，文本↔文本）
   - 非对角项：跨模态匹配（用户视觉兴趣 → 物品文本，用户文本兴趣 → 物品视觉）

2. W_u 低秩分解：W_u = W_shared + Δ_u，Δ_u 由 u_cf 动态生成
   - W_shared：所有用户共享的全局模态交互模式
   - Δ_u = f_δ(u_cf)：用户个性化修正（无额外用户参数表，O(d·M²) 参数量）

3. 级联训练（三阶段，物理语义清晰）：
   Stage 1：纯 CF 预热（W_u = 0，只训练协同嵌入）
   Stage 2：对角 W_u（学习单模态匹配，cross-modal 项被掩码）
   Stage 3：完整 W_u（gate 机制从 0 平滑激活跨模态项）

4. 总打分函数（问题4的完整定义）：
   ŷ(u,i) = λ·u_cf·i_cf + Σ_{m1,m2} [W_u]_{m1,m2}·(u_{m1}·i_{m2})
   全量排序：einsum('bpq, bpd, nqd -> bn', W_u, u_modal_b, i_modal)，O(B·M²·N·d/d)=O(B·M²·N)

5. W_shared 对称性约束（可选）：鼓励 w_{vt} ≈ w_{tv}，减少冗余
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss


class HMIF(GeneralRecommender):

    def __init__(self, config, dataset):
        super(HMIF, self).__init__(config, dataset)

        self.embedding_dim  = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']   # d：每个模态的嵌入维度
        self.n_ui_layers    = config['n_ui_layers']
        self.reg_weight     = config['reg_weight']
        self.sym_weight     = config['sym_weight']       # W_shared 对称约束权重
        self.cl_weight      = config['cl_weight']
        self.cf_weight      = config['cf_weight']        # CF 分支权重 λ
        self.dropout        = config['dropout']
        self.tau            = config['tau']
        self.stage2_epoch   = config['stage2_epoch']
        self.stage3_epoch   = config['stage3_epoch']

        self._stage  = 1    # 当前 stage (1/2/3)
        self._epoch  = 0

        # 确定活跃模态数 M
        self.M = (1 if self.v_feat is not None else 0) + \
                 (1 if self.t_feat is not None else 0)
        assert self.M >= 1, 'HMIF 需要至少一个多模态特征文件'

        self.n_nodes = self.n_users + self.n_items
        d = self.feat_embed_dim

        # ── 交互图 ────────────────────────────────────────────────────────
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self._rows = torch.from_numpy(self.interaction_matrix.row).long()
        self._cols = torch.from_numpy(self.interaction_matrix.col).long()
        self.norm_adj = self._build_norm_adj().to(self.device)

        # ── 协同过滤嵌入 ──────────────────────────────────────────────────
        self.user_embedding    = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # ── 物品侧模态投影：raw_feat_dim → d ─────────────────────────────
        # modal_order 记录 [v, t] 的激活顺序，用于 stack 时的一致性
        self.modal_order = []
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.proj_v = nn.Linear(self.v_feat.shape[1], d, bias=False)
            nn.init.xavier_normal_(self.proj_v.weight)
            self.modal_order.append('v')
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.proj_t = nn.Linear(self.t_feat.shape[1], d, bias=False)
            nn.init.xavier_normal_(self.proj_t.weight)
            self.modal_order.append('t')

        # ── 用户侧模态兴趣投影：embedding_dim → d，每个模态一个 ──────────
        # user_proj[m] 将 u_cf 投影到第 m 个模态的兴趣空间
        self.user_proj = nn.ModuleList([
            nn.Linear(self.embedding_dim, d, bias=False) for _ in range(self.M)
        ])
        for proj in self.user_proj:
            nn.init.xavier_normal_(proj.weight)

        # ── 模态交互矩阵参数 ─────────────────────────────────────────────

        # W_shared ∈ R^{M×M}：全局共享的模态交互模式
        # 初始化为单位阵：对角=1（单模态匹配有效），非对角=0（跨模态初始无贡献）
        self.W_shared = nn.Parameter(torch.eye(self.M))

        # f_delta：从 u_cf 生成用户个性化修正 Δ_u ∈ R^{M×M}
        # 初始化权重和偏置为 0：确保训练初期 W_u ≈ W_shared
        self.f_delta = nn.Linear(self.embedding_dim, self.M * self.M, bias=True)
        nn.init.zeros_(self.f_delta.weight)
        nn.init.zeros_(self.f_delta.bias)

        # gate_cross：控制跨模态项的激活强度
        # sigmoid(-4.0) ≈ 0.018，Stage 3 初始时跨模态贡献约 2%，平滑过渡
        self.gate_cross = nn.Parameter(torch.tensor(-4.0))

        # 对角 / 非对角掩码（用于 Stage 切换）
        self.register_buffer('diag_mask',    torch.eye(self.M))
        self.register_buffer('offdiag_mask', 1.0 - torch.eye(self.M))

        # ── 正则化 ────────────────────────────────────────────────────────
        self.reg_loss = EmbLoss()

    # ── 图工具 ─────────────────────────────────────────────────────────────

    def _build_norm_adj(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        M   = self.interaction_matrix
        M_t = M.transpose()
        data = dict(zip(zip(M.row,   M.col + self.n_users), [1] * M.nnz))
        data.update(zip(zip(M_t.row + self.n_users, M_t.col), [1] * M_t.nnz))
        A._update(data)
        sumArr = np.array((A > 0).sum(axis=1)).flatten() + 1e-7
        D_inv  = sp.diags(np.power(sumArr, -0.5))
        L = sp.coo_matrix(D_inv * A * D_inv)
        idx  = torch.LongTensor(np.array([L.row, L.col]))
        vals = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(idx, vals, torch.Size((self.n_nodes, self.n_nodes)))

    # ── 级联 Stage 切换 ────────────────────────────────────────────────────

    def pre_epoch_processing(self):
        """由 Trainer 每轮训练前调用。"""
        self._epoch += 1
        if self._stage == 1 and self._epoch >= self.stage2_epoch:
            self._stage = 2
            # Stage 1→2：W_shared 仍为对角，f_delta 已初始化为零
            # 无需额外操作，平滑过渡
        elif self._stage == 2 and self._epoch >= self.stage3_epoch:
            with torch.no_grad():
                # Stage 2→3：重置跨模态门控，确保从接近零开始激活
                self.gate_cross.fill_(-4.0)
            self._stage = 3

    # ── W_u 计算（含 Stage 控制） ──────────────────────────────────────────

    def _compute_W_u(self, u_cf_b):
        """
        计算 per-user 模态交互矩阵 W_u = W_shared + Δ_u。

        Stage 2：非对角项置零（仅单模态匹配）
        Stage 3：非对角项通过 gate_cross 平滑激活

        Args:
            u_cf_b : [B, embedding_dim]  批量用户协同嵌入
        Returns:
            W_u    : [B, M, M]
        """
        B = u_cf_b.shape[0]
        delta = self.f_delta(u_cf_b).view(B, self.M, self.M)    # [B, M, M]
        W_u   = self.W_shared.unsqueeze(0) + delta               # [B, M, M]

        if self._stage <= 2:
            # 对角掩码：非对角项为 0
            W_u = W_u * self.diag_mask                           # broadcast [B, M, M]
        else:
            # Stage 3：对角项保持全值，非对角项受门控
            g   = torch.sigmoid(self.gate_cross)
            W_u = W_u * (self.diag_mask + g * self.offdiag_mask)

        return W_u

    # ── 模态特征提取 ───────────────────────────────────────────────────────

    def _item_modal_feats(self):
        """
        提取并投影所有物品的模态特征。
        Returns: i_modal [n_items, M, d]
        """
        feats = []
        if 'v' in self.modal_order:
            v = F.normalize(self.proj_v(self.image_embedding.weight))  # [N, d]
            feats.append(v)
        if 't' in self.modal_order:
            t = F.normalize(self.proj_t(self.text_embedding.weight))   # [N, d]
            feats.append(t)
        return torch.stack(feats, dim=1)   # [n_items, M, d]

    def _user_modal_feats(self, u_cf):
        """
        从协同嵌入生成用户多模态兴趣矩阵。
        Returns: u_modal [n_users, M, d]
        """
        feats = [F.normalize(proj(u_cf)) for proj in self.user_proj]
        return torch.stack(feats, dim=1)   # [n_users, M, d]

    # ── 前向传播 ────────────────────────────────────────────────────────────

    def forward(self):
        """
        Returns
        -------
        u_cf    : [n_users, embedding_dim]
        i_cf    : [n_items, embedding_dim]
        u_modal : [n_users, M, d]   用户模态兴趣矩阵 M_u
        i_modal : [n_items, M, d]   物品模态特征矩阵 F_i
        """
        # LightGCN 传播
        ego = torch.cat([self.user_embedding.weight,
                         self.item_id_embedding.weight], dim=0)
        all_embs = [ego]
        x = ego
        for _ in range(self.n_ui_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            all_embs.append(x)
        all_embs = torch.stack(all_embs, dim=1).mean(dim=1)
        u_cf, i_cf = torch.split(all_embs, [self.n_users, self.n_items], dim=0)

        u_modal = self._user_modal_feats(u_cf)    # [n_users, M, d]
        i_modal = self._item_modal_feats()         # [n_items, M, d]

        return u_cf, i_cf, u_modal, i_modal

    # ── 模态打分（einsum 实现） ────────────────────────────────────────────

    def _modal_score_batch(self, W_u, u_modal_b, i_modal_b):
        """
        批量（正/负样本对）模态打分。

        score[b] = Σ_{m1,m2} W_u[b,m1,m2] · (u_modal_b[b,m1] · i_modal_b[b,m2])
                 = einsum('bpq, bpd, bqd -> b', W_u, u_modal_b, i_modal_b)

        等价于 Tr(M_u[b] · W_u[b] · F_i[b]^T) ，即论文中的双线性模态交互。
        """
        return torch.einsum('bpq, bpd, bqd -> b', W_u, u_modal_b, i_modal_b)

    def _modal_score_full(self, W_u, u_modal_b, i_modal_all):
        """
        全量物品模态打分（用于推荐）。

        score[b,n] = Σ_{m1,m2} W_u[b,m1,m2] · (u_modal_b[b,m1] · i_modal_all[n,m2])
                   = einsum('bpq, bpd, nqd -> bn', W_u, u_modal_b, i_modal_all)

        复杂度：O(B · M² · N) ≈ O(B · 4 · N)，等价于 4 次矩阵乘法。
        """
        return torch.einsum('bpq, bpd, nqd -> bn', W_u, u_modal_b, i_modal_all)

    # ── 对比正则化 ────────────────────────────────────────────────────────

    def _cl_loss(self, u_modal, i_modal, users):
        """
        对每个模态 m，将用户的模态兴趣向量 u_{m}
        与其历史交互物品的模态特征均值对齐（InfoNCE）。

        这确保 user_proj 学到有意义的模态偏好，而非退化为随机投影。
        """
        rows = self._rows.to(self.device)
        cols = self._cols.to(self.device)
        cnt  = torch.bincount(rows, minlength=self.n_users).float().to(self.device) + 1e-7

        loss = 0.0
        for m in range(self.M):
            # 物品第 m 模态特征的用户聚合：历史交互物品的均值
            i_feat_m = i_modal[:, m, :]                              # [n_items, d]
            u_agg_m  = torch.zeros(self.n_users, i_feat_m.shape[1], device=self.device)
            u_agg_m.index_add_(0, rows, i_feat_m[cols])
            u_agg_m  = F.normalize(u_agg_m / cnt.unsqueeze(1))      # [n_users, d]

            # 用户第 m 模态兴趣 vs. 历史聚合
            q     = F.normalize(u_modal[:, m, :][users])             # [B, d]
            k_pos = u_agg_m[users]                                   # [B, d]
            pos   = torch.exp((q * k_pos).sum(-1) / self.tau)
            all_  = torch.exp(q @ u_agg_m.T / self.tau).sum(-1)
            loss  = loss + (-torch.log(pos / (all_ + 1e-8))).mean()

        return loss

    # ── BPR 损失 ─────────────────────────────────────────────────────────────

    def _bpr_loss(self, pos, neg):
        return -F.logsigmoid(pos - neg).mean()

    # ── 训练 ─────────────────────────────────────────────────────────────────

    def calculate_loss(self, interaction):
        users     = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_cf, i_cf, u_modal, i_modal = self.forward()

        u_cf_d = F.dropout(u_cf, p=self.dropout, training=self.training)
        i_cf_d = F.dropout(i_cf, p=self.dropout, training=self.training)

        u_b     = u_cf_d[users]
        pos_i_b = i_cf_d[pos_items]
        neg_i_b = i_cf_d[neg_items]

        # ── CF 得分（所有 stage 保留） ───────────────────────────────────
        pos_cf = (u_b * pos_i_b).sum(-1)
        neg_cf = (u_b * neg_i_b).sum(-1)

        pos_scores = self.cf_weight * pos_cf
        neg_scores = self.cf_weight * neg_cf

        # ── 模态交互打分（Stage 2 起启用） ───────────────────────────────
        if self._stage >= 2:
            W_u        = self._compute_W_u(u_b)              # [B, M, M]
            u_modal_b  = u_modal[users]                       # [B, M, d]
            pos_modal_b = i_modal[pos_items]                  # [B, M, d]
            neg_modal_b = i_modal[neg_items]                  # [B, M, d]

            pos_modal = self._modal_score_batch(W_u, u_modal_b, pos_modal_b)
            neg_modal = self._modal_score_batch(W_u, u_modal_b, neg_modal_b)

            pos_scores = pos_scores + pos_modal
            neg_scores = neg_scores + neg_modal

        # ── 各项损失 ─────────────────────────────────────────────────────
        bpr = self._bpr_loss(pos_scores, neg_scores)
        reg = self.reg_loss(u_b, pos_i_b, neg_i_b)

        # W_shared 对称性约束：鼓励 w_{vt} ≈ w_{tv}，减少非对称冗余
        # （允许 sym_weight=0 放开非对称，观察方向性的价值）
        sym = (self.W_shared - self.W_shared.T).norm() ** 2

        cl = self._cl_loss(u_modal, i_modal, users) if self._stage >= 2 else 0.0

        return (bpr
                + self.reg_weight * reg
                + self.sym_weight * sym
                + self.cl_weight  * cl)

    # ── 全量推荐打分 ──────────────────────────────────────────────────────────

    def full_sort_predict(self, interaction):
        """
        ŷ(u, all_i) = λ · u_cf · I_cf^T
                     + Σ_{m1,m2} [W_u]_{m1,m2} · (u_{m1} · I_{m2}^T)

        = cf_weight * (u_cf @ i_cf.T)
        + einsum('bpq, bpd, nqd -> bn', W_u, u_modal_b, i_modal)
        """
        user = interaction[0]
        u_cf, i_cf, u_modal, i_modal = self.forward()

        u_b       = u_cf[user]             # [B, embedding_dim]
        u_modal_b = u_modal[user]          # [B, M, d]

        # CF 得分
        cf_scores = self.cf_weight * (u_b @ i_cf.T)         # [B, n_items]

        # 模态交互得分
        if self._stage >= 2:
            W_u          = self._compute_W_u(u_b)           # [B, M, M]
            modal_scores = self._modal_score_full(W_u, u_modal_b, i_modal)  # [B, n_items]
        else:
            modal_scores = 0.0

        return cf_scores + modal_scores
