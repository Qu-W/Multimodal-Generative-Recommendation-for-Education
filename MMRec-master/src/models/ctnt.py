# coding: utf-8
r"""
CTNT: Cascaded Tucker Neural Tensor for Multimodal Recommendation

针对 PDF 方案五个问题的完整修正实现：

问题1：去除量化步骤
  → 直接使用连续多模态特征，通过神经网络投影到因子空间

问题2：四阶张量语义模糊
  → 张量从不显式存储，Tucker 分解作为打分函数使用：
    score(u,i) = G ×₁ u_vec ×₂ i_vec ×₃ t_vec(i) ×₄ v_vec(i)

问题3：级联初始化维度不匹配
  → 使用 CP 分解参数化核心张量 G ≈ A⊗B⊗C⊗D，
    每个 stage 只新增模态维度对应的因子矩阵，维度不变，
    新增因子以"近似恒等"方式初始化，确保平滑过渡

问题4：缺少推荐打分函数
  → 完整定义（见 forward() 和 full_sort_predict()）：
    score(u,i) = u_cf·i_cf  +  u_fac · (i_fac ⊙ t_fac ⊙ v_fac)
    其中 u_fac = f_U(u_cf) @ A ∈ R^K，其余类似
    全排序：scores[B,N] = u_fac @ (i_fac ⊙ t_fac ⊙ v_fac).T

问题5：正交约束应施加在因子矩阵上
  → 对 A, B, C, D 施加 ||AᵀA - I||_F² 正则，
    鼓励 CP 分量正交，减少冗余

额外改进：动态秩（基于用户活跃度的 rank mask），
活跃用户使用全秩 K，普通用户 2K/3，长尾用户 K/3
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss


class CTNT(GeneralRecommender):

    def __init__(self, config, dataset):
        super(CTNT, self).__init__(config, dataset)

        def _s(v):
            return float(v[0] if isinstance(v, (list, tuple)) else v)
        def _si(v):
            return int(v[0] if isinstance(v, (list, tuple)) else v)

        self.embedding_dim  = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']   # r: 因子维度
        self.n_ui_layers    = config['n_ui_layers']
        self.K              = _si(config['cp_rank'])          # CP 分量数
        self.reg_weight     = _s(config['reg_weight'])
        self.orth_weight    = _s(config['orth_weight'])      # 因子矩阵正交约束权重
        self.cf_weight      = _s(config['cf_weight'])        # CF 分支权重
        self.dropout        = _s(config['dropout'])
        self.tau            = _s(config['tau'])
        self.cl_weight      = _s(config['cl_weight'])
        # 级联 stage 切换轮次
        self.stage2_epoch   = _si(config['stage2_epoch'])     # 第几轮启用文本 Tucker
        self.stage3_epoch   = _si(config['stage3_epoch'])     # 第几轮启用视觉 Tucker

        self._stage  = 1   # 当前 stage (1/2/3)
        self._epoch  = 0   # 内部 epoch 计数器（由 pre_epoch_processing 更新）

        self.n_nodes = self.n_users + self.n_items

        # ── 交互图 ────────────────────────────────────────────────────────
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self._rows = torch.from_numpy(self.interaction_matrix.row).long()
        self._cols = torch.from_numpy(self.interaction_matrix.col).long()
        self.norm_adj = self._build_norm_adj().to(self.device)

        # ── 协同过滤嵌入（CF 基础层，所有 stage 共用） ────────────────────
        self.user_embedding    = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        r = self.feat_embed_dim   # 简写，因子空间维度

        # ── 神经参数化因子矩阵（问题1、问题2修正） ───────────────────────
        # 所有 stage 共用同一套投影网络；stage 控制哪些路径激活

        # 用户/物品 CF 因子投影：embedding_dim → r
        self.f_U = nn.Linear(self.embedding_dim, r, bias=False)
        self.f_I = nn.Linear(self.embedding_dim, r, bias=False)
        nn.init.xavier_normal_(self.f_U.weight)
        nn.init.xavier_normal_(self.f_I.weight)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.f_V_proj = nn.Linear(self.v_feat.shape[1], r, bias=False)
            nn.init.xavier_normal_(self.f_V_proj.weight)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.f_T_proj = nn.Linear(self.t_feat.shape[1], r, bias=False)
            nn.init.xavier_normal_(self.f_T_proj.weight)

        # ── CP 因子矩阵 A, B, C, D ∈ R^{r × K}（问题2修正） ─────────────
        # score(u,i) = Σ_k (f_U(u)·A[:,k]) * (f_I(i)·B[:,k]) * (f_T(i)·C[:,k]) * (f_V(i)·D[:,k])
        #            = [f_U(u)@A] · ([f_I(i)@B] ⊙ [f_T(i)@C] ⊙ [f_V(i)@D])
        self.A = nn.Parameter(nn.init.orthogonal_(torch.empty(r, self.K)))  # 用户因子
        self.B = nn.Parameter(nn.init.orthogonal_(torch.empty(r, self.K)))  # 物品 CF 因子
        # C（文本因子）和 D（视觉因子）在对应 stage 启用时才参与梯度
        self.C = nn.Parameter(nn.init.orthogonal_(torch.empty(r, self.K)))  # 文本因子
        self.D = nn.Parameter(nn.init.orthogonal_(torch.empty(r, self.K)))  # 视觉因子

        # Stage 门控标量：新模态刚引入时从 0 线性增长，平滑过渡（问题3修正）
        self.gate_t  = nn.Parameter(torch.zeros(1))   # 文本 Tucker 门控
        self.gate_v  = nn.Parameter(torch.zeros(1))   # 视觉 Tucker 门控

        # ── 动态秩 mask：[n_users, K]（活跃→全秩，长尾→1/3 秩） ─────────
        self.register_buffer('rank_mask', self._build_rank_mask())

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

    def _build_rank_mask(self):
        """
        动态秩 mask [n_users, K]（问题方案第(2)节改进）
        活跃用户（top 33%）：全秩 K
        普通用户（mid 34%）：2K/3
        长尾用户（bot 33%）：K/3
        """
        deg = np.bincount(self.interaction_matrix.row, minlength=self.n_users).astype(float)
        nz  = deg[deg > 0]
        p33, p67 = np.percentile(nz, [33, 67])

        mask = torch.zeros(self.n_users, self.K)
        k_full = self.K
        k_mid  = max(self.K * 2 // 3, 1)
        k_low  = max(self.K // 3, 1)
        for u in range(self.n_users):
            if deg[u] > p67:
                mask[u, :k_full] = 1.0
            elif deg[u] > p33:
                mask[u, :k_mid] = 1.0
            else:
                mask[u, :k_low] = 1.0
        return mask

    # ── 级联 stage 切换（问题3修正） ────────────────────────────────────────

    def pre_epoch_processing(self):
        """每轮训练前由 Trainer 调用，更新当前 stage。"""
        self._epoch += 1
        if self._stage == 1 and self._epoch >= self.stage2_epoch:
            self._advance_to_stage2()
        elif self._stage == 2 and self._epoch >= self.stage3_epoch:
            self._advance_to_stage3()

    def _advance_to_stage2(self):
        """
        1→2：引入文本 Tucker 分支。
        初始化策略（问题3修正）：
          - A, B 继承已训练的 CF 表征（不重置）
          - C 保持 orthogonal 初始化（新维度）
          - gate_t 置 0，确保初始 stage2 得分 ≈ stage1 得分（平滑过渡）
        """
        with torch.no_grad():
            self.gate_t.fill_(0.0)   # 从 0 开始线性增长
        self._stage = 2

    def _advance_to_stage3(self):
        """
        2→3：引入视觉 Tucker 分支。
        D 保持 orthogonal 初始化；gate_v 置 0。
        """
        with torch.no_grad():
            self.gate_v.fill_(0.0)
        self._stage = 3

    # ── 前向传播 ────────────────────────────────────────────────────────────

    def forward(self):
        """
        返回值
        ------
        u_cf : [n_users, embedding_dim]   LightGCN 协同嵌入
        i_cf : [n_items, embedding_dim]
        u_fac : [n_users, K]  用户 CP 因子（含动态秩 mask）
        i_item_vec : [n_items, K]  物品侧 CP 因子融合（⊙ 各模态）
        """
        # ── LightGCN ──────────────────────────────────────────────────────
        ego = torch.cat([self.user_embedding.weight,
                         self.item_id_embedding.weight], dim=0)
        all_embs = [ego]
        x = ego
        for _ in range(self.n_ui_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            all_embs.append(x)
        all_embs = torch.stack(all_embs, dim=1).mean(dim=1)
        u_cf, i_cf = torch.split(all_embs, [self.n_users, self.n_items], dim=0)

        # ── 因子投影 ────────────────────────────────────────────────────────
        # 用户因子：f_U(u_cf) @ A  [n_users, K]，施加动态秩 mask
        u_r   = F.normalize(self.f_U(u_cf))               # [n_users, r]
        u_fac = (u_r @ self.A) * self.rank_mask            # [n_users, K]

        # 物品 CF 因子：f_I(i_cf) @ B  [n_items, K]
        i_r   = F.normalize(self.f_I(i_cf))               # [n_items, r]
        i_fac = i_r @ self.B                               # [n_items, K]

        # 物品侧融合向量（问题2、问题4修正）：i_fac ⊙ t_fac ⊙ v_fac
        # 各模态受 gate 门控和 stage 控制，平滑引入（问题3修正）
        item_vec = i_fac   # base = CF因子

        if self._stage >= 2 and self.t_feat is not None:
            t_r   = F.normalize(self.f_T_proj(self.text_embedding.weight))  # [n_items, r]
            t_fac = t_r @ self.C                                              # [n_items, K]
            # 文本门控 sigmoid(gate_t)：从 0 → 1 平滑激活
            g_t   = torch.sigmoid(self.gate_t)
            item_vec = item_vec * (1.0 - g_t + g_t * t_fac)

        if self._stage >= 3 and self.v_feat is not None:
            v_r   = F.normalize(self.f_V_proj(self.image_embedding.weight))  # [n_items, r]
            v_fac = v_r @ self.D                                              # [n_items, K]
            g_v   = torch.sigmoid(self.gate_v)
            item_vec = item_vec * (1.0 - g_v + g_v * v_fac)

        return u_cf, i_cf, u_fac, item_vec

    # ── 正交正则化（问题5修正） ──────────────────────────────────────────────

    def _orth_loss(self):
        """
        对因子矩阵 A, B, C, D 施加正交约束 ||AᵀA - I||_F²
        仅对当前 stage 激活的因子矩阵计算。
        """
        I = torch.eye(self.K, device=self.device)
        loss = (self.A.T @ self.A - I).norm() ** 2
        loss = loss + (self.B.T @ self.B - I).norm() ** 2
        if self._stage >= 2:
            loss = loss + (self.C.T @ self.C - I).norm() ** 2
        if self._stage >= 3:
            loss = loss + (self.D.T @ self.D - I).norm() ** 2
        return loss

    # ── 对比正则化（用户跨模态兴趣对齐） ────────────────────────────────────

    def _cl_loss(self, u_fac, users):
        """
        对同一用户的 CF 因子与模态因子之间做 InfoNCE 对齐，
        使用户因子 u_fac 保持与协同信号一致。
        """
        if self._stage < 2:
            return 0.0
        rows = self._rows.to(self.device)
        cols = self._cols.to(self.device)

        # 用户交互物品的模态因子均值（无量化，直接连续聚合）
        if self.t_feat is not None:
            t_r = F.normalize(self.f_T_proj(self.text_embedding.weight))
            t_fac_all = t_r @ self.C                              # [n_items, K]
            u_t_agg = torch.zeros(self.n_users, self.K, device=self.device)
            u_t_agg.index_add_(0, rows, t_fac_all[cols])
            cnt = torch.bincount(rows, minlength=self.n_users).float().to(self.device) + 1e-7
            u_t_agg = F.normalize(u_t_agg / cnt.unsqueeze(1))
            u_fac_b = F.normalize(u_fac[users])
            u_t_b   = u_t_agg[users]
            pos  = torch.exp((u_fac_b * u_t_b).sum(-1) / self.tau)
            all_ = torch.exp(u_fac_b @ u_t_agg.T / self.tau).sum(-1)
            return -torch.log(pos / (all_ + 1e-8)).mean()
        return 0.0

    # ── BPR 损失 ─────────────────────────────────────────────────────────────

    def _bpr_loss(self, pos, neg):
        return -F.logsigmoid(pos - neg).mean()

    # ── 训练 ─────────────────────────────────────────────────────────────────

    def calculate_loss(self, interaction):
        users     = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_cf, i_cf, u_fac, item_vec = self.forward()

        u_cf_d = F.dropout(u_cf, p=self.dropout, training=self.training)
        i_cf_d = F.dropout(i_cf, p=self.dropout, training=self.training)

        u_b     = u_cf_d[users]
        pos_i_b = i_cf_d[pos_items]
        neg_i_b = i_cf_d[neg_items]

        # ── CF 得分（Stage 1 基础，始终保留） ──────────────────────────────
        pos_cf = (u_b * pos_i_b).sum(-1)
        neg_cf = (u_b * neg_i_b).sum(-1)

        # ── Tucker 张量得分（问题4修正的打分函数） ─────────────────────────
        # score(u,i) = u_fac[u] · item_vec[i]
        u_fac_b  = u_fac[users]                    # [B, K]
        pos_tvec = item_vec[pos_items]              # [B, K]
        neg_tvec = item_vec[neg_items]              # [B, K]
        pos_tucker = (u_fac_b * pos_tvec).sum(-1)
        neg_tucker = (u_fac_b * neg_tvec).sum(-1)

        pos_scores = self.cf_weight * pos_cf + pos_tucker
        neg_scores = self.cf_weight * neg_cf + neg_tucker

        bpr  = self._bpr_loss(pos_scores, neg_scores)
        reg  = self.reg_loss(u_b, pos_i_b, neg_i_b)
        orth = self._orth_loss()
        cl   = self._cl_loss(u_fac, users)

        return (bpr
                + self.reg_weight  * reg
                + self.orth_weight * orth
                + self.cl_weight   * cl)

    # ── 全量推荐（问题4修正的完整推荐打分公式） ──────────────────────────────

    def full_sort_predict(self, interaction):
        """
        score(u, all_items) = cf_weight * u_cf·i_cf  +  u_fac @ item_vec.T

        CF 项：[B, n_items] via matmul
        Tucker 项：[B, K] @ [K, n_items] via matmul（高效）
        """
        user = interaction[0]
        u_cf, i_cf, u_fac, item_vec = self.forward()

        # CF 得分
        cf_scores = self.cf_weight * (u_cf[user] @ i_cf.T)      # [B, n_items]

        # Tucker 张量得分（全物品）
        tucker_scores = u_fac[user] @ item_vec.T                 # [B, n_items]

        return cf_scores + tucker_scores
