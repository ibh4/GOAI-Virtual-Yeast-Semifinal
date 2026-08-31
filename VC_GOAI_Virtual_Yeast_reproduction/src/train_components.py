"""训练组件 (自包含提取版)。

从 virtual_yeast/src/train_v5c.py 与 train_v5e_loss.py 原样提取最终
模型训练所需的四个函数与常数, 数学与原实现逐字一致:

  - compute_delta_pca_basis (train_v5c)
  - build_delta_targets    (train_v5c)
  - build_fc_cache         (train_v5e_loss)
  - fc_pcc_loss            (train_v5e_loss)
"""
import numpy as np
import torch

from src.scoring_official_v5b import (
    CONTROL_NAMES, exact_match_control)

EPS_PCC = 1e-8          # 显式 PCC 分母 eps
MIN_STD_SQ = 1e-12      # 方差下限
MIN_PROTEINS = 200      # 与评分器一致的有效蛋白数


def compute_delta_pca_basis(dp, train_indices, bank, rank):
    """在精确匹配 treatment−control 对的 Δ 矩阵上拟合 PCA。

    返回 (components[P, rank], center[P])。Δ 在两者均观测位计算,
    单样本内 NaN 用列中位数填 (仅 PCA 拟合用, 不作为训练标签)。
    """
    from src.model_v5 import compute_pca_basis

    meta = dp.train_meta
    pert = meta['perturbation_no_concentration'].astype(str).str.strip()
    is_ctrl = pert.isin(CONTROL_NAMES).values
    deltas = []
    for ti in train_indices:
        if is_ctrl[ti]:
            continue
        row = meta.iloc[ti]
        ctrl = exact_match_control(row, bank)
        if ctrl is None:
            continue
        m = dp.train_observed_mask[ti] > 0.5
        d = np.where(m & np.isfinite(ctrl),
                     dp.train_matrix[ti] - ctrl, np.nan)
        deltas.append(d)
    D = np.stack(deltas)
    col_median = np.nanmedian(D, axis=0)
    col_median = np.where(np.isfinite(col_median), col_median, 0.0)
    filled = np.where(np.isfinite(D), D, col_median)
    obs = np.isfinite(D).astype(float)
    return compute_pca_basis(filled, obs, rank=rank)


def build_delta_targets(dp, train_indices, bank):
    """每个训练样本的 Δ 目标与 mask (仅精确匹配 treatment)。"""
    meta = dp.train_meta
    pert = meta['perturbation_no_concentration'].astype(str).str.strip()
    is_ctrl = pert.isin(CONTROL_NAMES).values
    n, p = dp.train_matrix.shape
    delta_y = np.full((n, p), np.nan)
    delta_m = np.zeros((n, p))
    for ti in train_indices:
        if is_ctrl[ti]:
            continue
        ctrl = exact_match_control(meta.iloc[ti], bank)
        if ctrl is None:
            continue
        m = (dp.train_observed_mask[ti] > 0.5) & np.isfinite(ctrl)
        delta_y[ti] = np.where(m, dp.train_matrix[ti] - ctrl, 0.0)
        delta_m[ti] = m
    return delta_y, delta_m


def build_fc_cache(dp, train_indices, bank):
    """为训练样本缓存 matched control / true FC / mask。

    返回 (ctrl, true_fc, mask, fc_row_of_train_pos):
      ctrl/true_fc/mask: [N_fc, P] float32 (NaN 无; 无效位 true_fc=0)
      fc_row_of_train_pos: [n_train] → fc 行号 或 -1
    """
    meta = dp.train_meta
    pert = meta['perturbation_no_concentration'].astype(str).str.strip()
    is_ctrl = pert.isin(CONTROL_NAMES).values
    P = dp.n_proteins

    rows_ctrl, rows_fc, rows_mask, rows_pos = [], [], [], []
    for pos, ti in enumerate(train_indices):
        if is_ctrl[ti]:
            continue
        ctrl = exact_match_control(meta.iloc[ti], bank)
        if ctrl is None:
            continue
        m = (dp.train_observed_mask[ti] > 0.5) & np.isfinite(ctrl)
        if m.sum() <= MIN_PROTEINS:
            continue
        rows_ctrl.append(ctrl.astype(np.float32))
        y = dp.train_matrix[ti]
        rows_fc.append(np.where(m, y - ctrl, 0.0).astype(np.float32))
        rows_mask.append(m.astype(np.float32))
        rows_pos.append(pos)

    ctrl = np.stack(rows_ctrl) if rows_ctrl else np.zeros((0, P), np.float32)
    true_fc = np.stack(rows_fc) if rows_fc else np.zeros((0, P), np.float32)
    mask = np.stack(rows_mask) if rows_mask else np.zeros((0, P), np.float32)
    fc_row = np.full(len(train_indices), -1, dtype=np.int64)
    for r, pos in enumerate(rows_pos):
        fc_row[pos] = r
    return ctrl, true_fc, mask, fc_row


def fc_pcc_loss(pred, ctrl_b, true_fc_b, mask_b):
    """逐样本 masked PCC loss。返回 (loss, n_valid, n_low_var)。"""
    pred_fc = pred - ctrl_b
    n = mask_b.sum(dim=1).clamp(min=1)
    mp = (pred_fc * mask_b).sum(1) / n
    mt = (true_fc_b * mask_b).sum(1) / n
    dp_ = (pred_fc - mp.unsqueeze(1)) * mask_b
    dt = (true_fc_b - mt.unsqueeze(1)) * mask_b
    cov = (dp_ * dt).sum(1)
    vp = (dp_ * dp_).sum(1)
    vt = (dt * dt).sum(1)
    corr = cov / (torch.sqrt(vp * vt) + EPS_PCC)
    valid = ((mask_b.sum(1) > MIN_PROTEINS)
             & (vp > MIN_STD_SQ) & (vt > MIN_STD_SQ))
    n_valid = int(valid.sum().item())
    n_low = int((~valid).sum().item())
    if n_valid == 0:
        return None, 0, n_low
    loss = (1.0 - corr[valid]).mean()
    return loss, n_valid, n_low
