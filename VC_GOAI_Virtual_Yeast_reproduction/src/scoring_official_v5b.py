"""
scoring_official_v5b → 任务4-B 审计版: local_six_module_proxy。

相对任务3版的修改:
  1. 主入口改名 local_six_module_proxy() — 与组委会实现完全对齐前,
     所有分数统一以 local proxy 命名, 不代表官方成绩。
  2. Delta 目标仅使用 7 字段精确匹配对照 (Strains/Medium/Temperature/
     pert_time/instrument/Yeast_cell_plate/data_source), 删除 lenient fallback。
  3. 新增覆盖率统计: n_treatment / n_exact_matched / coverage / unmatched。
  4. absolute: 逐样本 corr + 逐蛋白 corr + masked R² (聚合方式写入报告)。
  5. FC: 逐样本 PCC(Δpred, Δtrue) 均值, Δ 在 (treat,ctrl) 均观测位计算。
  6. S1 残差: μ_ctx = 同 (strain,medium,temp,time) 训练药物 Δ 均值,
     PCC(Δpred−μ_ctx, Δtrue−μ_ctx);
     S2 残差: μ_drug = 同 chem_key 训练 Δ 均值。
     均仅用 fold 训练样本计算, 未见实体自动无参照 → 该样本不计入该模块。
  7. S3/time: 0.7·FC + 0.3·absolute (s3 样本并入 FC 集合)。
  8. DEP: |Δ_true|>1 方向准确率 0.4 + 高效应 PCC 0.4 + F1@100 0.2。

所有 NaN 处理: PCC 遇零方差返回 0.0; 样本内有效蛋白 <200 跳过。
"""
from collections import defaultdict

import numpy as np

CONTROL_NAMES = {'Water', 'DMSO'}
MATCH_FIELDS = ['Strains', 'Medium', 'Temperature', 'pert_time',
                'instrument', 'Yeast_cell_plate', 'data_source']


def _pcc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def build_exact_control_bank(train_meta, train_matrix, train_obs,
                             train_indices):
    """7 字段精确匹配 → 对照组均值向量 (仅 fold 训练样本)."""
    pert = train_meta['perturbation_no_concentration'].astype(str).str.strip()
    is_ctrl = pert.isin(CONTROL_NAMES).values
    bank_idx = defaultdict(list)
    for ci in train_indices:
        if not is_ctrl[ci]:
            continue
        row = train_meta.iloc[ci]
        key = tuple(str(row.get(f, '')) for f in MATCH_FIELDS)
        bank_idx[key].append(ci)
    bank = {}
    # 对照组内全缺失蛋白 → 回填 fold 训练集该蛋白中位数 (仅影响
    # 绝对预测场景; Δ 场景由 isfinite(ctrl) mask 保护, 不受回填影响)
    with np.errstate(all='ignore'):
        train_col_median = np.nanmedian(train_matrix[train_indices], axis=0)
    for key, idxs in bank_idx.items():
        m = train_obs[idxs] > 0.5
        with np.errstate(all='ignore'):
            mu = np.nanmean(np.where(m, train_matrix[idxs], np.nan), axis=0)
        bad = ~np.isfinite(mu)
        if bad.any():
            mu = mu.copy()
            mu[bad] = np.take(train_col_median, np.where(bad)[0])
        bank[key] = mu
    return bank


def exact_match_control(row, bank):
    key = tuple(str(row.get(f, '')) for f in MATCH_FIELDS)
    return bank.get(key)  # None = 未精确匹配, 不 fallback


def score_absolute(pred, true, obs):
    """0.5·sample_corr + 0.25·protein_corr + 0.25·max(masked R², 0)。"""
    scs, pcs = [], []
    for i in range(len(pred)):
        m = obs[i] > 0.5
        if m.sum() < 100:
            continue
        scs.append(_pcc(pred[i][m], true[i][m]))
    for j in range(pred.shape[1]):
        m = obs[:, j] > 0.5
        if m.sum() < 4:
            continue
        pcs.append(_pcc(pred[m, j], true[m, j]))
    p_flat, t_flat = pred[obs > 0.5], true[obs > 0.5]
    r2 = float(1 - ((t_flat - p_flat) ** 2).sum()
               / (((t_flat - t_flat.mean()) ** 2).sum() + 1e-8))
    sample_corr = float(np.mean(scs)) if scs else 0.0
    protein_corr = float(np.mean(pcs)) if pcs else 0.0
    score = 0.5 * sample_corr + 0.25 * protein_corr + 0.25 * max(r2, 0)
    return {'sample_corr': sample_corr, 'protein_corr': protein_corr,
            'r2': r2, 'score': score}


def _train_delta_references(train_meta, train_matrix, train_obs,
                            train_indices, bank):
    """μ_ctx 与 μ_drug: 仅精确匹配的训练 treatment 的 Δ_true 均值。"""
    pert = train_meta['perturbation_no_concentration'].astype(str).str.strip()
    is_ctrl = pert.isin(CONTROL_NAMES).values
    ctx_deltas, drug_deltas = defaultdict(list), defaultdict(list)
    for ti in train_indices:
        if is_ctrl[ti]:
            continue
        row = train_meta.iloc[ti]
        ctrl = exact_match_control(row, bank)
        if ctrl is None:
            continue
        m = train_obs[ti] > 0.5
        d = np.where(m & np.isfinite(ctrl), train_matrix[ti] - ctrl, np.nan)
        ctx_key = (str(row['Strains']), str(row['Medium']),
                   str(row['Temperature']), str(row['pert_time']))
        ctx_deltas[ctx_key].append(d)
        drug_deltas[str(row['_chem_key'])].append(d)
    ctx_mean = {k: np.nanmean(np.stack(v), axis=0)
                for k, v in ctx_deltas.items()}
    drug_mean = {k: np.nanmean(np.stack(v), axis=0)
                 for k, v in drug_deltas.items()}
    return ctx_mean, drug_mean


def official_six_module(val_pred, val_true, val_obs, val_meta,
                        train_meta, train_matrix, train_obs,
                        sample_splits=None):
    """兼容别名 (train_v5b 引用) — 注意旧调用缺 train_indices,
    此处以全量 meta 行号作 bank 来源 (仅 B 系列历史重评用途)。
    新代码一律调用 local_six_module。"""
    return local_six_module(
        val_pred, val_true, val_obs, val_meta,
        train_meta, train_matrix, train_obs,
        np.arange(len(train_meta)), sample_splits=sample_splits)


def local_six_module(val_pred, val_true, val_obs, val_meta,
                     train_meta, train_matrix, train_obs, train_indices,
                     sample_splits=None):
    """LOSO fold 上的本地六模块代理分 (仅精确匹配, 无 fallback)。"""
    bank = build_exact_control_bank(
        train_meta, train_matrix, train_obs, train_indices)
    ctx_mean, drug_mean = _train_delta_references(
        train_meta, train_matrix, train_obs, train_indices, bank)

    pert_val = val_meta['perturbation_no_concentration'].astype(str).str.strip()
    is_ctrl_val = pert_val.isin(CONTROL_NAMES).values

    dps_all, dts_all = [], []
    dps_s1, dts_s1, mus_ctx = [], [], []
    dps_s2, dts_s2, mus_drug = [], [], []
    n_treat = n_matched = 0
    unmatched_split = defaultdict(int)

    for i in range(len(val_meta)):
        if is_ctrl_val[i]:
            continue
        n_treat += 1
        row = val_meta.iloc[i]
        ctrl = exact_match_control(row, bank)
        if ctrl is None:
            sp = sample_splits[i] if sample_splits else '?'
            unmatched_split[sp] += 1
            continue
        n_matched += 1
        m = (val_obs[i] > 0.5) & np.isfinite(ctrl)
        dp_i = np.where(m, val_pred[i] - ctrl, np.nan)
        dt_i = np.where(m, val_true[i] - ctrl, np.nan)
        dps_all.append(dp_i); dts_all.append(dt_i)

        ctx_key = (str(row['Strains']), str(row['Medium']),
                   str(row['Temperature']), str(row['pert_time']))
        ck = str(row['_chem_key'])
        sp = sample_splits[i] if sample_splits else (
            's1' if ck not in drug_mean else 's2')
        if sp == 's1' and ctx_key in ctx_mean:
            dps_s1.append(dp_i); dts_s1.append(dt_i)
            mus_ctx.append(ctx_mean[ctx_key])
        elif sp == 's2' and ck in drug_mean:
            dps_s2.append(dp_i); dts_s2.append(dt_i)
            mus_drug.append(drug_mean[ck])

    abs_m = score_absolute(val_pred, val_true, val_obs)

    fc_pccs = [_pcc(a[np.isfinite(b)], b[np.isfinite(b)])
               for a, b in zip(dps_all, dts_all)
               if np.isfinite(b).sum() > 200]
    fc = float(np.mean(fc_pccs)) if fc_pccs else 0.0

    s1_pccs = [_pcc((a - mu)[np.isfinite(b - mu)], (b - mu)[np.isfinite(b - mu)])
               for a, b, mu in zip(dps_s1, dts_s1, mus_ctx)
               if np.isfinite(b - mu).sum() > 200]
    s1_res = float(np.mean(s1_pccs)) if s1_pccs else 0.0

    s2_pccs = [_pcc((a - mu)[np.isfinite(b - mu)], (b - mu)[np.isfinite(b - mu)])
               for a, b, mu in zip(dps_s2, dts_s2, mus_drug)
               if np.isfinite(b - mu).sum() > 200]
    s2_res = float(np.mean(s2_pccs)) if s2_pccs else 0.0

    # DEP: 合并所有精确匹配样本的 Δ
    dep = {'dir_acc': 0.0, 'high_pcc': 0.0, 'f1_at_k': 0.0, 'score': 0.0}
    if dts_all:
        all_dt = np.concatenate([d[np.isfinite(d)] for d in dts_all])
        all_dp = np.concatenate([d[np.isfinite(d)] for d in dps_all])
        high = np.abs(all_dt) > 1.0
        if high.any():
            dep['dir_acc'] = float(
                (np.sign(all_dp[high]) == np.sign(all_dt[high])).mean())
        if high.sum() > 10:
            dep['high_pcc'] = _pcc(all_dp[high], all_dt[high])
        k = min(100, len(all_dt))
        top = np.argsort(-np.abs(all_dp))[:k]
        prec = float(high[top].mean())
        n_high = int(high.sum())
        rec = float(high[top].sum() / n_high) if n_high else 0.0
        dep['f1_at_k'] = float(2 * prec * rec / (prec + rec + 1e-8))
        dep['score'] = (0.4 * dep['dir_acc'] + 0.4 * max(dep['high_pcc'], 0)
                        + 0.2 * dep['f1_at_k'])

    s3_time = 0.7 * fc + 0.3 * abs_m['score']
    weighted = (0.20 * abs_m['score'] + 0.25 * fc + 0.20 * s1_res
                + 0.20 * s2_res + 0.10 * s3_time + 0.05 * dep['score'])

    return {
        'scorer': 'local_six_module_proxy v2 (exact-match only)',
        'n_val': len(val_meta), 'n_treatment': n_treat,
        'n_exact_matched': n_matched,
        'exact_match_coverage': round(n_matched / max(1, n_treat), 4),
        'unmatched_by_split': dict(unmatched_split),
        'n_fc_samples': len(fc_pccs), 'n_s1_res': len(s1_pccs),
        'n_s2_res': len(s2_pccs),
        'absolute_score': abs_m['score'],
        'absolute_sample_corr': abs_m['sample_corr'],
        'absolute_protein_corr': abs_m['protein_corr'],
        'absolute_r2': abs_m['r2'],
        'matched_fc_pcc': fc,
        's1_residual_pcc': s1_res,
        's2_residual_pcc': s2_res,
        's3_time_score': s3_time,
        'dep_dir_acc': dep['dir_acc'], 'dep_high_pcc': dep['high_pcc'],
        'dep_f1_at_k': dep['f1_at_k'], 'dep_score': dep['score'],
        'local_six_module_proxy_score': float(weighted),
        # 兼容字段 (旧名指向同一数值, 标注 proxy)
        'official_weighted_score': float(weighted),
    }
