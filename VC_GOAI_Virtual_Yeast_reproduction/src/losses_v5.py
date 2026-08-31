"""
Losses v5: Masked losses for observed data only + S2 episodic loss.

Key improvements over v2/v3:
  1. Masked Huber: only compute loss on observed protein positions
  2. S2 episodic residual loss (leave-one-strain-out)
  3. Pearson correlation loss with minimum valid protein threshold
  4. Unknown-strain consistency loss
"""
import torch
import torch.nn.functional as F


def masked_huber(pred, true, observed_mask, delta=1.0):
    """Huber loss computed only on observed positions."""
    diff = pred - true
    abs_diff = diff.abs()
    huber = torch.where(abs_diff < delta,
                        0.5 * diff ** 2,
                        delta * (abs_diff - 0.5 * delta))
    loss = (huber * observed_mask.float()).sum()
    n_valid = observed_mask.float().sum() + 1e-8
    return loss / n_valid


def masked_mse(pred, true, observed_mask):
    """MSE only on observed positions."""
    diff = pred - true
    sq_err = (diff ** 2) * observed_mask.float()
    return sq_err.sum() / (observed_mask.float().sum() + 1e-8)


def stable_correlation_loss(pred, target, mask,
                            min_valid_proteins=500,
                            min_target_std=0.03,
                            min_pred_std=0.01,
                            clip_loss=2.0,
                            eps=1e-6):
    """Stable Pearson correlation loss with variance guards.

    Only computes on samples with enough valid proteins and sufficient variance.
    Per-sample centering, correlation clamp [-0.999, 0.999], loss upper-bound.
    """
    B = pred.size(0)
    losses = []
    valid_flags = []

    for i in range(B):
        m = mask[i] > 0.5
        n_valid = m.sum().item()
        if n_valid < min_valid_proteins:
            valid_flags.append(False)
            continue

        p = pred[i][m]
        t = target[i][m]
        if t.std() < min_target_std or p.std() < min_pred_std:
            valid_flags.append(False)
            continue

        p_c = p - p.mean()
        t_c = t - t.mean()
        cov = (p_c * t_c).sum()
        p_std = torch.sqrt((p_c ** 2).sum() + eps)
        t_std = torch.sqrt((t_c ** 2).sum() + eps)
        corr = cov / (p_std * t_std + eps)
        corr = torch.clamp(corr, -0.999, 0.999)

        if not torch.isnan(corr):
            loss = 1.0 - corr
            loss = torch.clamp(loss, max=clip_loss)
            losses.append(loss)
            valid_flags.append(True)
        else:
            valid_flags.append(False)

    if not losses:
        return torch.tensor(0.0, device=pred.device), 0

    return torch.stack(losses).mean(), len(losses)


def pearson_corr_loss(pred, true, observed_mask, min_valid=500, eps=1e-6):
    """Per-sample Pearson correlation loss.

    1 - PCC(pred[i], true[i]) averaged over samples with enough valid proteins.
    Only positions where BOTH are observed are used.
    """
    B = pred.size(0)
    both_mask = observed_mask  # assume pre-filtered

    losses = []
    valid_samples = 0
    for i in range(B):
        mask = both_mask[i] > 0.5
        n_valid = mask.sum().item()
        if n_valid < min_valid:
            continue
        p = pred[i][mask]
        t = true[i][mask]
        p_c = p - p.mean()
        t_c = t - t.mean()
        cov = (p_c * t_c).sum()
        p_std = torch.sqrt((p_c ** 2).sum() + eps)
        t_std = torch.sqrt((t_c ** 2).sum() + eps)
        corr = cov / (p_std * t_std + eps)
        corr = torch.clamp(corr, -1.0, 1.0)
        if not torch.isnan(corr):
            losses.append(1.0 - corr)
            valid_samples += 1

    if valid_samples == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(losses).mean()


def s2_episodic_loss(model, support_data, query_data, device,
                     compound_to_idx, strain_to_idx,
                     min_valid=500):
    """S2 episodic residual loss for leave-one-strain-out training.

    For a held-out strain (query):
      1. Compute mu_drug_support = mean delta per drug across support strains
      2. strain_residual_true = delta_query_true - mu_drug_support
      3. strain_residual_pred = delta_query_pred - mu_drug_support_pred
      4. Loss = 1 - PCC(strain_residual_pred, strain_residual_true)
    """
    # Simplified: compute drug-wise mean delta on support
    # Get support data predictions
    support_pred, support_comp = support_data['pred'], support_data['compound_idx']
    support_true = support_data['true']

    # Compute mu_drug per compound
    unique_compounds = support_comp.unique()
    mu_drug = {}
    for ci in unique_compounds:
        mask = support_comp == ci
        if mask.sum() > 0:
            mu_drug[ci.item()] = support_true[mask].mean(dim=0)

    # Compute residual for query
    query_pred = query_data['pred']
    query_true = query_data['true']
    query_comp = query_data['compound_idx']

    residuals_pred = []
    residuals_true = []
    for i in range(len(query_pred)):
        ci = query_comp[i].item()
        if ci in mu_drug:
            residuals_pred.append(query_pred[i] - mu_drug[ci])
            residuals_true.append(query_true[i] - mu_drug[ci])

    if len(residuals_pred) == 0:
        return torch.tensor(0.0, device=device)

    r_pred = torch.stack(residuals_pred)
    r_true = torch.stack(residuals_true)

    return pearson_corr_loss(r_pred, r_true,
                             torch.ones_like(r_pred), min_valid=min_valid)


def strain_unknown_consistency_loss(model, strain_idx, compound_idx,
                                     ctx_indices, ctx_cont):
    """Unknown-strain consistency: same drug effect with known vs unknown strain.

    shared_drug_effect should be consistent regardless of strain embedding.
    """
    # Forward with known strain
    pred_known, comp_known = model(
        strain_idx, compound_idx, ctx_indices, ctx_cont,
        force_unknown=False)

    # Forward with unknown strain
    pred_unknown, comp_unknown = model(
        strain_idx, compound_idx, ctx_indices, ctx_cont,
        force_unknown=True)

    # Drug latent should be consistent
    drug_known = comp_known['drug_delta']
    drug_unknown = comp_unknown['drug_delta']

    return F.mse_loss(drug_known, drug_unknown)


def graph_smoothness_loss(delta_pred, edge_index, edge_weight,
                          n_nodes, max_edges=50000, device='cuda'):
    """Graph Laplacian regularization on delta predictions.
    Samples edges to avoid memory explosion.
    """
    if edge_index is None or edge_index.size(1) == 0:
        return torch.tensor(0.0, device=device)

    n_edges = edge_index.size(1)
    if n_edges > max_edges:
        perm = torch.randperm(n_edges, device=device)[:max_edges]
        ei = edge_index[:, perm]
        ew = edge_weight[perm]
    else:
        ei = edge_index
        ew = edge_weight

    B = delta_pred.size(0)
    if B == 0:
        return torch.tensor(0.0, device=device)

    src, dst = ei[0], ei[1]
    diff = delta_pred[:, src] - delta_pred[:, dst]  # [B, E]
    loss = (diff ** 2 * ew.unsqueeze(0)).mean()
    return loss


def dep_direction_loss(delta_pred, delta_true, observed_mask, threshold=1.0):
    """High-effect protein direction consistency loss."""
    both_mask = observed_mask
    high_effect = (delta_true.abs() > threshold) & (both_mask > 0.5)
    n_high = high_effect.sum()

    if n_high < 10:
        return torch.tensor(0.0, device=delta_pred.device)

    sign_match = (delta_pred.sign() == delta_true.sign()).float()
    direction_acc = (sign_match * high_effect.float()).sum() / n_high.float()
    return 1.0 - direction_acc
