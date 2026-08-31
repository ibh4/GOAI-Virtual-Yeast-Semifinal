"""命令 2: 从官方训练数据从头训练最终模型 (L2b 3-seed full refit)。

数学与提交版本逐字一致 (原 scripts/train_l2b_full.py 移植):
  - 模型: ControlAnchoredLowRankModelV5B (C2-r256, 无外部特征)
  - loss: masked_huber + 0.5·pearson + 0.5·delta_mse
           + lambda_fc(0.07)·warmup(5)·逐样本 masked FC-PCC
  - optimizer: AdamW (lr 3e-4, decoder/center lr×0.2, wd 1e-4, clip 1.0)
  - checkpoint: best-by-train-loss; 40 epochs; batch 24
  - seeds: 42 / 2026 / 3407 (等权集成, 权重预固定)
  - 全部统计量 (control bank / 列中位数 / PCA 基底) 仅由
    split_final=='train'+'val' 的 8,958 行 (官方 train_val 文件全部行,
    不含任何测试样本) 拟合; 测试蛋白真值从不读取。

用法 (语义等价于 README 三主命令之二):
  python scripts/train.py --metadata <train_val_metadata.csv> \
      --proteome <train_val_proteome.csv> \
      --test-metadata <test_metadata.csv> \
      --config configs/final.yaml --output-dir runs/final

说明: --test-metadata 仅提供 test 样本的菌株/化合物/上下文词表
(实体编码需要 train+test 联合词表, 与训练时一致), 不含也不读取
任何测试表型。
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))

from src.data_processor_v5 import DataProcessorV5
from src.model_v5b import ControlAnchoredLowRankModelV5B
from src.model_v5 import compute_pca_basis
from src.losses_v5 import masked_huber, pearson_corr_loss
from src.scoring_official_v5b import (
    build_exact_control_bank, CONTROL_NAMES)
from src.train_components import (
    compute_delta_pca_basis, build_delta_targets, build_fc_cache,
    fc_pcc_loss, EPS_PCC, MIN_STD_SQ, MIN_PROTEINS)

LAMBDA_FC = 0.07
WARMUP = 5
FULL_EPOCHS = 40
BATCH = 24
LR = 3e-4
DELTA_RANK = 256
SEEDS = [42, 2026, 3407]


def log_line(path, msg):
    ts = time.strftime('%H:%M:%S')
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg, flush=True)


def train_seed(seed, dp, full_idx, audit, out_root, device):
    seed_dir = out_root / "seed_" + str(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_path = seed_dir / "train.log"

    config = {
        'model': 'ControlAnchoredLowRankModelV5B (C2-r256 + L2b FC loss)',
        'seed': seed, 'full_epochs': FULL_EPOCHS,
        'delta_rank': DELTA_RANK, 'lambda_fc': LAMBDA_FC,
        'fc_warmup_epochs': WARMUP,
        'loss': 'masked_huber + 0.5*pearson_fc + '
                '0.5*delta_mse(matched) + lambda_fc*warmup*L_fc',
        'eps_pcc': EPS_PCC, 'min_std_sq': MIN_STD_SQ,
        'min_proteins': MIN_PROTEINS,
        'optimizer': 'AdamW', 'lr': LR, 'decoder_lr_ratio': 0.2,
        'weight_decay': 1e-4, 'grad_clip': 1.0, 'batch_size': BATCH,
        'checkpoint_rule': 'best_by_train_loss',
        'n_fit_samples': audit['n_fit'],
        'data_sample_id_sha256': audit['sample_id_sha256'],
        'split_schema': 'full_8958_no_loso', 'used_test_labels': False,
    }
    with open(seed_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    with open(seed_dir / "data_audit.json", 'w') as f:
        json.dump(audit, f, indent=2)

    t0 = time.time()
    log_line(log_path, f"== START L2b-full seed={seed} "
                       f"epochs={FULL_EPOCHS} lambda_fc={LAMBDA_FC}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    meta = dp.train_meta
    pert = meta['perturbation_no_concentration'].astype(str).str.strip()
    is_ctrl = pert.isin(CONTROL_NAMES).values

    bank = build_exact_control_bank(
        meta, dp.train_matrix_filled, dp.train_observed_mask, full_idx)
    ctrl_rows = [i for i in full_idx if is_ctrl[i]]
    cc, cc_c = compute_pca_basis(
        dp.train_matrix_filled[ctrl_rows],
        dp.train_observed_mask[ctrl_rows], rank=192)
    log_line(log_path, f"  control-PCA on {len(ctrl_rows)} control samples")
    dc, dc_c = compute_delta_pca_basis(dp, full_idx, bank,
                                       rank=DELTA_RANK)
    delta_y_np, delta_m_np = build_delta_targets(dp, full_idx, bank)
    n_match = int((delta_m_np.sum(axis=1) > 0).sum())
    log_line(log_path, f"  delta targets: {n_match}/{len(full_idx)} matched")

    fc_ctrl_np, fc_true_np, fc_mask_np, fc_row = build_fc_cache(
        dp, full_idx, bank)
    log_line(log_path, f"  FC cache: {len(fc_ctrl_np)} samples "
                       f"(> {MIN_PROTEINS} valid proteins)")
    fc_ctrl = torch.tensor(fc_ctrl_np)
    fc_true = torch.tensor(fc_true_np)
    fc_mask = torch.tensor(fc_mask_np)
    has_fc = torch.tensor(fc_row >= 0)

    model = ControlAnchoredLowRankModelV5B(
        n_proteins=dp.n_proteins, n_strains=len(dp.strains),
        n_compounds=len(dp.chemical_keys),
        context_vocabs=dp.context_vocabs,
        context_emb_dims=dp.context_emb_dims,
        delta_rank=DELTA_RANK, use_chem_fp=False, use_protein_prior=False,
    ).to(device)
    model.init_decoders_from_pca(cc, cc_c, dc, dc_c)

    train_meta = meta.iloc[full_idx].reset_index(drop=True)
    train_y = torch.tensor(dp.train_matrix_filled[full_idx],
                           dtype=torch.float32)
    train_obs = torch.tensor(dp.train_observed_mask[full_idx],
                             dtype=torch.float32)
    d_y = torch.tensor(np.nan_to_num(delta_y_np[full_idx]),
                       dtype=torch.float32)
    d_m = torch.tensor(delta_m_np[full_idx], dtype=torch.float32)

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters()
                    if 'decoder' not in n and 'center' not in n], 'lr': LR},
        {'params': [p for n, p in model.named_parameters()
                    if 'decoder' in n or 'center' in n], 'lr': LR * 0.2},
    ], weight_decay=1e-4)

    n_samples = len(full_idx)
    best_loss, best_state, best_ep = float('inf'), None, -1
    history, n_naninf = [], 0

    for epoch in range(1, FULL_EPOCHS + 1):
        model.train()
        torch.manual_seed(seed + epoch)
        perm = torch.randperm(n_samples)
        eff_lambda = LAMBDA_FC * min(1.0, epoch / WARMUP)
        ep = {'loss': 0.0, 'c2': 0.0, 'fc': 0.0, 'n_fc': 0, 'nb': 0}
        for i in range(0, n_samples, BATCH):
            idx = perm[i:i + BATCH]
            mb = train_meta.iloc[idx]
            si, ci, cxi, cxc = dp.encode_sample_batch(mb, device)
            y_b = train_y[idx].to(device)
            obs_b = train_obs[idx].to(device)

            pred, comp = model(si, ci, cxi, cxc)
            loss_c2 = masked_huber(pred, y_b, obs_b)
            loss_c2 = loss_c2 + 0.5 * pearson_corr_loss(
                pred, y_b, obs_b, min_valid=200)
            dpred = comp['delta_pred']
            dy_b = d_y[idx].to(device)
            dm_b = d_m[idx].to(device)
            if dm_b.sum() > 0:
                se = (dpred - dy_b) ** 2 * dm_b
                loss_c2 = loss_c2 + 0.5 * (
                    se.sum() / dm_b.sum().clamp(min=1))

            loss = loss_c2
            hf = has_fc[idx]
            if hf.any():
                rows = fc_row[idx][hf]
                loss_fc, n_v, _ = fc_pcc_loss(
                    pred[hf], fc_ctrl[rows].to(device),
                    fc_true[rows].to(device),
                    fc_mask[rows].to(device))
                if loss_fc is not None:
                    loss = loss + eff_lambda * loss_fc
                    ep['fc'] += float(loss_fc.item())
                    ep['n_fc'] += n_v

            if not torch.isfinite(loss):
                n_naninf += 1
                log_line(log_path, f"  NaN/Inf @ep{epoch} — 终止")
                raise RuntimeError(f"non-finite loss @ep{epoch}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep['loss'] += loss.item()
            ep['c2'] += loss_c2.item()
            ep['nb'] += 1

        nb = max(1, ep['nb'])
        avg = ep['loss'] / nb
        history.append({'epoch': epoch, 'loss_total': round(avg, 6),
                        'loss_c2': round(ep['c2'] / nb, 6),
                        'loss_fc': round(ep['fc'] / nb, 6),
                        'eff_lambda_fc': eff_lambda,
                        'n_fc_valid': ep['n_fc']})
        if avg < best_loss:
            best_loss, best_ep = avg, epoch
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        log_line(log_path, f"  ep{epoch}/{FULL_EPOCHS} loss={avg:.4f} "
                           f"(c2={ep['c2']/nb:.4f} fc={ep['fc']/nb:.4f}) "
                           f"best={best_loss:.4f}@{best_ep}")
        if device == 'cuda':
            torch.cuda.empty_cache()

    torch.save(best_state, seed_dir / "final.pt")
    with open(seed_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    with open(seed_dir / "final_state.json", 'w') as f:
        json.dump({'seed': seed, 'full_epochs': FULL_EPOCHS,
                   'best_loss': best_loss, 'best_epoch': best_ep,
                   'checkpoint_rule': 'best_by_train_loss',
                   'n_naninf': n_naninf,
                   'elapsed_s': round(time.time() - t0, 1),
                   'final_pt_sha256': hashlib.sha256(
                       (seed_dir / "final.pt").read_bytes()).hexdigest()},
                  f, indent=2)
    log_line(log_path, f"== DONE seed={seed} "
                       f"({time.time()-t0:.0f}s, best@ep{best_ep})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', required=False, default=None,
                    help='官方 train_val metadata CSV')
    ap.add_argument('--proteome', required=False, default=None,
                    help='官方 train_val proteome raw CSV')
    ap.add_argument('--test-metadata', required=False, default=None,
                    help='官方 test metadata CSV (仅实体词表, 无表型)')
    ap.add_argument('--input-dir', required=False, default=None,
                    help='备选: 自动探测目录 (含上述官方文件)')
    ap.add_argument('--config', required=False, default=None,
                    help='configs/final.yaml (配置冻结声明, '
                         '训练参数以本脚本常量为准)')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    files = None
    if args.metadata and args.proteome:
        files = {'meta_train': args.metadata,
                 'proteome_train': args.proteome,
                 'meta_test': args.test_metadata,
                 'proteome_test': None}
        assert args.test_metadata, ("训练需要 test metadata 构建联合"
                                    "实体词表 (与提交版本一致); 不读取"
                                    "任何测试表型")
    dp = DataProcessorV5(input_dir=args.input_dir, files=files,
                        allow_test_labels=False)
    dp.load_all(score_threshold=700)
    full_idx = np.arange(len(dp.train_meta)).tolist()
    assert len(dp.train_meta) == 8958, "train_val 行数 != 8958"

    audit = {
        'n_total': int(len(dp.train_meta)), 'n_fit': int(len(full_idx)),
        'split_final_counts':
            dp.train_meta['split_final'].value_counts().to_dict(),
        'strain_counts':
            dp.train_meta['Strains'].value_counts().to_dict(),
        'n_compounds': int(dp.train_meta['_chem_key'].nunique()),
        'n_proteins': int(dp.n_proteins),
        'sample_id_sha256': hashlib.sha256('|'.join(
            dp.train_meta['sample_ID'].astype(str).tolist())
            .encode()).hexdigest(),
        'used_test_labels': False,
    }
    print(f"审计: n=8958, split={audit['split_final_counts']}, "
          f"proteins={audit['n_proteins']}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() or args.device == \
        'cpu' else 'cpu'
    for seed in SEEDS:
        train_seed(seed, dp, full_idx, audit, out_root, device)
    print("TRAIN_DONE — 3 seed checkpoint 位于", out_root)


if __name__ == '__main__':
    main()
