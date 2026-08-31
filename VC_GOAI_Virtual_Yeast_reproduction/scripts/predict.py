"""命令 3: 冻结模型推理 → prediction.csv (官方 4,422 蛋白契约)。

仅读取: test metadata / train_val metadata / train_val proteome
(蛋白空间与官方列名) / 3 个 seed checkpoint。
从不读取 test proteome (测试蛋白真值) 或任何测试表型。

输入文件中若存在 test proteome 也不会被加载 (allow_test_labels=False
隔离)。集成规则: 3 seed 等权平均 (预固定)。
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))

from src.data_processor_v5 import DataProcessorV5
from src.model_v5b import ControlAnchoredLowRankModelV5B

SEEDS = [42, 2026, 3407]
DELTA_RANK = 256


def sha256_np(x):
    return hashlib.sha256(np.ascontiguousarray(
        x.astype(np.float32))).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', required=False, default=None,
                    help='官方 test metadata CSV')
    ap.add_argument('--train-metadata', required=False, default=None)
    ap.add_argument('--train-proteome', required=False, default=None)
    ap.add_argument('--input-dir', required=False, default=None,
                    help='备选: 自动探测目录')
    ap.add_argument('--run-dir', required=False, default=None,
                    help='训练输出目录 (含 seed_*/final.pt); '
                         '缺省用包内 checkpoints/')
    ap.add_argument('--contract', required=False, default=None,
                    help='特征契约 JSON; 缺省用包内 '
                         'artifacts/feature_contract.json')
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    files = None
    if args.metadata:
        files = {'meta_test': args.metadata,
                 'meta_train': args.train_metadata,
                 'proteome_train': args.train_proteome,
                 'proteome_test': None}
        assert args.train_metadata and args.train_proteome, (
            "显式 --metadata 模式需同时提供 --train-metadata 与 "
            "--train-proteome (蛋白空间与官方列名来源)")
    dp = DataProcessorV5(input_dir=args.input_dir, files=files,
                        allow_test_labels=False)
    dp.load_all(score_threshold=700)

    contract_path = Path(args.contract) if args.contract else \
        PKG_ROOT / "artifacts" / "feature_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    proteins_contract = contract["proteins"]
    assert len(proteins_contract) == 4422

    run_dir = Path(args.run_dir) if args.run_dir else \
        PKG_ROOT / "checkpoints"
    device = args.device if torch.cuda.is_available() or args.device == \
        'cpu' else 'cpu'

    test_meta = dp.test_meta.reset_index(drop=True)
    test_ids = test_meta['sample_ID'].astype(str).tolist()
    assert len(test_ids) == 4454

    preds = {}
    for seed in SEEDS:
        ckpt_path = run_dir / f"seed_{seed}" / "final.pt"
        assert ckpt_path.exists(), f"missing {ckpt_path}"
        model = ControlAnchoredLowRankModelV5B(
            n_proteins=dp.n_proteins, n_strains=len(dp.strains),
            n_compounds=len(dp.chemical_keys),
            context_vocabs=dp.context_vocabs,
            context_emb_dims=dp.context_emb_dims,
            delta_rank=DELTA_RANK, use_chem_fp=False,
            use_protein_prior=False).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                         weights_only=True))
        model.eval()
        preds_seed = []
        with torch.no_grad():
            for i in range(0, len(test_meta), 64):
                mb = test_meta.iloc[i:i + 64]
                si, ci, cxi, cxc = dp.encode_sample_batch(mb, device)
                p, _ = model(si, ci, cxi, cxc)
                preds_seed.append(p.cpu().numpy())
        pred = np.concatenate(preds_seed, axis=0).astype(np.float32)
        assert pred.shape == (4454, dp.n_proteins)
        assert np.all(np.isfinite(pred))
        preds[seed] = pred
        print(f"  seed {seed}: shape={pred.shape} "
              f"SHA={sha256_np(pred)[:16]}")

    ens = ((preds[42].astype(np.float64) + preds[2026].astype(np.float64)
            + preds[3407].astype(np.float64)) / 3.0).astype(np.float32)
    assert np.all(np.isfinite(ens))

    official_cols = [dp.protein_name_to_official[p] for p in dp.proteins]
    col_idx = {c: i for i, c in enumerate(official_cols)}
    sub = ens[:, [col_idx[c] for c in proteins_contract]]
    df = pd.DataFrame(sub, columns=proteins_contract)
    df.insert(0, 'sample_ID', test_meta['sample_ID'].values)
    assert df['sample_ID'].is_unique
    assert list(df['sample_ID'].astype(str)) == test_ids
    assert not df.isna().any().any()
    assert np.all(np.isfinite(df[proteins_contract].to_numpy()))

    out = Path(args.output)
    df.to_csv(out, index=False)
    csv_sha = sha256_file(out)
    print(f"prediction.csv: {df.shape}, prediction_scale=log2, "
          f"SHA256={csv_sha}")

    manifest = {
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'prediction_scale': 'log2',
        'model': 'L2b (C2-r256 + lambda_fc=0.07), 3-seed equal ensemble',
        'seeds': SEEDS, 'ensemble_rule': 'equal weight mean (pre-fixed)',
        'rows': 4454, 'columns': 1 + len(proteins_contract),
        'per_seed_pred_sha256': {str(s): sha256_np(preds[s])
                                 for s in SEEDS},
        'ensemble_pred_sha256_full_space': sha256_np(ens),
        'csv_sha256': csv_sha,
        'test_sample_id_sha256': hashlib.sha256(
            '|'.join(test_ids).encode()).hexdigest(),
        'used_test_labels': False,
        'contract_proteins_sha256': contract['proteins_sha256'],
    }
    with open(Path(args.output).with_name(
            Path(args.output).stem + "_manifest.json"), 'w',
            encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("PREDICT_DONE — 未读取任何测试表型")


if __name__ == '__main__':
    main()
