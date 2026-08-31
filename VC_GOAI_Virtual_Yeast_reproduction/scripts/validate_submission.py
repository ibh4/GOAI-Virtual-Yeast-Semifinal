"""命令 4: prediction.csv 提交前格式校验。

检查: 4,454 行 × (sample_ID + 4,422 官方蛋白列); sample_ID 唯一且
与官方 test metadata 顺序一致; 无 NA/Inf/重复列; 数值有限;
prediction_scale=log2 (manifest 声明); SHA256 输出。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prediction', required=True)
    ap.add_argument('--test-metadata', required=False, default=None)
    ap.add_argument('--input-dir', required=False, default=None)
    args = ap.parse_args()

    contract = json.loads((PKG_ROOT / "artifacts" / "feature_contract.json")
                          .read_text(encoding="utf-8"))
    proteins = contract["proteins"]

    # 官方 test metadata (sample_ID 顺序)
    from src.data_processor_v5 import DataProcessorV5
    dp = DataProcessorV5(input_dir=args.input_dir,
                        files={'meta_test': args.test_metadata,
                               'meta_train': None, 'proteome_train': None,
                               'proteome_test': None}
                        if args.test_metadata else None,
                        allow_test_labels=False)
    # 仅需要 test metadata 的 sample_ID 顺序 — 直接读 CSV 避免全量加载
    if args.test_metadata:
        tm = pd.read_csv(args.test_metadata)
        test_ids = tm['sample_ID'].astype(str).tolist()
    else:
        files = dp._files
        raise SystemExit("请提供 --test-metadata (官方 test metadata)")

    checks = {}

    df = pd.read_csv(args.prediction)
    checks['rows'] = len(df)
    checks['rows_ok'] = len(df) == 4454
    checks['columns'] = df.shape[1]
    checks['columns_ok'] = df.shape[1] == 4423

    cols = list(df.columns)
    checks['protein_order_ok'] = cols[1:] == proteins
    checks['sample_id_unique'] = bool(df['sample_ID'].is_unique)
    ids = df['sample_ID'].astype(str).tolist()
    checks['sample_id_order_ok'] = ids == test_ids

    vals = df[proteins].to_numpy()
    checks['all_finite'] = bool(np.all(np.isfinite(vals)))
    checks['no_na'] = bool(not df.isna().any().any())
    checks['no_dup_columns'] = len(set(cols)) == len(cols)
    checks['no_extra_columns'] = set(cols) == set(['sample_ID'] + proteins)
    checks['range_log2_plausible'] = bool(vals.min() > -10
                                          and vals.max() < 60)

    pman_path = PKG_ROOT / "artifacts" / "prediction_manifest.json"
    pman = json.loads(pman_path.read_text(encoding="utf-8"))
    checks['prediction_scale_declared_log2'] = \
        pman.get('prediction_scale') == 'log2'

    csv_sha = sha256_file(Path(args.prediction))
    checks['csv_sha256'] = csv_sha
    checks['sha_matches_manifest'] = (
        csv_sha == pman.get('csv_sha256'))

    print(json.dumps(checks, indent=2, ensure_ascii=False))
    must_pass = ['rows_ok', 'columns_ok', 'protein_order_ok',
                 'sample_id_unique', 'sample_id_order_ok', 'all_finite',
                 'no_na', 'no_dup_columns', 'no_extra_columns',
                 'prediction_scale_declared_log2']
    failed = [k for k in must_pass if not checks.get(k)]
    if failed:
        print(f"VALIDATION_FAILED: {failed}")
        return 1
    print("VALIDATION_PASSED")
    return 0


if __name__ == '__main__':
    sys.exit(main())
