"""冒烟测试: 数据加载契约 + 模型构建 + forward 形状 (不训练)。"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

from src.data_processor_v5 import DataProcessorV5
from src.model_v5b import ControlAnchoredLowRankModelV5B
from src.train_components import (
    compute_delta_pca_basis, build_delta_targets, build_fc_cache)
from src.scoring_official_v5b import (
    build_exact_control_bank, CONTROL_NAMES)


def test_contract():
    c = json.loads((PKG / "artifacts" / "feature_contract.json")
                   .read_text(encoding="utf-8"))
    assert c["n_proteins"] == 4422
    assert len(set(c["proteins"])) == 4422
    print("contract OK: 4,422 proteins")


def test_model_forward():
    import os
    input_dir = os.environ.get(
        "GOAI_INPUT_DIR",
        str(PKG.parent.parent.parent / "input"))
    dp = DataProcessorV5(input_dir=input_dir, allow_test_labels=False)
    dp.load_all(score_threshold=700)
    assert dp.n_proteins == 5243
    assert len(dp.train_meta) == 8958
    assert len(dp.test_meta) == 4454
    assert dp.test_matrix is None            # 测试真值隔离
    model = ControlAnchoredLowRankModelV5B(
        n_proteins=dp.n_proteins, n_strains=len(dp.strains),
        n_compounds=len(dp.chemical_keys),
        context_vocabs=dp.context_vocabs,
        context_emb_dims=dp.context_emb_dims,
        delta_rank=256, use_chem_fp=False, use_protein_prior=False)
    mb = dp.train_meta.iloc[:4]
    si, ci, cxi, cxc = dp.encode_sample_batch(mb, "cpu")
    with torch.no_grad():
        pred, comp = model(si, ci, cxi, cxc)
    assert pred.shape == (4, dp.n_proteins)
    assert comp["z_delta"].shape == (4, 256)
    assert np.all(np.isfinite(pred.numpy()))
    # 训练组件冒烟
    idx = list(range(200))
    bank = build_exact_control_bank(
        dp.train_meta, dp.train_matrix_filled, dp.train_observed_mask, idx)
    dc, dc_c = compute_delta_pca_basis(dp, idx, bank, rank=256)
    # 200 样本子集 → 有效 Δ 行数有限, SVD rank 相应收缩
    assert dc.shape[0] == dp.n_proteins and dc.shape[1] >= 100
    dy, dm = build_delta_targets(dp, idx, bank)
    fc = build_fc_cache(dp, idx, bank)
    assert fc[0].shape[1] == dp.n_proteins
    print("model/components forward OK")


if __name__ == "__main__":
    test_contract()
    test_model_forward()
    print("SMOKE_TESTS_PASSED")
