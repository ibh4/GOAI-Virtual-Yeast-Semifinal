"""
model_v5b: ControlAnchoredLowRankModelV5 + 外部实体特征扩展 (任务3 B 系列实验)。

扩展点 (均默认关闭, 向后兼容 A0):
  B1 化合物 FP:  fp_proj(32→drug_embed_dim, bias-free) 加到 drug embedding 上。
                 零 FP (DMSO/Water/QC/无 SMILES) → 零贡献, 行为退化为 A0。
  B2 蛋白先验门控: prior_net(7→1) 生成 per-protein 门控,
                 delta_pred *= (1 + tanh(gate)*0.5), 末层零初始化 → 初始为恒等。
"""
import torch
import torch.nn as nn

from src.model_v5 import ControlAnchoredLowRankModelV5


class ControlAnchoredLowRankModelV5B(ControlAnchoredLowRankModelV5):

    def __init__(self, *args, fp_dim=32, prior_dim=7,
                 use_chem_fp=False, use_protein_prior=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_chem_fp = use_chem_fp
        self.use_protein_prior = use_protein_prior

        drug_embed_dim = self.drug_embedding.embedding_dim

        if use_chem_fp:
            # bias-free: FP=0 时不影响原 embedding
            self.fp_proj = nn.Sequential(
                nn.Linear(fp_dim, drug_embed_dim, bias=False),
                nn.LayerNorm(drug_embed_dim),
            )
            # LayerNorm 在零输入时输出非零, 需再加门控: 零 FP → 完全旁路
            self.fp_gate = nn.Linear(fp_dim, 1, bias=True)
            nn.init.zeros_(self.fp_gate.weight)
            nn.init.constant_(self.fp_gate.bias, -2.0)  # sigmoid(-2)≈0.12, 冷启动低权重

        if use_protein_prior:
            self.prior_net = nn.Sequential(
                nn.Linear(prior_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            # 末层零初始化 → 门控初始为 1+0=恒等
            nn.init.zeros_(self.prior_net[-1].weight)
            nn.init.zeros_(self.prior_net[-1].bias)

    def forward(self, strain_idx, compound_idx, ctx_indices, ctx_cont,
                chem_fp=None, protein_prior=None,
                strain_dropout_prob=0.0, force_unknown=False,
                context_dropout_fields=None):
        B = strain_idx.size(0)

        ctx_hidden = self.context_encoder(ctx_indices, ctx_cont,
                                          context_dropout_fields)

        # ── Control prediction (同 A0) ──
        strain_emb = self.get_strain_emb(strain_idx, force_unknown)
        z_control = self.control_head(torch.cat([ctx_hidden, strain_emb], dim=1))
        control_pred = z_control @ self.control_decoder.T + self.control_center

        # ── Drug encoding (B1: + FP) ──
        drug_emb_raw = self.drug_embedding(compound_idx)
        if self.use_chem_fp and chem_fp is not None:
            fp_feat = self.fp_proj(chem_fp)                       # [B, D]
            fp_gate = torch.sigmoid(self.fp_gate(chem_fp))        # [B, 1]
            drug_emb_raw = drug_emb_raw + fp_gate * fp_feat
        drug_hidden = self.drug_encoder(drug_emb_raw)
        drug_delta = self.drug_to_delta(drug_hidden)

        # FiLM strain modulation (protein space)
        drug_in_protein = drug_delta @ self.delta_decoder.T + self.delta_center
        strain_mod, film_strain_emb = self.film(
            strain_idx, drug_in_protein, strain_dropout_prob)

        # Interaction
        d_repr = self.W_d(drug_hidden)
        s_repr = self.W_s(strain_emb)
        interaction = d_repr * s_repr
        interaction_z = self.interaction_proj(interaction)

        combined = torch.cat([drug_delta, interaction_z, ctx_hidden], dim=1)
        z_delta = self.delta_head(combined)
        delta_pred = z_delta @ self.delta_decoder.T + self.delta_center

        # B2: 蛋白先验门控 (per-protein scale, 初始恒等)
        if self.use_protein_prior and protein_prior is not None:
            gate = 1.0 + 0.5 * torch.tanh(self.prior_net(protein_prior))  # [P,1]
            delta_pred = delta_pred * gate.squeeze(-1).unsqueeze(0)

        time_factor = ctx_cont[:, 0:1]
        delta_pred = delta_pred * time_factor * 0.1

        final_pred = control_pred + delta_pred

        components = {
            'control_pred': control_pred,
            'delta_pred': delta_pred,
            'drug_delta': drug_delta,
            'strain_mod': strain_mod,
            'z_control': z_control,
            'z_delta': z_delta,
            'strain_emb': strain_emb,
            'film_strain_emb': film_strain_emb,
            'drug_hidden': drug_hidden,
        }
        return final_pred, components
