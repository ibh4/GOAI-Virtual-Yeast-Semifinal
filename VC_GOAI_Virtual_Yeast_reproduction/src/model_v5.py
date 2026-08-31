"""
model_v5: ControlAnchoredLowRankModel for S2 strain generalization.

Architecture:
  pred = control_pred + delta_pred
  control_pred = z_control @ decoder_control + control_center
  delta_pred = z_delta @ decoder_delta + delta_center

  z_delta = shared_drug_effect + strain_modulation(FiLM) + drug_strain_interaction

Key features:
  1. Low-rank PCA decoding (no per-sample full GAT)
  2. FiLM strain modulation with unknown-strain prototype
  3. Strain dropout + unknown-strain consistency
  4. Chemical encoder with UNK support
  5. Context embeddings with dropout
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FiLMStrainModulation(nn.Module):
    """Low-rank FiLM for strain-conditioned modulation.

    strain_scale = strain_factors @ W_scale.T   [B, N]
    strain_shift = strain_factors @ W_shift.T   [B, N]
    output = scale * input + shift
    """

    def __init__(self, n_proteins, strain_embed_dim, film_rank, n_strains,
                 dropout=0.15):
        super().__init__()
        self.film_rank = film_rank
        self.n_strains = n_strains

        self.strain_embedding = nn.Embedding(n_strains, strain_embed_dim)
        self.strain_proj = nn.Sequential(
            nn.Linear(strain_embed_dim, strain_embed_dim),
            nn.GELU(),
            nn.BatchNorm1d(strain_embed_dim),
            nn.Dropout(dropout),
            nn.Linear(strain_embed_dim, film_rank * 2),
        )

        # Shared protein-level patterns
        self.film_scale_W = nn.Parameter(torch.randn(n_proteins, film_rank) * 0.02)
        self.film_shift_W = nn.Parameter(torch.randn(n_proteins, film_rank) * 0.02)

        # Strain baseline shift
        self.strain_baseline = nn.Parameter(torch.zeros(n_strains, film_rank))

    def forward(self, strain_idx, drug_latent, strain_dropout_prob=0.0):
        """Apply FiLM modulation to drug latent.

        Args:
            strain_idx: [B] strain indices
            drug_latent: [B, N] drug effect in protein space
            strain_dropout_prob: probability of zeroing strain modulation

        Returns:
            modulated: [B, N] strain-modulated drug effect
            strain_emb: [B, strain_embed_dim] for consistency loss
        """
        B = strain_idx.size(0)
        strain_emb = self.strain_embedding(strain_idx)
        factors = self.strain_proj(strain_emb)
        scale_coeff = factors[:, :self.film_rank]         # [B, rank]
        shift_coeff = factors[:, self.film_rank:]         # [B, rank]

        # Low-rank: [B, rank] @ [rank, N]^T → [B, N]
        strain_scale = scale_coeff @ self.film_scale_W.T
        strain_shift = shift_coeff @ self.film_shift_W.T

        # Strain dropout
        if self.training and strain_dropout_prob > 0:
            dropout_mask = (torch.rand(B, 1, device=strain_idx.device)
                            > strain_dropout_prob).float()
            strain_scale = strain_scale * dropout_mask
            strain_shift = strain_shift * dropout_mask

        modulated = strain_scale * drug_latent + strain_shift * 0.1
        return modulated, strain_emb


class ContextEncoder(nn.Module):
    """Context encoder with embeddings and dropout for batch fields."""

    def __init__(self, context_vocabs, context_emb_dims, output_dim, dropout=0.15):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 3  # time, log_time, temp (continuous)

        for field, vocab in context_vocabs.items():
            n_cats = len(vocab)
            emb_dim = context_emb_dims.get(field, min(8, n_cats))
            self.embeddings[field] = nn.Embedding(n_cats, emb_dim)
            total_emb_dim += emb_dim

        self.output_proj = nn.Sequential(
            nn.Linear(total_emb_dim, output_dim),
            nn.GELU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout),
        )
        self.total_emb_dim = total_emb_dim

    def forward(self, ctx_indices, ctx_cont, dropout_fields=None):
        """Encode context.

        Args:
            ctx_indices: dict of field → [B] LongTensor
            ctx_cont: [B, 3] continuous features (time, log_time, temp)
            dropout_fields: set of field names to zero-out (batch dropout)
        """
        B = ctx_cont.size(0)
        parts = [ctx_cont]

        for field, emb in self.embeddings.items():
            indices = ctx_indices.get(field)
            if indices is None:
                continue
            emb_out = emb(indices)
            if self.training and dropout_fields and field in dropout_fields:
                drop_mask = (torch.rand(B, 1, device=indices.device) > 0.3).float()
                emb_out = emb_out * drop_mask
            parts.append(emb_out)

        combined = torch.cat(parts, dim=1)
        return self.output_proj(combined)


class ControlAnchoredLowRankModelV5(nn.Module):
    """
    Low-rank model with control-anchored prediction and FiLM strain modulation.

    Formula:
      final = control_pred + delta_pred
      control_pred = control_decoder(z_control) + control_center
      delta_pred = delta_decoder(z_delta) + delta_center
    """

    def __init__(self, n_proteins, n_strains, n_compounds,
                 context_vocabs, context_emb_dims,
                 drug_embed_dim=320, strain_embed_dim=192,
                 hidden_dim=256, film_rank=64,
                 control_rank=192, delta_rank=256,
                 dropout=0.15):
        super().__init__()
        self.n_proteins = n_proteins
        self.n_strains = n_strains
        self.control_rank = control_rank
        self.delta_rank = delta_rank

        # ── Decoders (fixed PCA, optionally fine-tuned) ──
        self.control_decoder = nn.Parameter(
            torch.randn(n_proteins, control_rank) * 0.01)
        self.delta_decoder = nn.Parameter(
            torch.randn(n_proteins, delta_rank) * 0.01)
        self.control_center = nn.Parameter(torch.zeros(n_proteins))
        self.delta_center = nn.Parameter(torch.zeros(n_proteins))

        # ── Chemical Encoder ──
        self.drug_embedding = nn.Embedding(n_compounds, drug_embed_dim)
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_embed_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Drug → delta latent projection
        self.drug_to_delta = nn.Linear(hidden_dim, delta_rank)

        # ── Strain Encoder (with UNK support) ──
        self.strain_embedding = nn.Embedding(n_strains, strain_embed_dim)
        self.unknown_strain_emb = nn.Parameter(
            torch.zeros(strain_embed_dim))

        # ── FiLM Strain Modulation ──
        self.film = FiLMStrainModulation(
            n_proteins, strain_embed_dim, film_rank, n_strains, dropout)

        # ── Context Encoder ──
        self.context_encoder = ContextEncoder(
            context_vocabs, context_emb_dims, hidden_dim, dropout)

        # ── Control Head: context + strain → control latent ──
        self.control_head = nn.Sequential(
            nn.Linear(hidden_dim + strain_embed_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, control_rank),
        )

        # ── Drug-Strain Interaction (low-rank bilinear) ──
        interaction_rank = 64
        self.W_d = nn.Linear(hidden_dim, interaction_rank, bias=False)
        self.W_s = nn.Linear(strain_embed_dim, interaction_rank, bias=False)
        self.interaction_proj = nn.Sequential(
            nn.Linear(interaction_rank, interaction_rank * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_rank * 2, delta_rank),
        )

        # ── Delta Head: combine drug + interaction + context → delta latent ──
        self.delta_head = nn.Sequential(
            nn.Linear(delta_rank * 2 + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, delta_rank),
        )

        # ── Time factor ──
        self.time_scale = nn.Parameter(torch.ones(delta_rank) * 0.5)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.05)

    def get_strain_emb(self, strain_idx, force_unknown=False):
        """Get strain embedding, with optional unknown mode."""
        if force_unknown:
            return self.unknown_strain_emb.unsqueeze(0).expand(
                strain_idx.size(0), -1)
        return self.strain_embedding(strain_idx)

    def forward(self, strain_idx, compound_idx, ctx_indices, ctx_cont,
                strain_dropout_prob=0.0, force_unknown=False,
                context_dropout_fields=None):
        """
        Args:
            strain_idx: [B]
            compound_idx: [B]
            ctx_indices: dict of field → [B] LongTensor
            ctx_cont: [B, 3] (time, log_time, temp)
            strain_dropout_prob: probability of zeroing strain modulation
            force_unknown: use UNK_STRAIN embedding
            context_dropout_fields: set of fields to zero-out
        """
        B = strain_idx.size(0)

        # ── Context encoding ──
        ctx_hidden = self.context_encoder(ctx_indices, ctx_cont,
                                          context_dropout_fields)

        # ── Control prediction ──
        strain_emb = self.get_strain_emb(strain_idx, force_unknown)
        z_control = self.control_head(torch.cat([ctx_hidden, strain_emb], dim=1))
        # [B, rank_c] @ [rank_c, N]^T → [B, N]
        control_pred = z_control @ self.control_decoder.T + self.control_center

        # ── Delta prediction ──
        drug_emb_raw = self.drug_embedding(compound_idx)
        drug_hidden = self.drug_encoder(drug_emb_raw)
        drug_delta = self.drug_to_delta(drug_hidden)   # [B, delta_rank]

        # FiLM strain modulation on drug delta (in protein space)
        drug_in_protein = drug_delta @ self.delta_decoder.T + self.delta_center
        strain_mod, film_strain_emb = self.film(
            strain_idx, drug_in_protein, strain_dropout_prob)

        # Drug-strain interaction (low-rank bilinear)
        d_repr = self.W_d(drug_hidden)
        s_repr = self.W_s(strain_emb)
        interaction = d_repr * s_repr                    # [B, interaction_rank]
        interaction_z = self.interaction_proj(interaction)  # [B, delta_rank]

        # Combine: drug + interaction → delta latent
        combined = torch.cat([drug_delta, interaction_z, ctx_hidden], dim=1)
        z_delta = self.delta_head(combined)

        # Delta in protein space
        delta_pred = z_delta @ self.delta_decoder.T + self.delta_center

        # ── Time modulation ──
        time_factor = ctx_cont[:, 0:1]  # [B, 1], normalized time
        delta_pred = delta_pred * time_factor * 0.1

        # ── Final prediction ──
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

    def predict_control(self, strain_idx, ctx_indices, ctx_cont):
        """Predict control abundance only."""
        ctx_hidden = self.context_encoder(ctx_indices, ctx_cont)
        strain_emb = self.get_strain_emb(strain_idx)
        z_control = self.control_head(torch.cat([ctx_hidden, strain_emb], dim=1))
        return z_control @ self.control_decoder.T + self.control_center

    def init_decoders_from_pca(self, control_pca_components, control_center,
                                delta_pca_components, delta_center):
        """Initialize decoders from PCA components."""
        with torch.no_grad():
            self.control_decoder.copy_(
                torch.tensor(control_pca_components[:, :self.control_rank],
                             dtype=torch.float32))
            self.control_center.copy_(
                torch.tensor(control_center, dtype=torch.float32))
            self.delta_decoder.copy_(
                torch.tensor(delta_pca_components[:, :self.delta_rank],
                             dtype=torch.float32))
            self.delta_center.copy_(
                torch.tensor(delta_center, dtype=torch.float32))


def compute_pca_basis(matrix, observed_mask, rank, random_state=42):
    """Compute PCA basis from training data (masked)."""
    from sklearn.decomposition import TruncatedSVD

    # Fill NaN for SVD
    col_means = np.nanmean(matrix, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    filled = matrix.copy()
    nan_mask = np.isnan(filled)
    filled[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    center = np.mean(filled, axis=0)
    centered = filled - center

    svd = TruncatedSVD(n_components=min(rank, min(centered.shape) - 1),
                       random_state=random_state)
    svd.fit(centered)
    components = svd.components_.T  # [P, rank]

    return components, center
