"""
Data processor v5: Leak-proof data loading for official competition.

Key fixes over v2:
  1. Chemical key = {data_source_family}::{normalized_name}
  2. Observed mask for NaN-aware loss computation
  3. Col_medians strictly from training fold
  4. Protein column name fix (1-Oct → OCT1)
  5. Context embeddings instead of large one-hot
  6. Test label quarantine
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
DATA_DIR = PROJECT_ROOT / "data"

ALLOW_TEST_LABELS = False

GENE_ALIAS_FIX = {
    "1-Oct": "OCT1",
}
OFFICIAL_PROTEIN_ORDER = None


def normalize_chemical_name(name):
    """Normalize chemical perturbation name."""
    if pd.isna(name):
        return "unknown"
    s = str(name).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^a-z0-9\s\-\(\)\[\],.+]+', '', s)
    return s


def make_chemical_key(data_source, chemical_name):
    """Create data-source-aware chemical key."""
    ds_family = str(data_source).split('_')[0].upper()
    return f"{ds_family}::{normalize_chemical_name(chemical_name)}"


def find_input_files(input_dir=None):
    """Auto-detect official input files (portable: 支持显式目录)."""
    d = Path(input_dir) if input_dir else INPUT_DIR
    meta_train = list(d.glob("*metadata_train_val*.csv"))
    meta_test = list(d.glob("*metadata_test*.csv"))
    proteome_train = list(d.glob("*proteome_raw_train_val*.csv"))
    proteome_test = list(d.glob("*proteome_raw_test*.csv"))

    assert meta_train, "No train metadata found"
    assert meta_test, "No test metadata found"
    assert proteome_train, "No train proteome found"

    return {
        'meta_train': str(meta_train[0]),
        'meta_test': str(meta_test[0]),
        'proteome_train': str(proteome_train[0]),
        'proteome_test': str(proteome_test[0]) if proteome_test else None,
    }


class DataProcessorV5:
    """Leak-proof data processor for model_v5."""

    def __init__(self, input_dir=None, allow_test_labels=False,
                 files=None):
        self.input_dir = Path(input_dir) if input_dir else INPUT_DIR
        self.allow_test_labels = allow_test_labels
        self._files = dict(files) if files else None

        # Core data
        self.train_meta = None
        self.test_meta = None
        self.train_matrix = None         # [N_train, P] log2 abundance
        self.test_matrix = None          # [N_test, P] or None if quarantined
        self.train_observed_mask = None  # [N_train, P] bool
        self.test_observed_mask = None   # [N_test, P] bool or None

        # Protein info
        self.proteins = None
        self.protein_to_idx = {}
        self.n_proteins = 0

        # Encoding
        self.strains = []
        self.compounds = []
        self.chemical_keys = []
        self.strain_to_idx = {}
        self.compound_to_idx = {}
        self.chemical_key_map = {}
        self.chemical_collision_report = []

        # Context categories
        self.context_vocabs = {}
        self.context_emb_dims = {}

        # Imputation
        self.col_medians_train = None

        # Graph (optional, for open mode)
        self.edge_index = None
        self.edge_weight = None
        self.protein_features = None
        self.protein_features_dim = 0
        self.pathway_dict = {}
        self.pathway_names = {}
        self.protein_pathways = {}

    def load_all(self, score_threshold=700):
        print("[DataProcessorV5] Loading data...")
        files = self._files if self._files else find_input_files(self.input_dir)

        self._load_metadata(files)
        self._load_proteome(files)
        self._encode_entities()
        self._load_string(score_threshold)
        self._load_kegg()
        self._build_protein_features()
        print(f"[DataProcessorV5] Done. {self.n_proteins} proteins, "
              f"{len(self.train_meta)} train, {len(self.test_meta)} test")
        return self

    def _load_metadata(self, files):
        self.train_meta = pd.read_csv(files['meta_train'])
        self.test_meta = pd.read_csv(files['meta_test'])

        if not self.allow_test_labels:
            print("  [SECURITY] TEST LABEL QUARANTINE: ENABLED")
            print("  Test protein truth will NOT be loaded.")
        else:
            print("  [WARNING] Test labels WILL be loaded (diagnostic mode)")

        assert len(set(self.test_meta['sample_ID']) &
                   set(self.train_meta['sample_ID'])) == 0, \
            "sample_ID overlap between train and test!"

    def _load_proteome(self, files):
        train_raw = pd.read_csv(files['proteome_train'])
        protein_cols = [c for c in train_raw.columns if c != 'sample_ID']

        # Apply gene alias fixes
        global OFFICIAL_PROTEIN_ORDER
        OFFICIAL_PROTEIN_ORDER = list(protein_cols)
        fixed_cols = []
        for c in protein_cols:
            if c in GENE_ALIAS_FIX:
                fixed_cols.append(GENE_ALIAS_FIX[c])
            elif ',' in c or "'" in c:
                fixed_cols.append(c)  # Keep complex names as-is
            else:
                fixed_cols.append(c)
        self.proteins = fixed_cols
        self.n_proteins = len(self.proteins)
        self.protein_to_idx = {p: i for i, p in enumerate(self.proteins)}
        self.protein_name_to_official = dict(zip(fixed_cols, protein_cols))

        # Align and convert to log2
        train_raw_idx = train_raw.set_index('sample_ID')
        train_mat = train_raw_idx.loc[self.train_meta['sample_ID'].values,
                                       protein_cols].values.astype(np.float64)
        # Log2 transform
        with np.errstate(divide='ignore', invalid='ignore'):
            train_mat = np.where(train_mat > 0, np.log2(train_mat), np.nan)

        self.train_observed_mask = np.isfinite(train_mat)
        self.train_matrix = train_mat

        # Impute from training data only
        with np.errstate(all='ignore'):
            col_medians = np.nanmedian(train_mat, axis=0)
        col_medians = np.where(np.isnan(col_medians),
                               np.nanmedian(train_mat), col_medians)
        col_medians[np.isnan(col_medians)] = 0.0
        self.col_medians_train = col_medians

        train_filled = train_mat.copy()
        nan_mask = np.isnan(train_filled)
        train_filled[nan_mask] = np.take(col_medians,
                                         np.where(nan_mask)[1])
        self.train_matrix_filled = train_filled

        # Missingness report
        missing_per_sample = (~self.train_observed_mask).mean(axis=1)
        missing_per_protein = (~self.train_observed_mask).mean(axis=0)
        print(f"  Train proteome: {self.n_proteins} proteins, "
              f"{len(train_mat)} samples")
        print(f"  Missing rate: sample mean={missing_per_sample.mean():.3f}, "
              f"protein mean={missing_per_protein.mean():.3f}")

        # Test proteome (quarantined unless explicitly allowed)
        if files['proteome_test'] and self.allow_test_labels:
            test_raw = pd.read_csv(files['proteome_test'])
            test_raw_idx = test_raw.set_index('sample_ID')
            test_mat = test_raw_idx.loc[self.test_meta['sample_ID'].values,
                                         protein_cols].values.astype(np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                test_mat = np.where(test_mat > 0, np.log2(test_mat), np.nan)
            self.test_observed_mask = np.isfinite(test_mat)
            self.test_matrix = test_mat

            test_filled = test_mat.copy()
            test_nan = np.isnan(test_filled)
            test_filled[test_nan] = np.take(col_medians,
                                            np.where(test_nan)[1])
            self.test_matrix_filled = test_filled
            print(f"  Test proteome: {len(test_mat)} samples (DIAGNOSTIC)")
        else:
            self.test_matrix = None
            self.test_observed_mask = None
            self.test_matrix_filled = None
            print(f"  Test proteome: {len(self.test_meta)} samples "
                  f"(TRUTH QUARANTINED)")

    def _encode_entities(self):
        all_meta = pd.concat([self.train_meta, self.test_meta], ignore_index=True)

        # Strains
        self.strains = sorted(all_meta['Strains'].unique())
        self.strain_to_idx = {s: i for i, s in enumerate(self.strains)}

        # Chemical keys (data-source-aware)
        all_meta['_chem_name'] = all_meta['perturbation_no_concentration'].apply(
            normalize_chemical_name)
        all_meta['_chem_key'] = all_meta.apply(
            lambda r: make_chemical_key(r['data_source'], r['_chem_name']), axis=1)

        # Collision detection
        pert_to_keys = defaultdict(set)
        for _, row in all_meta.iterrows():
            pert_to_keys[row['pert_id']].add(row['_chem_key'])
        collision_count = sum(1 for v in pert_to_keys.values() if len(v) > 1)
        print(f"  Chemical collisions (same pert_id, different keys): {collision_count}")

        self.chemical_keys = sorted(all_meta['_chem_key'].unique())
        self.compound_to_idx = {k: i for i, k in enumerate(self.chemical_keys)}
        all_meta['_compound_idx'] = all_meta['_chem_key'].map(self.compound_to_idx)

        # Map back to standardized name
        self.chemical_key_map = {
            k: all_meta[all_meta['_chem_key'] == k]['perturbation_no_concentration'].iloc[0]
            for k in self.chemical_keys
        }

        # Store compound indices in metadata
        self.train_meta['_chem_key'] = self.train_meta.apply(
            lambda r: make_chemical_key(r['data_source'],
                          normalize_chemical_name(r['perturbation_no_concentration'])),
            axis=1)
        self.test_meta['_chem_key'] = self.test_meta.apply(
            lambda r: make_chemical_key(r['data_source'],
                          normalize_chemical_name(r['perturbation_no_concentration'])),
            axis=1)
        self.train_meta['_compound_idx'] = self.train_meta['_chem_key'].map(
            self.compound_to_idx)
        self.test_meta['_compound_idx'] = self.test_meta['_chem_key'].map(
            self.compound_to_idx)

        # Context embeddings setup
        self.context_vocabs = {}
        self.context_emb_dims = {
            'data_source': 8,
            'Medium': 4,
            'instrument': 8,
            'Yeast_cell_plate': 12,
            'protein_well_row': 4,
            'protein_well_col': 6,
        }
        for field in ['data_source', 'Medium', 'instrument', 'Yeast_cell_plate']:
            values = sorted(all_meta[field].astype(str).unique())
            self.context_vocabs[field] = {v: i for i, v in enumerate(values)}
        # protein_well → row + col
        wells = sorted(all_meta['protein_well'].astype(str).unique())
        row_vals = sorted(set(w[0] for w in wells if len(w) >= 2))
        col_vals = sorted(set(w[1:] for w in wells if len(w) >= 2))
        self.context_vocabs['protein_well_row'] = {r: i for i, r in enumerate(row_vals)}
        self.context_vocabs['protein_well_col'] = {c: i for i, c in enumerate(col_vals)}

        print(f"  Strains: {len(self.strains)}, Chemical keys: {len(self.chemical_keys)}")
        print(f"  Context fields: {list(self.context_vocabs.keys())}")

    def encode_context_batch(self, meta_df, device='cpu'):
        """Encode context as list of category indices (for embeddings)."""
        n = len(meta_df)
        indices = {}
        for field in ['data_source', 'Medium', 'instrument', 'Yeast_cell_plate']:
            vocab = self.context_vocabs[field]
            vals = meta_df[field].astype(str).map(
                lambda x: vocab.get(x, 0)).fillna(0).values
            indices[field] = torch.tensor(vals, dtype=torch.long, device=device)

        # protein_well → row + col
        well_vals = meta_df['protein_well'].astype(str)
        row_vocab = self.context_vocabs['protein_well_row']
        col_vocab = self.context_vocabs['protein_well_col']
        row_idx = well_vals.map(lambda x: row_vocab.get(x[0], 0) if len(x) >= 1 else 0)
        col_idx = well_vals.map(lambda x: col_vocab.get(x[1:], 0) if len(x) >= 2 else 0)
        indices['protein_well_row'] = torch.tensor(row_idx.fillna(0).values,
                                                    dtype=torch.long, device=device)
        indices['protein_well_col'] = torch.tensor(col_idx.fillna(0).values,
                                                    dtype=torch.long, device=device)

        # Continuous features
        time_feat = torch.tensor(
            meta_df['pert_time'].values / 240.0, dtype=torch.float32, device=device)
        temp_feat = torch.tensor(
            meta_df['Temperature'].values / 37.0, dtype=torch.float32, device=device)
        log_time = torch.log1p(time_feat)

        cont = torch.stack([time_feat, log_time, temp_feat], dim=1)  # [N, 3]

        return indices, cont

    def encode_sample_batch(self, meta_df, device='cpu'):
        """Encode strain_idx, compound_idx, and context for a batch."""
        strain_idx = torch.tensor(
            [self.strain_to_idx[s] for s in meta_df['Strains']],
            dtype=torch.long, device=device)
        compound_idx = torch.tensor(
            meta_df['_compound_idx'].values, dtype=torch.long, device=device)
        ctx_indices, ctx_cont = self.encode_context_batch(meta_df, device)
        return strain_idx, compound_idx, ctx_indices, ctx_cont

    # ── STRING PPI ──
    def _load_string(self, score_threshold=700):
        path = DATA_DIR / "string" / "protein.links.txt"
        if not path.exists():
            return

        gene_to_orf = self._build_gene_orf_mapping()
        string_to_idx = {}
        for i, p in enumerate(self.proteins):
            orf = gene_to_orf.get(p)
            if orf:
                string_to_idx[f"4932.{orf}"] = i

        edges = []
        for chunk in pd.read_csv(str(path), sep=' ', chunksize=500000):
            chunk = chunk[chunk['combined_score'] >= score_threshold]
            edges.append(chunk)
        df = pd.concat(edges, ignore_index=True)

        src_list, dst_list, w_list = [], [], []
        for _, row in df.iterrows():
            if row['protein1'] in string_to_idx and row['protein2'] in string_to_idx:
                src_list.append(string_to_idx[row['protein1']])
                dst_list.append(string_to_idx[row['protein2']])
                w_list.append(row['combined_score'] / 1000.0)

        all_src = src_list + dst_list
        all_dst = dst_list + src_list
        all_w = np.concatenate([w_list, w_list])

        self.edge_index = torch.tensor(np.array([all_src, all_dst]), dtype=torch.long)
        self.edge_weight = torch.tensor(all_w, dtype=torch.float32)
        nodes_with_edges = len(set(all_src))
        print(f"  STRING: {nodes_with_edges}/{self.n_proteins} proteins, "
              f"{len(all_src)} undirected edges")

    def _build_gene_orf_mapping(self):
        gaf_path = DATA_DIR / "go" / "sgd.gaf"
        gene_to_orf = {}
        if not gaf_path.exists():
            return gene_to_orf
        orf_pattern = re.compile(r'^[YQ][A-Z]{2}\d{3}[CW]$|^Q\d{4,5}$')
        with open(gaf_path, 'r') as f:
            for line in f:
                if line.startswith('!'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 11:
                    continue
                gene_symbol = parts[2]
                aliases = parts[10]
                for alias in aliases.split('|'):
                    if orf_pattern.match(alias):
                        gene_to_orf[gene_symbol] = alias
                        break
        for p in self.proteins:
            if p not in gene_to_orf and orf_pattern.match(p):
                gene_to_orf[p] = p
        return gene_to_orf

    def _load_kegg(self):
        genes_path = DATA_DIR / "kegg" / "pathway_genes.txt"
        list_path = DATA_DIR / "kegg" / "pathway_list.txt"

        if list_path.exists():
            with open(list_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        self.pathway_names[parts[0].replace('path:', '')] = parts[1]

        if not genes_path.exists():
            self._create_pseudo_pathways()
            return

        gene_to_orf = self._build_gene_orf_mapping()
        orf_to_idx = {}
        for i, p in enumerate(self.proteins):
            orf = gene_to_orf.get(p)
            if orf:
                orf_to_idx[orf] = i

        with open(genes_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    gene_id = parts[2].replace('sce:', '')
                    if gene_id in orf_to_idx:
                        pw_id = parts[0]
                        idx = orf_to_idx[gene_id]
                        self.pathway_dict.setdefault(pw_id, set()).add(idx)
                        self.protein_pathways.setdefault(idx, []).append(pw_id)

        if not self.pathway_dict:
            self._create_pseudo_pathways()
        print(f"  KEGG: {len(self.pathway_dict)} pathways")

    def _create_pseudo_pathways(self, n_pw=50):
        if self.edge_index is not None:
            deg = torch.zeros(self.n_proteins)
            deg.scatter_add_(0, self.edge_index[0],
                             torch.ones(self.edge_index.shape[1]))
            core = torch.argsort(deg, descending=True)[:min(self.n_proteins, 512)].tolist()
            pw_size = max(1, len(core) // n_pw)
            for i in range(n_pw):
                members = set(core[i * pw_size:(i + 1) * pw_size])
                pw_id = f"PS{i:02d}"
                self.pathway_dict[pw_id] = members
                self.pathway_names[pw_id] = f"Cluster_{i}"
                for idx in members:
                    self.protein_pathways.setdefault(idx, []).append(pw_id)

    def _build_protein_features(self):
        if self.n_proteins == 0:
            return
        pw_ids = sorted(self.pathway_dict.keys())
        n_pw = len(pw_ids)
        n_feat = n_pw + 1
        features = np.zeros((self.n_proteins, n_feat), dtype=np.float32)
        for pwi, pw_id in enumerate(pw_ids):
            for idx in self.pathway_dict[pw_id]:
                if 0 <= idx < self.n_proteins:
                    features[idx, pwi] = 1.0
        if self.edge_index is not None:
            deg = torch.zeros(self.n_proteins)
            deg.scatter_add_(0, self.edge_index[0],
                             torch.ones(self.edge_index.shape[1]))
            if deg.max() > 0:
                features[:, -1] = (deg / deg.max()).numpy()
        self.protein_features = torch.tensor(features, dtype=torch.float32)
        self.protein_features_dim = n_feat
