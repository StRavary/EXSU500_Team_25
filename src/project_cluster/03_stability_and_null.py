"""
03_stability_and_null.py
Training-set (n = 2,460) robustness and statistical validity benchmarks:
  1. 100-iteration bootstrap cluster stability (Hungarian-matched Jaccard similarity)
  2. 100-iteration column-wise permutation null baseline (empirical p-values)
  3. Multi-seed initialization consistency (pairwise ARI across 5 Autoencoder seeds)
"""

from importlib.util import module_from_spec, spec_from_file_location
from itertools import combinations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
import torch

# Load model definitions and training helpers directly from 02_train_and_tune.py
_SPEC_02 = spec_from_file_location(
    "mod02", Path("src/project_cluster/02_train_and_tune.py")
)
_MOD_02 = module_from_spec(_SPEC_02)
assert _SPEC_02 is not None and _SPEC_02.loader is not None
_SPEC_02.loader.exec_module(_MOD_02)

UndercompleteAutoencoder = _MOD_02.UndercompleteAutoencoder
train_autoencoder = _MOD_02.train_autoencoder
encode_latent = _MOD_02.encode_latent

# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------
PREPARED_DIR = Path("artifacts") / "prepared"
TUNING_DIR = Path("artifacts") / "tuning"
STABILITY_DIR = Path("artifacts") / "stability"

BIOMARKERS = [
    "HbA1c_Percent",
    "Triglycerides",
    "Total_Cholesterol",
    "HDL_Cholesterol",
    "BMI",
    "Systolic_BP",
]

N_BOOTSTRAP = 100
N_PERMUTATIONS = 100
BASE_SEED = 42
AE_SEEDS = [42, 43, 44, 45, 46]


# ------------------------------------------------------------------------------
# Hungarian-Matched Bootstrap Jaccard Similarity
# ------------------------------------------------------------------------------
def hungarian_jaccard(
    ref_labels: np.ndarray, boot_labels: np.ndarray, k: int
) -> tuple[float, list[float]]:
    """
    Align bootstrap cluster IDs to reference cluster IDs via optimal bipartite matching
    and compute per-cluster and mean Jaccard similarity on shared in-bag indices.
    """
    jaccard_matrix = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        ref_mask = ref_labels == i
        for j in range(k):
            boot_mask = boot_labels == j
            union = np.logical_or(ref_mask, boot_mask).sum()
            if union > 0:
                inter = np.logical_and(ref_mask, boot_mask).sum()
                jaccard_matrix[i, j] = inter / union

    row_ind, col_ind = linear_sum_assignment(-jaccard_matrix)
    matched_jaccards = [float(jaccard_matrix[r, c]) for r, c in zip(row_ind, col_ind)]
    return float(np.mean(matched_jaccards)), matched_jaccards


# ------------------------------------------------------------------------------
# Model Fit/Predict Wrappers
# ------------------------------------------------------------------------------
def fit_predict_kmeans(
    X: np.ndarray, k: int, seed: int
) -> tuple[np.ndarray, float, float]:
    km = KMeans(n_clusters=k, init="k-means++", n_init=30, random_state=seed).fit(X)
    labels = km.labels_
    sil = float(silhouette_score(X, labels))
    db = float(davies_bouldin_score(X, labels))
    return labels, sil, db


def fit_predict_gmm(
    X: np.ndarray, k: int, seed: int
) -> tuple[np.ndarray, float, float]:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=10,
        reg_covar=1e-5,
        random_state=seed,
    ).fit(X)
    labels = gmm.predict(X)
    if len(np.unique(labels)) < 2:
        return labels, float("nan"), float("nan")
    sil = float(silhouette_score(X, labels))
    db = float(davies_bouldin_score(X, labels))
    return labels, sil, db


def fit_predict_ae(
    X: np.ndarray, k: int, bottleneck_dim: int, seed: int, epochs: int = 100
) -> tuple[np.ndarray, float, float, float]:
    ae, _ = train_autoencoder(
        X, bottleneck_dim=bottleneck_dim, seed=seed, epochs=epochs
    )
    Z = encode_latent(ae, X)
    km = KMeans(n_clusters=k, init="k-means++", n_init=30, random_state=seed).fit(Z)
    labels = km.labels_
    if len(np.unique(labels)) < 2:
        return labels, float("nan"), float("nan"), float("nan")
    sil_latent = float(silhouette_score(Z, labels))
    sil_ambient = float(silhouette_score(X, labels))
    db_ambient = float(davies_bouldin_score(X, labels))
    return labels, sil_latent, sil_ambient, db_ambient


# ------------------------------------------------------------------------------
# Main Stability & Permutation Null Routines
# ------------------------------------------------------------------------------
def main() -> None:
    STABILITY_DIR.mkdir(parents=True, exist_ok=True)

    train_scaled = pd.read_parquet(PREPARED_DIR / "train_scaled.parquet")
    X_train = train_scaled[BIOMARKERS].to_numpy(dtype=np.float64)
    n_samples, n_features = X_train.shape

    with open(TUNING_DIR / "selected_models.json", "r", encoding="utf-8") as f:
        selected = json.load(f)

    models_dir = TUNING_DIR / "fitted_train_models"

    # Load reference full-training partitions
    km_ref = joblib.load(models_dir / "kmeans_full_train.joblib")
    ref_labels_km = km_ref.predict(X_train)
    obs_sil_km = float(silhouette_score(X_train, ref_labels_km))

    gmm_ref = joblib.load(models_dir / "gmm_full_train.joblib")
    ref_labels_gmm = gmm_ref.predict(X_train)
    obs_sil_gmm = float(silhouette_score(X_train, ref_labels_gmm))

    k_ae = selected["Autoencoder"]["k"]
    b_dim = selected["Autoencoder"]["bottleneck_dim"]

    # Evaluate multi-seed consistency for Autoencoder on full training set
    ae_seed_labels: dict[int, np.ndarray] = {}
    ae_seed_sils_latent: list[float] = []
    ae_seed_sils_ambient: list[float] = []

    for seed in AE_SEEDS:
        ae_state = models_dir / f"ae_dim{b_dim}_seed{seed}.pt"
        km_ae_ref = joblib.load(models_dir / f"ae_kmeans_dim{b_dim}_seed{seed}.joblib")

        ae_net = UndercompleteAutoencoder(input_dim=n_features, bottleneck_dim=b_dim)
        ae_net.load_state_dict(
            torch.load(ae_state, map_location="cpu", weights_only=True)
        )
        ae_net.eval()

        Z_ref = encode_latent(ae_net, X_train)
        lbls = km_ae_ref.predict(Z_ref)
        ae_seed_labels[seed] = lbls
        ae_seed_sils_latent.append(float(silhouette_score(Z_ref, lbls)))
        ae_seed_sils_ambient.append(float(silhouette_score(X_train, lbls)))

    pairwise_aris = [
        float(adjusted_rand_score(ae_seed_labels[s1], ae_seed_labels[s2]))
        for s1, s2 in combinations(AE_SEEDS, 2)
    ]
    ref_labels_ae = ae_seed_labels[BASE_SEED]
    obs_sil_ae_latent = ae_seed_sils_latent[0]
    obs_sil_ae_ambient = ae_seed_sils_ambient[0]

    print("=== Autoencoder Multi-Seed Consistency (Full Train, Seeds 42..46) ===")
    print(
        f"Pairwise ARI across 5 seeds: {np.mean(pairwise_aris):.4f} +/- {np.std(pairwise_aris):.4f}"
    )
    print(f"Observed Silhouette (Latent {b_dim}D, seed 42): {obs_sil_ae_latent:.4f}")
    print(f"Observed Silhouette (Ambient 6D, seed 42): {obs_sil_ae_ambient:.4f}\n")

    # --------------------------------------------------------------------------
    # 1. 100-Bootstrap Cluster Stability
    # --------------------------------------------------------------------------
    print(
        f"Running {N_BOOTSTRAP} bootstrap resamples for cluster stability (Jaccard)..."
    )
    rng = np.random.default_rng(BASE_SEED)
    boot_records = []

    k_km = selected["k-Means"]["k"]
    k_gmm = selected["GMM"]["k"]

    for b in range(1, N_BOOTSTRAP + 1):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_train[boot_idx]

        # k-Means bootstrap
        lbl_km, _, _ = fit_predict_kmeans(X_boot, k=k_km, seed=BASE_SEED + b)
        j_km, _ = hungarian_jaccard(ref_labels_km[boot_idx], lbl_km, k=k_km)
        boot_records.append(
            {"model_family": "k-Means", "bootstrap_iter": b, "mean_jaccard": j_km}
        )

        # GMM bootstrap
        lbl_gmm, _, _ = fit_predict_gmm(X_boot, k=k_gmm, seed=BASE_SEED + b)
        j_gmm, _ = hungarian_jaccard(ref_labels_gmm[boot_idx], lbl_gmm, k=k_gmm)
        boot_records.append(
            {"model_family": "GMM", "bootstrap_iter": b, "mean_jaccard": j_gmm}
        )

        # Autoencoder bootstrap
        lbl_ae, _, _, _ = fit_predict_ae(
            X_boot, k=k_ae, bottleneck_dim=b_dim, seed=BASE_SEED + b, epochs=100
        )
        j_ae, _ = hungarian_jaccard(ref_labels_ae[boot_idx], lbl_ae, k=k_ae)
        boot_records.append(
            {"model_family": "Autoencoder", "bootstrap_iter": b, "mean_jaccard": j_ae}
        )

        if b % 25 == 0:
            print(f"  Completed {b}/{N_BOOTSTRAP} bootstrap iterations")

    boot_df = pd.DataFrame(boot_records)
    boot_df.to_csv(STABILITY_DIR / "bootstrap_jaccard_iterations.csv", index=False)

    # --------------------------------------------------------------------------
    # 2. 100-Iteration Column-Wise Permutation Null Baseline
    # --------------------------------------------------------------------------
    print(f"\nRunning {N_PERMUTATIONS} feature-permutation null trials...")
    null_records = []

    for p in range(1, N_PERMUTATIONS + 1):
        # Independently permute each biomarker column to destroy multivariate structure
        X_null = np.column_stack(
            [rng.permutation(X_train[:, col]) for col in range(n_features)]
        )

        _, sil_km_null, _ = fit_predict_kmeans(X_null, k=k_km, seed=BASE_SEED + p)
        null_records.append(
            {
                "model_family": "k-Means",
                "perm_iter": p,
                "null_silhouette_ambient": sil_km_null,
                "null_silhouette_latent": np.nan,
            }
        )

        _, sil_gmm_null, _ = fit_predict_gmm(X_null, k=k_gmm, seed=BASE_SEED + p)
        null_records.append(
            {
                "model_family": "GMM",
                "perm_iter": p,
                "null_silhouette_ambient": sil_gmm_null,
                "null_silhouette_latent": np.nan,
            }
        )

        _, sil_ae_lat_null, sil_ae_amb_null, _ = fit_predict_ae(
            X_null, k=k_ae, bottleneck_dim=b_dim, seed=BASE_SEED + p, epochs=100
        )
        null_records.append(
            {
                "model_family": "Autoencoder",
                "perm_iter": p,
                "null_silhouette_ambient": sil_ae_amb_null,
                "null_silhouette_latent": sil_ae_lat_null,
            }
        )

        if p % 25 == 0:
            print(f"  Completed {p}/{N_PERMUTATIONS} permutation null iterations")

    null_df = pd.DataFrame(null_records)
    null_df.to_csv(STABILITY_DIR / "permutation_null_iterations.csv", index=False)

    # --------------------------------------------------------------------------
    # Aggregate Summary Report
    # --------------------------------------------------------------------------
    summary_rows = []
    obs_map = {
        "k-Means": (obs_sil_km, np.nan),
        "GMM": (obs_sil_gmm, np.nan),
        "Autoencoder": (obs_sil_ae_ambient, obs_sil_ae_latent),
    }

    for family in ["k-Means", "GMM", "Autoencoder"]:
        b_vals = boot_df[boot_df["model_family"] == family]["mean_jaccard"]
        n_sub = null_df[null_df["model_family"] == family]
        obs_amb, obs_lat = obs_map[family]

        null_amb = n_sub["null_silhouette_ambient"].dropna()
        p_amb = float((np.sum(null_amb >= obs_amb) + 1) / (len(null_amb) + 1))

        if family == "Autoencoder":
            null_lat = n_sub["null_silhouette_latent"].dropna()
            p_lat = float((np.sum(null_lat >= obs_lat) + 1) / (len(null_lat) + 1))
            null_lat_mean = float(null_lat.mean())
            null_lat_sd = float(null_lat.std())
        else:
            p_lat = np.nan
            null_lat_mean = np.nan
            null_lat_sd = np.nan

        summary_rows.append(
            {
                "model_family": family,
                "boot_jaccard_mean": round(float(b_vals.mean()), 4),
                "boot_jaccard_sd": round(float(b_vals.std()), 4),
                "boot_jaccard_q025": round(float(b_vals.quantile(0.025)), 4),
                "boot_jaccard_q975": round(float(b_vals.quantile(0.975)), 4),
                "obs_sil_ambient_6d": round(obs_amb, 4),
                "null_sil_ambient_mean": round(float(null_amb.mean()), 4),
                "null_sil_ambient_sd": round(float(null_amb.std()), 4),
                "perm_p_ambient": round(p_amb, 4),
                "obs_sil_latent": round(obs_lat, 4)
                if not np.isnan(obs_lat)
                else np.nan,
                "null_sil_latent_mean": round(null_lat_mean, 4)
                if not np.isnan(null_lat_mean)
                else np.nan,
                "null_sil_latent_sd": round(null_lat_sd, 4)
                if not np.isnan(null_lat_sd)
                else np.nan,
                "perm_p_latent": round(p_lat, 4) if not np.isnan(p_lat) else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(STABILITY_DIR / "stability_and_null_summary.csv", index=False)

    seed_meta = {
        "ae_seeds": AE_SEEDS,
        "ae_pairwise_ari_mean": round(float(np.mean(pairwise_aris)), 4),
        "ae_pairwise_ari_sd": round(float(np.std(pairwise_aris)), 4),
        "ae_seed_silhouettes_latent": [round(x, 4) for x in ae_seed_sils_latent],
        "ae_seed_silhouettes_ambient": [round(x, 4) for x in ae_seed_sils_ambient],
    }
    with open(STABILITY_DIR / "ae_seed_consistency.json", "w", encoding="utf-8") as f:
        json.dump(seed_meta, f, indent=2)

    print("\n=== Bootstrap Stability & Permutation Null Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved stability and null artifacts to {STABILITY_DIR}/")


if __name__ == "__main__":
    main()
