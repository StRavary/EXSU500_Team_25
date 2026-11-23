"""
02_train_and_tune.py
Five-fold cross-validation and hyperparameter tuning on the training set (n = 2,460)
across k in {2, 3, 4, 5, 6} for:
  1. k-Means (baseline floor, k-means++, 50 restarts)
  2. Gaussian Mixture Models (full covariance EM, BIC + internal metrics)
  3. Undercomplete Autoencoder + Latent k-Means (bottleneck dim in {2, 3}, 5 seeds)
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------
PREPARED_DIR = Path("artifacts") / "prepared"
TUNING_DIR = Path("artifacts") / "tuning"

BIOMARKERS = [
    "HbA1c_Percent",
    "Triglycerides",
    "Total_Cholesterol",
    "HDL_Cholesterol",
    "BMI",
    "Systolic_BP",
]

K_CANDIDATES = [2, 3, 4, 5, 6]
N_FOLDS = 5
BASE_SEED = 42
AE_SEEDS = [42, 43, 44, 45, 46]
AE_BOTTLENECK_DIMS = [2, 3]


# ------------------------------------------------------------------------------
# PyTorch Undercomplete Autoencoder Architecture
# ------------------------------------------------------------------------------
class UndercompleteAutoencoder(nn.Module):
    """Compact symmetric autoencoder for 6D tabular biomarker manifold learning."""

    def __init__(self, input_dim: int = 6, bottleneck_dim: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.SiLU(),
            nn.Linear(16, 8),
            nn.SiLU(),
            nn.Linear(8, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 8),
            nn.SiLU(),
            nn.Linear(8, 16),
            nn.LayerNorm(16),
            nn.SiLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def train_autoencoder(
    X_train: np.ndarray,
    bottleneck_dim: int,
    seed: int,
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[UndercompleteAutoencoder, float]:
    """Train the undercomplete autoencoder on CPU/GPU and return model + final MSE."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UndercompleteAutoencoder(
        input_dim=X_train.shape[1], bottleneck_dim=bottleneck_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(batch_x)
            loss = criterion(x_hat, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
        final_loss = epoch_loss / len(X_train)

    model.eval()
    return model, final_loss


@torch.no_grad()
def encode_latent(model: UndercompleteAutoencoder, X: np.ndarray) -> np.ndarray:
    """Project standardized features into the learned bottleneck space."""
    device = next(model.parameters()).device
    tensor_x = torch.tensor(X, dtype=torch.float32, device=device)
    _, z = model(tensor_x)
    return z.cpu().numpy()


# ------------------------------------------------------------------------------
# Metric Helpers
# ------------------------------------------------------------------------------
def compute_internal_metrics(X: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute Silhouette (higher is better) and Davies-Bouldin (lower is better)."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return float("nan"), float("nan")
    sil = float(silhouette_score(X, labels, metric="euclidean"))
    db = float(davies_bouldin_score(X, labels))
    return sil, db


# ------------------------------------------------------------------------------
# 5-Fold Cross-Validation Routines
# ------------------------------------------------------------------------------
def run_cv_kmeans(X: np.ndarray, strata: np.ndarray) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BASE_SEED)
    records = []

    for k in K_CANDIDATES:
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, strata), start=1):
            X_tr, X_val = X[tr_idx], X[val_idx]
            km = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=50,
                random_state=BASE_SEED + fold_idx,
            )
            km.fit(X_tr)
            val_labels = km.predict(X_val)
            sil, db = compute_internal_metrics(X_val, val_labels)
            records.append(
                {
                    "model_family": "k-Means",
                    "hyperparams": f"k={k}, n_init=50",
                    "k": k,
                    "bottleneck_dim": np.nan,
                    "seed": BASE_SEED,
                    "fold": fold_idx,
                    "val_silhouette": sil,
                    "val_davies_bouldin": db,
                    "val_bic": np.nan,
                    "train_recon_mse": np.nan,
                }
            )
    return pd.DataFrame(records)


def run_cv_gmm(X: np.ndarray, strata: np.ndarray) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BASE_SEED)
    records = []

    for k in K_CANDIDATES:
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, strata), start=1):
            X_tr, X_val = X[tr_idx], X[val_idx]
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=20,
                reg_covar=1e-5,
                random_state=BASE_SEED + fold_idx,
            )
            gmm.fit(X_tr)
            val_labels = gmm.predict(X_val)
            sil, db = compute_internal_metrics(X_val, val_labels)
            val_bic = float(gmm.bic(X_val))
            records.append(
                {
                    "model_family": "GMM",
                    "hyperparams": f"k={k}, cov=full",
                    "k": k,
                    "bottleneck_dim": np.nan,
                    "seed": BASE_SEED,
                    "fold": fold_idx,
                    "val_silhouette": sil,
                    "val_davies_bouldin": db,
                    "val_bic": val_bic,
                    "train_recon_mse": np.nan,
                }
            )
    return pd.DataFrame(records)


def run_cv_autoencoder(X: np.ndarray, strata: np.ndarray) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BASE_SEED)
    records = []

    for b_dim in AE_BOTTLENECK_DIMS:
        for seed in AE_SEEDS:
            for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, strata), start=1):
                X_tr, X_val = X[tr_idx], X[val_idx]
                ae_model, recon_mse = train_autoencoder(
                    X_tr, bottleneck_dim=b_dim, seed=seed + fold_idx
                )
                Z_tr = encode_latent(ae_model, X_tr)
                Z_val = encode_latent(ae_model, X_val)

                for k in K_CANDIDATES:
                    km = KMeans(
                        n_clusters=k,
                        init="k-means++",
                        n_init=50,
                        random_state=seed + fold_idx,
                    )
                    km.fit(Z_tr)
                    val_labels = km.predict(Z_val)
                    # Evaluate internal separation in the latent space
                    sil, db = compute_internal_metrics(Z_val, val_labels)
                    records.append(
                        {
                            "model_family": "Autoencoder",
                            "hyperparams": f"k={k}, dim={b_dim}",
                            "k": k,
                            "bottleneck_dim": b_dim,
                            "seed": seed,
                            "fold": fold_idx,
                            "val_silhouette": sil,
                            "val_davies_bouldin": db,
                            "val_bic": np.nan,
                            "train_recon_mse": recon_mse,
                        }
                    )
    return pd.DataFrame(records)


# ------------------------------------------------------------------------------
# Main Execution & Full-Training Refitting
# ------------------------------------------------------------------------------
def main() -> None:
    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    train_scaled = pd.read_parquet(PREPARED_DIR / "train_scaled.parquet")
    train_raw = pd.read_parquet(PREPARED_DIR / "train_raw.parquet")

    X_train = train_scaled[BIOMARKERS].to_numpy(dtype=np.float64)
    # Stratify CV folds by age decile, matching the outer train/test split
    age_strata = pd.qcut(
        train_raw["Age_Years"], q=10, labels=False, duplicates="drop"
    ).to_numpy()

    print(
        f"Loaded training matrix: {X_train.shape[0]:,} participants x {X_train.shape[1]} biomarkers"
    )
    print("Running 5-fold CV for k-Means (k = 2..6)...")
    df_km = run_cv_kmeans(X_train, age_strata)

    print("Running 5-fold CV for Gaussian Mixture Models (k = 2..6)...")
    df_gmm = run_cv_gmm(X_train, age_strata)

    print(
        "Running 5-fold CV for Undercomplete Autoencoder (dim in {2,3}, 5 seeds, k = 2..6)..."
    )
    df_ae = run_cv_autoencoder(X_train, age_strata)

    cv_results = pd.concat([df_km, df_gmm, df_ae], ignore_index=True)
    cv_results.to_csv(TUNING_DIR / "cv_fold_results.csv", index=False)

    # Aggregate across folds (and seeds for Autoencoder)
    summary = (
        cv_results.groupby(
            ["model_family", "hyperparams", "k", "bottleneck_dim"], dropna=False
        )
        .agg(
            mean_val_silhouette=("val_silhouette", "mean"),
            sd_val_silhouette=("val_silhouette", "std"),
            mean_val_db=("val_davies_bouldin", "mean"),
            sd_val_db=("val_davies_bouldin", "std"),
            mean_val_bic=("val_bic", "mean"),
            mean_recon_mse=("train_recon_mse", "mean"),
        )
        .reset_index()
        .sort_values(["model_family", "k", "bottleneck_dim"])
    )
    summary.to_csv(TUNING_DIR / "cv_summary.csv", index=False)

    print("\n=== 5-Fold Cross-Validation Summary (Training Set) ===")
    print(summary.to_string(index=False))

    # Select best k per model family (highest mean validation Silhouette; BIC tie-breaker for GMM)
    best_configs = {}
    for family in ["k-Means", "GMM", "Autoencoder"]:
        sub = summary[summary["model_family"] == family].copy()
        best_row = sub.sort_values(
            by=["mean_val_silhouette", "mean_val_db"], ascending=[False, True]
        ).iloc[0]
        best_configs[family] = {
            "k": int(best_row["k"]),
            "bottleneck_dim": None
            if pd.isna(best_row["bottleneck_dim"])
            else int(best_row["bottleneck_dim"]),
            "mean_val_silhouette": round(float(best_row["mean_val_silhouette"]), 4),
            "mean_val_db": round(float(best_row["mean_val_db"]), 4),
            "mean_val_bic": None
            if pd.isna(best_row["mean_val_bic"])
            else round(float(best_row["mean_val_bic"]), 2),
        }

    # Refit selected configurations on the full training set (n = 2,460) for downstream stability & test
    models_dir = TUNING_DIR / "fitted_train_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Refit k-Means
    k_km = best_configs["k-Means"]["k"]
    km_full = KMeans(
        n_clusters=k_km, init="k-means++", n_init=50, random_state=BASE_SEED
    ).fit(X_train)
    joblib.dump(km_full, models_dir / "kmeans_full_train.joblib")

    # 2. Refit GMM
    k_gmm = best_configs["GMM"]["k"]
    gmm_full = GaussianMixture(
        n_components=k_gmm,
        covariance_type="full",
        n_init=20,
        reg_covar=1e-5,
        random_state=BASE_SEED,
    ).fit(X_train)
    joblib.dump(gmm_full, models_dir / "gmm_full_train.joblib")

    # 3. Refit Autoencoder (save all 5 seeds on full training set for stability & seed variance)
    k_ae = best_configs["Autoencoder"]["k"]
    b_dim = best_configs["Autoencoder"]["bottleneck_dim"]
    for seed in AE_SEEDS:
        ae_full, _ = train_autoencoder(X_train, bottleneck_dim=b_dim, seed=seed)
        Z_full = encode_latent(ae_full, X_train)
        km_ae_full = KMeans(
            n_clusters=k_ae, init="k-means++", n_init=50, random_state=seed
        ).fit(Z_full)
        torch.save(ae_full.state_dict(), models_dir / f"ae_dim{b_dim}_seed{seed}.pt")
        joblib.dump(km_ae_full, models_dir / f"ae_kmeans_dim{b_dim}_seed{seed}.joblib")

    with open(TUNING_DIR / "selected_models.json", "w", encoding="utf-8") as f:
        json.dump(best_configs, f, indent=2)

    print("\n=== Selected Model Configurations (Refitted on Full Train) ===")
    print(json.dumps(best_configs, indent=2))
    print(f"\nSaved tuning tables and fitted training models to {TUNING_DIR}/")


if __name__ == "__main__":
    main()
