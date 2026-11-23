"""
04_evaluate_test_and_disparities.py
Final locked evaluation on the untouched held-out test set (n = 1,055):
  1. Held-out geometric/probabilistic metrics & inter-model agreement (ARI)
  2. Clinical biomarker profiling (6 clustering features + 4 held-out reference markers)
  3. Sociodemographic disparity hypothesis tests (Chi-square, Cramer's V, ANOVA, Kruskal-Wallis)
     with Benjamini-Hochberg FDR correction
  4. NHANES complex survey design sensitivity analysis (WTSAFPRP, SDMVSTRA, SDMVPSU)
"""

from importlib.util import module_from_spec, spec_from_file_location
from itertools import combinations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
import torch

# Load Autoencoder architecture and latent encoder from 02_train_and_tune.py
_SPEC_02 = spec_from_file_location(
    "mod02", Path("src/project_cluster/02_train_and_tune.py")
)
_MOD_02 = module_from_spec(_SPEC_02)
assert _SPEC_02 is not None and _SPEC_02.loader is not None
_SPEC_02.loader.exec_module(_MOD_02)

UndercompleteAutoencoder = _MOD_02.UndercompleteAutoencoder
encode_latent = _MOD_02.encode_latent

# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------
PREPARED_DIR = Path("artifacts") / "prepared"
TUNING_DIR = Path("artifacts") / "tuning"
EVAL_DIR = Path("artifacts") / "final_evaluation"

BIOMARKERS = [
    "HbA1c_Percent",
    "Triglycerides",
    "Total_Cholesterol",
    "HDL_Cholesterol",
    "BMI",
    "Systolic_BP",
]

SECONDARY_CLINICAL = [
    "eGFR",
    "Inflammation_CRP",
    "Diastolic_BP",
    "LDL_Cholesterol",
]

BASE_SEED = 42


# ------------------------------------------------------------------------------
# Statistical Helpers
# ------------------------------------------------------------------------------
def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Compute Benjamini-Hochberg FDR-adjusted q-values."""
    p_arr = np.asarray(p_values, dtype=np.float64)
    n = len(p_arr)
    order = np.argsort(p_arr)
    ranked_p = p_arr[order]
    q_ranked = np.minimum.accumulate((ranked_p * n / np.arange(1, n + 1))[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q_out = np.empty(n, dtype=np.float64)
    q_out[order] = q_ranked
    return q_out.tolist()


def cramers_v(contingency_table: pd.DataFrame) -> float:
    """Compute bias-corrected Cramer's V for categorical association effect size."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.to_numpy().sum()
    r, k = contingency_table.shape
    if n <= 1:
        return 0.0
    phi2 = max(0.0, (chi2 / n) - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def survey_weighted_proportions(
    df: pd.DataFrame,
    cluster_col: str,
    weight_col: str = "WTSAFPRP",
    strata_col: str = "SDMVSTRA",
    psu_col: str = "SDMVPSU",
) -> pd.DataFrame:
    """
    Compute NHANES fasting-weight (WTSAFPRP) population prevalence per cluster
    with Taylor-series linearized standard errors accounting for SDMVSTRA and SDMVPSU.
    """
    sub = df.dropna(subset=[cluster_col, weight_col, strata_col, psu_col]).copy()
    total_w = sub[weight_col].sum()
    clusters = sorted(sub[cluster_col].unique())
    rows = []

    for c in clusters:
        y = (sub[cluster_col] == c).astype(float).to_numpy()
        w = sub[weight_col].to_numpy(dtype=np.float64)
        p_hat = float(np.sum(w * y) / total_w)

        # Linearized residual z_i = w_i * (y_i - p_hat) / total_w
        sub["_z"] = w * (y - p_hat) / total_w

        # Stratified PSU variance estimator: sum_h (n_h / (n_h - 1)) * sum_i (z_hi - z_h_bar)^2
        var_est = 0.0
        for _, stratum_df in sub.groupby(strata_col):
            psu_totals = stratum_df.groupby(psu_col)["_z"].sum().to_numpy()
            n_h = len(psu_totals)
            if n_h > 1:
                z_bar = np.mean(psu_totals)
                var_est += (n_h / (n_h - 1.0)) * np.sum((psu_totals - z_bar) ** 2)

        se_hat = float(np.sqrt(max(var_est, 0.0)))
        unweighted_n = int((sub[cluster_col] == c).sum())
        unweighted_pct = float(unweighted_n / len(sub) * 100.0)

        rows.append(
            {
                "cluster": int(c),
                "unweighted_n": unweighted_n,
                "unweighted_pct": round(unweighted_pct, 2),
                "weighted_pct": round(p_hat * 100.0, 2),
                "weighted_se_pct": round(se_hat * 100.0, 2),
                "weighted_ci95_low": round(
                    max(0.0, (p_hat - 1.96 * se_hat) * 100.0), 2
                ),
                "weighted_ci95_high": round(
                    min(100.0, (p_hat + 1.96 * se_hat) * 100.0), 2
                ),
            }
        )

    return pd.DataFrame(rows)


def survey_weighted_group_disparity(
    df: pd.DataFrame, cluster_col: str, group_col: str, weight_col: str = "WTSAFPRP"
) -> pd.DataFrame:
    """Compute weighted cluster prevalence across sociodemographic strata."""
    sub = df.dropna(subset=[cluster_col, group_col, weight_col]).copy()
    records = []
    for grp, gdf in sub.groupby(group_col):
        tot_w = gdf[weight_col].sum()
        row = {
            "sociodemographic_var": group_col,
            "stratum": str(grp),
            "unweighted_n": len(gdf),
        }
        for c in sorted(sub[cluster_col].unique()):
            unw_pct = (gdf[cluster_col] == c).mean() * 100.0
            w_pct = (gdf.loc[gdf[cluster_col] == c, weight_col].sum() / tot_w) * 100.0
            row[f"cluster_{int(c)}_unweighted_pct"] = round(float(unw_pct), 2)
            row[f"cluster_{int(c)}_weighted_pct"] = round(float(w_pct), 2)
        records.append(row)
    return pd.DataFrame(records)


# ------------------------------------------------------------------------------
# Main Held-Out Evaluation Pipeline
# ------------------------------------------------------------------------------
def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load frozen training and held-out test datasets
    train_scaled = pd.read_parquet(PREPARED_DIR / "train_scaled.parquet")
    test_scaled = pd.read_parquet(PREPARED_DIR / "test_scaled.parquet")
    test_raw = pd.read_parquet(PREPARED_DIR / "test_raw.parquet")

    X_train = train_scaled[BIOMARKERS].to_numpy(dtype=np.float64)
    X_test = test_scaled[BIOMARKERS].to_numpy(dtype=np.float64)

    with open(TUNING_DIR / "selected_models.json", "r", encoding="utf-8") as f:
        selected = json.load(f)

    models_dir = TUNING_DIR / "fitted_train_models"

    # 1. Predict with locked k-Means
    km_model = joblib.load(models_dir / "kmeans_full_train.joblib")
    labels_km = km_model.predict(X_test)

    # 2. Predict with locked GMM (CV Silhouette optimum k=2)
    gmm_model = joblib.load(models_dir / "gmm_full_train.joblib")
    labels_gmm = gmm_model.predict(X_test)
    probs_gmm = gmm_model.predict_proba(X_test)

    # 2b. Secondary sensitivity check: GMM at BIC optimum (k=3) fitted strictly on train
    gmm_k3 = GaussianMixture(
        n_components=3,
        covariance_type="full",
        n_init=20,
        reg_covar=1e-5,
        random_state=BASE_SEED,
    ).fit(X_train)
    labels_gmm_k3 = gmm_k3.predict(X_test)

    # 3. Predict with locked Autoencoder (seed 42 primary, plus across 5 seeds)
    b_dim = selected["Autoencoder"]["bottleneck_dim"]
    ae_net = UndercompleteAutoencoder(input_dim=X_test.shape[1], bottleneck_dim=b_dim)
    ae_net.load_state_dict(
        torch.load(
            models_dir / f"ae_dim{b_dim}_seed{BASE_SEED}.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    ae_net.eval()
    Z_test = encode_latent(ae_net, X_test)
    km_ae = joblib.load(models_dir / f"ae_kmeans_dim{b_dim}_seed{BASE_SEED}.joblib")
    labels_ae = km_ae.predict(Z_test)

    # Align cluster IDs across models so Cluster 1 consistently represents the higher-HbA1c risk group
    def orient_binary_labels(labels: np.ndarray, hba1c_vals: np.ndarray) -> np.ndarray:
        if len(np.unique(labels)) == 2:
            if hba1c_vals[labels == 0].mean() > hba1c_vals[labels == 1].mean():
                return 1 - labels
        return labels

    hba1c_test = test_raw["HbA1c_Percent"].to_numpy()
    labels_km = orient_binary_labels(labels_km, hba1c_test)
    labels_gmm = orient_binary_labels(labels_gmm, hba1c_test)
    labels_ae = orient_binary_labels(labels_ae, hba1c_test)

    # Order GMM k=3 labels by ascending mean HbA1c (0 = Low Risk, 1 = Intermediate/Lipid, 2 = High Glycemic)
    order_k3 = np.argsort([hba1c_test[labels_gmm_k3 == c].mean() for c in range(3)])
    remap_k3 = {old: new for new, old in enumerate(order_k3)}
    labels_gmm_k3 = np.vectorize(remap_k3.get)(labels_gmm_k3)

    # Save held-out test assignments
    assignments_df = test_raw.copy()
    assignments_df["cluster_kmeans"] = labels_km
    assignments_df["cluster_gmm"] = labels_gmm
    assignments_df["gmm_prob_high_risk"] = (
        probs_gmm[:, 1]
        if hba1c_test[gmm_model.predict(X_test) == 1].mean()
        >= hba1c_test[gmm_model.predict(X_test) == 0].mean()
        else probs_gmm[:, 0]
    )
    assignments_df["cluster_gmm_k3_bic"] = labels_gmm_k3
    assignments_df["cluster_ae"] = labels_ae
    assignments_df.to_parquet(
        EVAL_DIR / "test_set_cluster_assignments.parquet", index=False
    )

    # --------------------------------------------------------------------------
    # Table 1: Held-Out Test Set Internal Performance Metrics (n = 1,055)
    # --------------------------------------------------------------------------
    perf_rows = [
        {
            "model": f"k-Means (k={selected['k-Means']['k']})",
            "test_sil_ambient_6d": round(float(silhouette_score(X_test, labels_km)), 4),
            "test_db_ambient_6d": round(
                float(davies_bouldin_score(X_test, labels_km)), 4
            ),
            "test_sil_latent": np.nan,
            "test_bic": np.nan,
            "cluster_counts": {int(k): int(v) for k, v in zip(*np.unique(labels_km, return_counts=True))},
        },
        {
            "model": f"GMM (k={selected['GMM']['k']}, Silhouette opt)",
            "test_sil_ambient_6d": round(
                float(silhouette_score(X_test, labels_gmm)), 4
            ),
            "test_db_ambient_6d": round(
                float(davies_bouldin_score(X_test, labels_gmm)), 4
            ),
            "test_sil_latent": np.nan,
            "test_bic": round(float(gmm_model.bic(X_test)), 2),
            "cluster_counts": {int(k): int(v) for k, v in zip(*np.unique(labels_gmm, return_counts=True))},
        },
        {
            "model": "GMM (k=3, BIC opt sensitivity)",
            "test_sil_ambient_6d": round(
                float(silhouette_score(X_test, labels_gmm_k3)), 4
            ),
            "test_db_ambient_6d": round(
                float(davies_bouldin_score(X_test, labels_gmm_k3)), 4
            ),
            "test_sil_latent": np.nan,
            "test_bic": round(float(gmm_k3.bic(X_test)), 2),
            "cluster_counts": {int(k): int(v) for k, v in zip(*np.unique(labels_gmm_k3, return_counts=True))},
        },
        {
            "model": f"Autoencoder (k={selected['Autoencoder']['k']}, dim={b_dim})",
            "test_sil_ambient_6d": round(float(silhouette_score(X_test, labels_ae)), 4),
            "test_db_ambient_6d": round(
                float(davies_bouldin_score(X_test, labels_ae)), 4
            ),
            "test_sil_latent": round(float(silhouette_score(Z_test, labels_ae)), 4),
            "test_bic": np.nan,
            "cluster_counts": {int(k): int(v) for k, v in zip(*np.unique(labels_ae, return_counts=True))},
        },
    ]
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(EVAL_DIR / "test_internal_metrics.csv", index=False)

    # Inter-model Adjusted Rand Index on test set
    model_preds = {"k-Means": labels_km, "GMM": labels_gmm, "Autoencoder": labels_ae}
    ari_rows = [
        {
            "model_a": m1,
            "model_b": m2,
            "test_ari": round(
                float(adjusted_rand_score(model_preds[m1], model_preds[m2])), 4
            ),
        }
        for m1, m2 in combinations(model_preds.keys(), 2)
    ]
    ari_df = pd.DataFrame(ari_rows)
    ari_df.to_csv(EVAL_DIR / "test_intermodel_ari.csv", index=False)

    print("=== Table 1: Held-Out Test Set Performance (n = 1,055) ===")
    print(perf_df.to_string(index=False))
    print("\n=== Inter-Model Agreement (Adjusted Rand Index on Test Set) ===")
    print(ari_df.to_string(index=False))

    # --------------------------------------------------------------------------
    # Table 2: Clinical Biomarker Profiles (Clustering + Held-Out Reference)
    # --------------------------------------------------------------------------
    all_clinical_vars = BIOMARKERS + SECONDARY_CLINICAL + ["Age_Years"]
    profile_rows = []

    for model_name, col_name in [
        ("k-Means (k=2)", "cluster_kmeans"),
        ("GMM (k=2)", "cluster_gmm"),
        ("GMM (k=3, BIC)", "cluster_gmm_k3_bic"),
        ("Autoencoder (k=2)", "cluster_ae"),
    ]:
        clusters = sorted(assignments_df[col_name].unique())
        for var in all_clinical_vars:
            groups = [
                assignments_df.loc[assignments_df[col_name] == c, var].dropna()
                for c in clusters
            ]
            kw_stat, kw_p = stats.kruskal(*groups)
            row = {
                "model": model_name,
                "biomarker": var,
                "role": "Clustering" if var in BIOMARKERS else "Held-Out Reference",
                "kw_stat": round(float(kw_stat), 3),
                "kw_p_value": float(kw_p),
            }
            for c, g in zip(clusters, groups):
                row[f"C{c}_mean_sd"] = f"{g.mean():.2f} +/- {g.std():.2f}"
                row[f"C{c}_median_iqr"] = (
                    f"{g.median():.2f} ({g.quantile(0.25):.2f}-{g.quantile(0.75):.2f})"
                )
            profile_rows.append(row)

    profiles_df = pd.DataFrame(profile_rows)
    profiles_df.to_csv(EVAL_DIR / "test_clinical_biomarker_profiles.csv", index=False)

    print("\n=== Table 2A: Held-Out Clinical Profiles for k-Means (k=2) ===")
    print(
        profiles_df[profiles_df["model"] == "k-Means (k=2)"][
            ["biomarker", "role", "C0_mean_sd", "C1_mean_sd", "kw_stat", "kw_p_value"]
        ].to_string(index=False)
    )

    print("\n=== Table 2B: Held-Out Clinical Profiles for GMM (k=2) ===")
    print(
        profiles_df[profiles_df["model"] == "GMM (k=2)"][
            ["biomarker", "role", "C0_mean_sd", "C1_mean_sd", "kw_stat", "kw_p_value"]
        ].to_string(index=False)
    )

    # --------------------------------------------------------------------------
    # Table 3: Sociodemographic Disparity Hypothesis Tests + BH-FDR Correction
    # --------------------------------------------------------------------------
    disparity_rows = []
    for model_name, col_name in [
        ("k-Means (k=2)", "cluster_kmeans"),
        ("GMM (k=2)", "cluster_gmm"),
        ("GMM (k=3, BIC)", "cluster_gmm_k3_bic"),
        ("Autoencoder (k=2)", "cluster_ae"),
    ]:
        # Categorical variables: Pearson Chi-square + Cramer's V
        for cat_var in ["Race_Ethnicity", "Education_Level", "Gender"]:
            ct = pd.crosstab(assignments_df[cat_var], assignments_df[col_name])
            chi2, p_val, dof, _ = stats.chi2_contingency(ct)
            v_eff = cramers_v(ct)
            disparity_rows.append(
                {
                    "model": model_name,
                    "variable": cat_var,
                    "test_type": "Chi-Square",
                    "statistic": round(float(chi2), 4),
                    "df": int(dof),
                    "p_value": float(p_val),
                    "effect_size": f"Cramer's V = {v_eff:.4f}",
                }
            )

        # Continuous variables: Kruskal-Wallis + One-way ANOVA
        for cont_var in ["Income_Poverty_Ratio", "Age_Years"]:
            groups = [
                assignments_df.loc[assignments_df[col_name] == c, cont_var].dropna()
                for c in sorted(assignments_df[col_name].unique())
            ]
            kw_h, kw_p = stats.kruskal(*groups)
            f_stat, f_p = stats.f_oneway(*groups)
            disparity_rows.append(
                {
                    "model": model_name,
                    "variable": cont_var,
                    "test_type": "Kruskal-Wallis",
                    "statistic": round(float(kw_h), 4),
                    "df": len(groups) - 1,
                    "p_value": float(kw_p),
                    "effect_size": f"ANOVA F = {f_stat:.4f} (p={f_p:.4g})",
                }
            )

    disp_df = pd.DataFrame(disparity_rows)
    # Apply Benjamini-Hochberg FDR within each model family across the 5 sociodemographic tests
    disp_df["fdr_q_value"] = np.nan
    for model_name in disp_df["model"].unique():
        mask = disp_df["model"] == model_name
        disp_df.loc[mask, "fdr_q_value"] = benjamini_hochberg(
            disp_df.loc[mask, "p_value"].tolist()
        )

    disp_df["significant_fdr_05"] = disp_df["fdr_q_value"] < 0.05
    disp_df.to_csv(EVAL_DIR / "test_sociodemographic_disparities.csv", index=False)

    print(
        "\n=== Table 3: Held-Out Sociodemographic Disparity Tests (BH-FDR Corrected) ==="
    )
    print(disp_df.to_string(index=False))

    # --------------------------------------------------------------------------
    # Table 4: NHANES Survey-Weighted Sensitivity Analysis (WTSAFPRP / SDMVSTRA / SDMVPSU)
    # --------------------------------------------------------------------------
    weighted_prev_list = []
    weighted_strata_list = []
    for model_name, col_name in [
        ("k-Means (k=2)", "cluster_kmeans"),
        ("GMM (k=2)", "cluster_gmm"),
    ]:
        prev_df = survey_weighted_proportions(assignments_df, col_name)
        prev_df.insert(0, "model", model_name)
        weighted_prev_list.append(prev_df)

        for cat_var in ["Race_Ethnicity", "Education_Level"]:
            st_df = survey_weighted_group_disparity(assignments_df, col_name, cat_var)
            st_df.insert(0, "model", model_name)
            weighted_strata_list.append(st_df)

    w_prev_df = pd.concat(weighted_prev_list, ignore_index=True)
    w_strata_df = pd.concat(weighted_strata_list, ignore_index=True)

    w_prev_df.to_csv(EVAL_DIR / "test_survey_weighted_prevalence.csv", index=False)
    w_strata_df.to_csv(EVAL_DIR / "test_survey_weighted_disparities.csv", index=False)

    print("\n=== Table 4: NHANES Survey-Weighted Cluster Prevalence (Test Set) ===")
    print(w_prev_df.to_string(index=False))

    # Save executive JSON summary
    summary_payload = {
        "n_test": int(len(assignments_df)),
        "selected_models": selected,
        "test_internal_metrics": perf_rows,
        "test_intermodel_ari": ari_rows,
    }
    with open(EVAL_DIR / "final_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, default=str)

    print(f"\nAll final held-out evaluation tables saved to {EVAL_DIR}/")


if __name__ == "__main__":
    main()
