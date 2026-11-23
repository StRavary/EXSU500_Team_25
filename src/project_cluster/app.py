"""
app.py
Interactive Streamlit Dashboard for Data-Driven Cardiometabolic Phenotyping
and Sociodemographic Disparity Auditing (NHANES 2017-March 2020).
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Cardiometabolic Phenotyping Explorer | EXSU 500",
    page_icon="🫀",
    layout="wide",
)

PREPARED_DIR = Path("artifacts") / "prepared"
TUNING_DIR = Path("artifacts") / "tuning"
STABILITY_DIR = Path("artifacts") / "stability"
EVAL_DIR = Path("artifacts") / "final_evaluation"

BIOMARKERS = [
    "HbA1c_Percent",
    "Triglycerides",
    "Total_Cholesterol",
    "HDL_Cholesterol",
    "BMI",
    "Systolic_BP",
]


@st.cache_resource
def load_models_and_scaler():
    scaler = joblib.load(PREPARED_DIR / "biomarker_scaler.joblib")
    km = joblib.load(TUNING_DIR / "fitted_train_models" / "kmeans_full_train.joblib")
    gmm = joblib.load(TUNING_DIR / "fitted_train_models" / "gmm_full_train.joblib")
    train_raw = pd.read_parquet(PREPARED_DIR / "train_raw.parquet")
    train_scaled = pd.read_parquet(PREPARED_DIR / "train_scaled.parquet")

    # Determine which raw cluster index corresponds to higher mean HbA1c (High-Risk C1)
    X_tr = train_scaled[BIOMARKERS].to_numpy()
    hba1c_tr = train_raw["HbA1c_Percent"].to_numpy()

    km_lbls = km.predict(X_tr)
    km_high_idx = (
        1 if hba1c_tr[km_lbls == 1].mean() >= hba1c_tr[km_lbls == 0].mean() else 0
    )

    gmm_lbls = gmm.predict(X_tr)
    gmm_high_idx = (
        1 if hba1c_tr[gmm_lbls == 1].mean() >= hba1c_tr[gmm_lbls == 0].mean() else 0
    )

    return scaler, km, gmm, km_high_idx, gmm_high_idx


@st.cache_data
def load_artifacts():
    test_df = pd.read_parquet(EVAL_DIR / "test_set_cluster_assignments.parquet")
    perf_df = pd.read_csv(EVAL_DIR / "test_internal_metrics.csv")
    disp_df = pd.read_csv(EVAL_DIR / "test_sociodemographic_disparities.csv")
    w_prev_df = pd.read_csv(EVAL_DIR / "test_survey_weighted_prevalence.csv")
    w_strata_df = pd.read_csv(EVAL_DIR / "test_survey_weighted_disparities.csv")
    stab_df = pd.read_csv(STABILITY_DIR / "stability_and_null_summary.csv")
    boot_df = pd.read_csv(STABILITY_DIR / "bootstrap_jaccard_iterations.csv")
    return test_df, perf_df, disp_df, w_prev_df, w_strata_df, stab_df, boot_df


def main():
    st.title("🫀 Data-Driven Cardiometabolic Phenotyping & Fairness Explorer")
    st.caption(
        "EXSU 500: Fundamentals of Artificial Intelligence in Medicine (McGill University) | "
        "CDC NHANES 2017–March 2020 Adult Fasting Cohort (N = 3,515)"
    )

    if not (EVAL_DIR / "test_set_cluster_assignments.parquet").exists():
        st.error(
            "Pipeline artifacts not found. Run `uv run cardiometabolic-pheno all` first to generate artifacts."
        )
        st.stop()

    scaler, km, gmm, km_high_idx, gmm_high_idx = load_models_and_scaler()
    test_df, perf_df, disp_df, w_prev_df, w_strata_df, stab_df, boot_df = (
        load_artifacts()
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔬 Interactive Phenotype Simulator",
            "🩺 Held-Out Clinical Validation (n=1,055)",
            "⚖️ Sociodemographic & Survey Fairness",
            "📊 Model Benchmarks & Null Audits",
        ]
    )

    # --------------------------------------------------------------------------
    # TAB 1: Interactive Endotype Simulator
    # --------------------------------------------------------------------------
    with tab1:
        st.warning(
            "**Research Prototype Notice:** For epidemiological and educational demonstration only. "
            "Not a medical device or clinical diagnostic tool."
        )
        col_in, col_out = st.columns([1, 1.3], gap="large")

        with col_in:
            st.subheader("Input Physiological Biomarkers")
            hba1c = st.slider("Glycated Hemoglobin — HbA1c (%)", 4.0, 14.0, 5.6, 0.1)
            tg = st.slider("Fasting Triglycerides (mg/dL)", 30, 600, 105, 5)
            tchol = st.slider("Total Cholesterol (mg/dL)", 100, 350, 185, 5)
            hdl = st.slider("HDL Cholesterol (mg/dL)", 20, 110, 52, 1)
            bmi = st.slider("Body Mass Index — BMI (kg/m²)", 16.0, 55.0, 28.5, 0.5)
            sbp = st.slider("Mean Systolic Blood Pressure (mmHg)", 85, 200, 122, 1)

        sample_raw = np.array([[hba1c, tg, tchol, hdl, bmi, sbp]], dtype=np.float64)
        sample_scaled = scaler.transform(sample_raw)

        gmm_probs = gmm.predict_proba(sample_scaled)[0]
        p_high_gmm = float(gmm_probs[gmm_high_idx])
        gmm_class = (
            "Cluster 1: High-Risk Cardiometabolic Endotype"
            if p_high_gmm >= 0.5
            else "Cluster 0: Lower-Risk Metabolic Profile"
        )

        km_pred_raw = int(km.predict(sample_scaled)[0])
        km_is_high = km_pred_raw == km_high_idx
        km_class = (
            "Cluster 1: Elevated Adiposity/Lipid Risk"
            if km_is_high
            else "Cluster 0: Lower-Risk Profile"
        )

        with col_out:
            st.subheader("Model Endotype Assignment")
            m1, m2 = st.columns(2)
            m1.metric(
                "GMM (k=2) High-Risk Probability",
                f"{p_high_gmm * 100:.1f}%",
                delta="High-Risk Endotype"
                if p_high_gmm >= 0.5
                else "Lower-Risk Endotype",
                delta_color="inverse" if p_high_gmm >= 0.5 else "normal",
            )
            m2.metric(
                "k-Means (k=2) Partition",
                "Cluster 1 (High)" if km_is_high else "Cluster 0 (Low)",
            )

            st.info(
                f"**GMM Classification:** {gmm_class}\n\n**k-Means Classification:** {km_class}"
            )

            # Standardized Z-Score Radar Profile
            z_vals = sample_scaled[0].tolist()
            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=z_vals + [z_vals[0]],
                    theta=BIOMARKERS + [BIOMARKERS[0]],
                    fill="toself",
                    name="Simulated Profile (Z-Score)",
                )
            )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[0] * (len(BIOMARKERS) + 1),
                    theta=BIOMARKERS + [BIOMARKERS[0]],
                    name="Training Cohort Mean (Z=0)",
                    line=dict(dash="dash", color="gray"),
                )
            )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[-2.5, 4.0])),
                title="Standardized Biomarker Profile Relative to NHANES Training Mean",
                height=380,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 2: Held-Out Clinical Validation (n = 1,055)
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("Held-Out Test Cohort Stratification (n = 1,055)")
        model_choice = st.radio(
            "Select Clustering Model:",
            ["GMM (k=2)", "k-Means (k=2)"],
            horizontal=True,
        )
        cluster_col = "cluster_gmm" if model_choice == "GMM (k=2)" else "cluster_kmeans"

        plot_df = test_df.copy()
        plot_df["Phenotype"] = plot_df[cluster_col].map(
            {0: "Cluster 0 (Lower Risk)", 1: "Cluster 1 (High Cardiometabolic Risk)"}
        )

        c1, c2 = st.columns(2)
        with c1:
            fig_scatter = px.scatter(
                plot_df,
                x="HbA1c_Percent",
                y="Triglycerides",
                color="Phenotype",
                opacity=0.7,
                title=f"{model_choice}: Glycemic vs. Triglyceride Separation (Held-Out Test)",
                labels={
                    "HbA1c_Percent": "HbA1c (%)",
                    "Triglycerides": "Triglycerides (mg/dL)",
                },
                color_discrete_sequence=["#2b5c8f", "#d95f02"],
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with c2:
            ref_marker = st.selectbox(
                "Inspect Unseen Held-Out Clinical Reference Marker:",
                [
                    "Inflammation_CRP",
                    "eGFR",
                    "Diastolic_BP",
                    "LDL_Cholesterol",
                    "Age_Years",
                ],
            )
            fig_box = px.box(
                plot_df,
                x="Phenotype",
                y=ref_marker,
                color="Phenotype",
                points="outliers",
                title=f"External Validation on Unclustered Marker: {ref_marker}",
                color_discrete_sequence=["#2b5c8f", "#d95f02"],
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 3: Sociodemographic & Survey Fairness
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader(
            "Sociodemographic Disparity Audits (BH-FDR Corrected) & Survey Weighting"
        )
        st.dataframe(disp_df, use_container_width=True, hide_index=True)

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.markdown("#### NHANES Survey-Weighted National Prevalence (`WTSAFPRP`)")
            st.dataframe(w_prev_df, use_container_width=True, hide_index=True)

        with col_w2:
            st.markdown("#### Subgroup High-Risk Cluster (`C1`) Prevalence")
            sel_model_w = st.selectbox(
                "Model for Subgroup Breakdown:", ["GMM (k=2)", "k-Means (k=2)"]
            )
            sel_var_w = st.selectbox(
                "Sociodemographic Dimension:", ["Education_Level", "Race_Ethnicity"]
            )
            sub_w = w_strata_df[
                (w_strata_df["model"] == sel_model_w)
                & (w_strata_df["sociodemographic_var"] == sel_var_w)
            ]
            fig_bar = px.bar(
                sub_w,
                x="stratum",
                y=["cluster_1_unweighted_pct", "cluster_1_weighted_pct"],
                barmode="group",
                title=f"{sel_model_w}: High-Risk Endotype (C1) % by {sel_var_w}",
                labels={
                    "value": "Prevalence (%)",
                    "stratum": sel_var_w,
                    "variable": "Estimate Type",
                },
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 4: Model Benchmarks & Stability Audits
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader(
            "100-Bootstrap Jaccard Stability & Permutation Null Benchmarks (n = 2,460)"
        )
        st.dataframe(stab_df, use_container_width=True, hide_index=True)

        fig_boot = px.box(
            boot_df,
            x="model_family",
            y="mean_jaccard",
            color="model_family",
            title="100-Iteration Bootstrap Cluster Stability (Hungarian-Matched Jaccard Similarity)",
            labels={
                "model_family": "Model Family",
                "mean_jaccard": "Mean Jaccard Similarity",
            },
        )
        fig_boot.add_hline(
            y=0.75,
            line_dash="dash",
            line_color="red",
            annotation_text="Hennig Stability Threshold (0.75)",
        )
        st.plotly_chart(fig_boot, use_container_width=True)

        st.subheader("Held-Out Test Set Internal Metrics (n = 1,055)")
        st.dataframe(perf_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
