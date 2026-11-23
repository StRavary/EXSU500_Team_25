# Model Card, Governance, and Reproducibility Guide

**Repository:** [https://github.com/StRavary/EXSU500_Team_25](https://github.com/StRavary/EXSU500_Team_25)  
**Course:** EXSU 500: Fundamentals of Artificial Intelligence in Medicine (Fall 2026)  
**Institution:** Department of Surgery, McGill University | Mila (Quebec AI Institute)

---

## 1. Model Card Summary

* **Intended Use:** Population-level epidemiological phenotyping and fairness auditing of unsupervised cardiometabolic risk stratification in U.S. adults.
* **Out-of-Scope / Non-Clinical Use:** Not a cleared medical device or clinical decision-support tool; must not be used for individual patient diagnosis, triage, or treatment selection.
* **Training & Evaluation Cohorts:** NHANES 2017–March 2020 adult fasting subsample ($N = 3,515$; $n = 2,460$ train / $n = 1,055$ held-out test).
* **Primary Evaluation Metrics:** Out-of-fold and held-out Silhouette Coefficient, Davies–Bouldin Index, Bayesian Information Criterion (BIC), 100-bootstrap Hungarian-matched Jaccard stability, permutation null $p$-values, Cramér's $V$, and FDR-adjusted disparity tests.
* **Key Limitations:** Cross-sectional observational design precludes causal inference or longitudinal outcome tracking; complete-case morning fasting selection restricts generalizability to non-fasting clinical encounters.

---

## 2. CRediT Author Contribution Statement

* **Krisen Jean Julien (Data Lead):** Conceptualization, Data Curation, Software (`src/data_curation_pipeline.R`), Validation (`metadata_data_curation_manifest.txt`), Writing – Original Draft (Data Engineering Summary).
* **Steven Ravary (Modelling Lead):** Methodology (Leakage-Free Partitioning, 5-Fold CV, Stability & Disparity Protocol), Software (`src/project_cluster/*.py`), Formal Analysis (k-Means, GMM, Autoencoder Benchmarking), Writing – Review & Editing.
* **Interface Lead:** Software (Interactive Visualization Prototype), Visualization (Phenotype & Disparity Plots), Deployment Planning.
* **Writing Lead:** Project Administration, Investigation (Literature Synthesis), Writing – Original Draft & IEEE Manuscript Assembly.

---

## 3. Generative AI Disclosure Statement

In compliance with McGill University and EXSU 500 academic integrity policies, generative AI tools (Google Gemini) were used during software development and manuscript preparation for code scaffolding, debugging environment and Git workflows, mapping CDC NHANES variable schemas, and refining technical documentation. All data curation scripts, statistical formulas, neural network architectures, cross-validation runs, and empirical outputs were independently executed, verified, and audited locally by the authors, who retain full responsibility for all scientific claims and code in this repository.

---

## 4. License

This codebase is distributed under the **MIT License** (see `LICENSE`). Underlying NHANES survey datasets are public-domain works of the U.S. Federal Government (CDC/NCHS).

---

## 5. Reproducing the End-to-End Pipeline

### Step 1: Clone the Repository
```bash
git clone [https://github.com/StRavary/EXSU500_Team_25.git](https://github.com/StRavary/EXSU500_Team_25.git)
cd EXSU500_Team_25
```

### Step 2: Data Ingestion and Curation (R Environment)
Download the NHANES 2017–March 2020 components via `nhanesA`, compute 2021 CKD-EPI race-free eGFR and multi-reading blood pressure averages, and place the resulting matrix inside `data/`:
```bash
Rscript src/data_curation_pipeline.R
mkdir -p data && mv nhanes_phenotype_analytical_matrix.csv nhanes_phenotype_analytical_matrix.rds data/
```

### Step 3: Python Environment Setup
Using [`uv`](https://docs.astral.sh/uv/) (recommended) or standard `venv` + `pip`:
```bash
# Option A: uv
uv sync

# Option B: standard venv + pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 4: Execute the Modeling & Evaluation Pipeline
Run the four modular Python stages sequentially from the repository root:
```bash
# 1. Filter N=3,515 cohort, fetch CDC survey design weights, and create stratified 70/30 splits
uv run python src/project_cluster/01_prepare_cohort.py

# 2. Run 5-fold cross-validation across k = 2..6 for k-Means, GMM, and Autoencoder
uv run python src/project_cluster/02_train_and_tune.py

# 3. Run 100-bootstrap Jaccard stability and 100-permutation null baseline tests
uv run python src/project_cluster/03_stability_and_null.py

# 4. Evaluate locked models on held-out test set (n=1,055) and execute FDR-corrected disparity tests
uv run python src/project_cluster/04_evaluate_test_and_disparities.py
```
*(If using a standard virtual environment instead of `uv`, omit `uv run` and invoke `python src/project_cluster/...` directly.)*

# Run the entire 4-stage pipeline sequentially
uv run cardiometabolic-pheno all

# Launch the interactive Streamlit phenotyping & fairness dashboard
uv run cardiometabolic-pheno ui