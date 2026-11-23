# Data-Driven Cardiometabolic Phenotyping vs. Sociodemographic Confounding

**Repository:** [https://github.com/StRavary/EXSU500_Team_25](https://github.com/StRavary/EXSU500_Team_25)  
**Problem Class:** Unsupervised Clustering & Clinical Phenotyping  
**Dataset:** CDC NHANES (2017–March 2020 Pre-Pandemic Cycle)  
**Course:** EXSU 500: Fundamentals of Artificial Intelligence in Medicine (Fall 2026)  
**Institution:** Department of Surgery, McGill University | Mila (Quebec AI Institute)  
**Repository Access:** Graded academic repository (Read access granted to course staff `@AHooshiar` and teaching assistants)

> **Research Question:** *Do data-driven cardiometabolic phenotypes discovered in NHANES capture distinct physiological endotypes, or do they primarily reproduce demographic and socioeconomic stratification?*

---

## 1. Project Overview and Task Definition

Cardiometabolic diseases—including type 2 diabetes mellitus, dyslipidemia, and arterial hypertension—are routinely evaluated using isolated clinical cutoffs (such as single fasting glucose or systolic blood pressure thresholds). While convenient, univariate thresholds obscure multi-system physiological interactions across glycemic, lipid, adiposity, and hemodynamic pathways.

This project formulates an unsupervised machine learning pipeline to identify latent cardiometabolic risk endotypes from six continuous physiological biomarkers, and then tests whether cluster membership diverges across social determinants of health (race/ethnicity, adult educational attainment, and family income-to-poverty ratio).

### Problem Formulation
* **Task Type:** Unsupervised clustering and clinical phenotyping.
* **Input Feature Space (6 Biomarkers):** Glycated Hemoglobin (`HbA1c_Percent`, %), Serum Triglycerides (`Triglycerides`, mg/dL), Total Cholesterol (`Total_Cholesterol`, mg/dL), High-Density Lipoprotein Cholesterol (`HDL_Cholesterol`, mg/dL), Body Mass Index (`BMI`, kg/m²), and Mean Systolic Blood Pressure (`Systolic_BP`, mmHg).
* **Held-Out Clinical Reference Markers:** Estimated Glomerular Filtration Rate (`eGFR`, mL/min/1.73m²), High-Sensitivity C-Reactive Protein (`Inflammation_CRP`, mg/L), Mean Diastolic Blood Pressure (`Diastolic_BP`, mmHg), and LDL Cholesterol (`LDL_Cholesterol`, mg/dL).
* **Downstream Hypothesis Testing:** Statistical evaluation of demographic and socioeconomic divergence across phenotypes using Pearson's $\chi^2$ tests, bias-corrected Cramér's $V$, Kruskal–Wallis tests, and Benjamini–Hochberg False Discovery Rate (FDR) correction, complemented by NHANES complex survey-weighted sensitivity analyses.

---

## 2. Dataset Provenance and Analytical Domain

### Source Dataset
Data are extracted from the National Health and Nutrition Examination Survey (NHANES) 2017 to March 2020 pre-pandemic release, administered by the National Center for Health Statistics (NCHS) at the Centers for Disease Control and Prevention (CDC). The raw cycle encompasses 15,560 participants across Demographics, Examination, and Laboratory modules.

* **NHANES Source Cycle:** 2017 to March 2020 Pre-Pandemic Examination Data
* **Data Custodian:** CDC / NCHS (Hyattsville, MD, USA)
* **License & Access:** Open-access public domain data; no credentialed access or data use agreement required.

### Analytical Domain & Missingness Handling
Clustering directly across all 15,560 records introduces severe structural distortion due to NHANES survey eligibility rules:
1. **Educational Attainment (`DMDEDUC2`):** Collected strictly for adults aged $\ge 20$ years ($n = 9,232$). Missingness in younger participants reflects survey eligibility rather than nonresponse.
2. **Fasting Lipid Panel (`LBXTR`, `LBDLDL`):** Collected strictly in the morning fasting subsample ($n = 4,650$), representing planned subsample missingness by design.

To prevent synthetic cluster artifacts from whole-table imputation, the analytical cohort is restricted to adults aged $\ge 20$ years with complete observations across all six clustering biomarkers, yielding an uncompromised cohort of **$N = 3,515$** participants. Non-informative survey codes (`"Don't Know"`, `"Refused"`) in `Education_Level` are recoded to `NaN` for downstream disparity testing.

### Feature Schema and Transformation Dictionary

| Variable Name | Concept / Measurement | NHANES Source / Code | Analytical Role |
| :--- | :--- | :--- | :--- |
| **`ID`** | Participant Sequence Token | `SEQN` | Relational primary key |
| **`HbA1c_Percent`** | Glycohemoglobin (%) | `P_GHB` (`LBXGH`) | Clustering feature |
| **`Triglycerides`** | Serum Triglycerides (mg/dL) | `P_TRIGLY` (`LBXTR`) | Clustering feature |
| **`Total_Cholesterol`** | Total Serum Cholesterol (mg/dL) | `P_TCHOL` (`LBXTC`) | Clustering feature |
| **`HDL_Cholesterol`** | High-Density Lipoprotein (mg/dL) | `P_HDL` (`LBDHDD`) | Clustering feature |
| **`BMI`** | Body Mass Index (kg/m²) | `P_BMX` (`BMXBMI`) | Clustering feature |
| **`Systolic_BP`** | Mean Systolic Blood Pressure (mmHg) | Engineered from `P_BPXO` | Clustering feature |
| **`Diastolic_BP`** | Mean Diastolic Blood Pressure (mmHg) | Engineered from `P_BPXO` | Clinical validation reference |
| **`eGFR`** | Renal Function (mL/min/1.73m²) | Engineered via 2021 CKD-EPI | Clinical validation reference |
| **`Inflammation_CRP`** | High-Sensitivity CRP (mg/L) | `P_HSCRP` (`LBXHSCRP`) | Clinical validation reference |
| **`LDL_Cholesterol`** | Low-Density Lipoprotein (mg/dL) | `P_TRIGLY` (`LBDLDL`) | Clinical validation reference |
| **`Age_Years`** | Participant Age at Exam | `P_DEMO` (`RIDAGEYR`) | Stratification / Splitting |
| **`Gender`** | Categorized Biological Sex | `P_DEMO` (`RIAGENDR`) | Demographic covariate |
| **`Race_Ethnicity`** | Self-Reported Ethno-Racial Origin | `P_DEMO` (`RIDRETH3`) | Sociodemographic disparity test |
| **`Education_Level`** | Adult Attainment (5 levels) | `P_DEMO` (`DMDEDUC2`) | Sociodemographic disparity test |
| **`Income_Poverty_Ratio`** | Family Income-to-Poverty Ratio | `P_DEMO` (`INDFMPIR`) | Sociodemographic disparity test |
| **`WTSAFPRP` / `SDMVSTRA` / `SDMVPSU`** | Fasting Subsample Weight, Stratum & PSU | `P_TRIGLY` & `P_DEMO` | Survey-weighted sensitivity analysis |

* **Blood Pressure Standardization:** `Systolic_BP` and `Diastolic_BP` are computed as row-wise means across up to three consecutive oscillometric readings (`BPXOSY1–3`, `BPXODI1–3`) to reduce single-measurement noise.
* **Race-Free Renal Filtration:** `eGFR` is calculated using the 2021 CKD-EPI Creatinine equation without race coefficients.

---

## 3. Modelling Strategy and Leakage-Free Protocol

### Model Families Under Comparison
Three distinct unsupervised learning paradigms are benchmarked under identical partitioning and validation rules:
1. **Baseline Floor (`k-Means`):** Euclidean k-Means with `k-means++` initialization across 50 random restarts (`n_init=50`), establishing a spherical partitioning baseline.
2. **Classical Probabilistic Model (`GMM`):** Full-covariance Gaussian Mixture Models (`covariance_type="full"`, `n_init=20`) fitted via Expectation-Maximization to capture ellipsoidal biomarker correlations and soft posterior assignment probabilities.
3. **Deep Representation Learning (`Undercomplete Autoencoder`):** A symmetric PyTorch neural network (`6 -> 16 -> 8 -> d -> 8 -> 16 -> 6` with `LayerNorm`, `SiLU` activations, and `AdamW` weight decay) compressing the 6 biomarkers into a low-dimensional latent manifold ($d \in \{2, 3\}$), followed by latent k-Means clustering across 5 initialization seeds (`42–46`).

### Partitioning, Validation, and Leakage Controls
* **Stratified 70/30 Split:** The $N = 3,515$ cohort is partitioned into a **Training Set ($70\%$, $n = 2,460$)** and a **Held-Out Test Set ($30\%$, $n = 1,055$)** stratified across age deciles (`random_state=42`).
* **Strict Normalization Isolation:** Z-score standardization (`StandardScaler`) is fitted exclusively on the training partition ($n = 2,460$) and applied to the test partition using frozen training parameters.
* **5-Fold Cross-Validation ($k \in \{2, 3, 4, 5, 6\}$):** Cluster granularity and autoencoder bottleneck dimensions are tuned strictly within the training set using out-of-fold Silhouette Coefficient, Davies–Bouldin Index, and Bayesian Information Criterion (BIC).
* **Stability & Permutation Null Testing:** Model robustness is audited on the training set via 100 non-parametric bootstrap resamples (Hungarian-matched Jaccard similarity) and 100 column-wise feature-permutation null trials.

---

## 4. Empirical Results Summary

### A. Training-Set Tuning, Stability & Permutation Null Benchmarks (`n = 2,460`)
Across $k \in \{2, 3, 4, 5, 6\}$, `k = 2` maximizes out-of-fold geometric Silhouette separation across all three model families, while `GMM` achieves its minimum Bayesian Information Criterion at `k = 3` (`BIC = 7383.57`). Training-set robustness audits demonstrate high stability for `GMM` (`Jaccard = 0.9349`) and `k-Means` (`Jaccard = 0.8824`), whereas the `Autoencoder` suffers from severe initialization variance (`pairwise ARI = 0.0781 +/- 0.1757`, `Jaccard = 0.3955`) and fails to beat column-permuted random noise (`p = 0.9109` ambient / `p = 0.9802` latent):

| Model Family | Selected Config | 5-Fold Val Silhouette | Bootstrap Jaccard (100x) | Permutation Null $p$-value |
| :--- | :--- | :---: | :---: | :---: |
| **k-Means (Baseline)** | `k = 2, n_init = 50` | $0.1764 \pm 0.0167$ | **$0.8824 \pm 0.1357$** | **$0.0396$** |
| **GMM (Classical ML)** | `k = 2, cov = full` | $0.2633 \pm 0.0038$ | **$0.9349 \pm 0.0384$** | $0.6436$ |
| **Autoencoder (Deep)** | `k = 2, bottleneck = 2D` | $0.6131 \pm 0.1528$ *(latent)* | $0.3955 \pm 0.1169$ | $0.9109$ *(6D)* / $0.9802$ *(2D)* |

### B. Held-Out Test Set Performance & External Clinical Validity (`n = 1,055`)
When applied once to the untouched held-out test set ($n = 1,055$), both `k-Means (k = 2)` and `GMM (k = 2)` stratify patients into clinically distinct cardiometabolic endotypes that generalize to **unseen clinical validation biomarkers** (`eGFR`, `Inflammation_CRP`, and `Diastolic_BP`):

| Model | Test Ambient 6D Silhouette ($\uparrow$) | Test Davies–Bouldin ($\downarrow$) | Test Cluster Counts | High-Risk Phenotype (`C1`) Profile vs. Low-Risk (`C0`) |
| :--- | :---: | :---: | :---: | :--- |
| **k-Means (`k = 2`)** | 0.1770 | **1.9960** | `C0: 566` / `C1: 489` | Higher HbA1c ($6.40\%$ vs. $5.45\%$), TG ($140.9$ vs. $80.8$), BMI ($34.8$ vs. $26.0$), SBP ($129.5$ vs. $119.8$), CRP ($6.02$ vs. $2.74\text{ mg/L}$, $p = 1.1 \times 10^{-43}$), lower HDL ($43.7$ vs. $60.6$) and eGFR ($98.5$ vs. $104.0$, $p = 0.0012$). |
| **GMM (`k = 2`)** | **0.2566** | 2.1993 | `C0: 832` / `C1: 223` | Overt dysglycemia (`HbA1c` $7.22\%$ vs. $5.54\%$, $p = 3.6 \times 10^{-44}$), hypertriglyceridemia ($183.1$ vs. $88.7$, $p = 5.3 \times 10^{-44}$), obesity (`BMI` $35.0$ vs. $28.8$), elevated CRP ($6.80$ vs. $3.58$, $p = 2.2 \times 10^{-13}$), lower eGFR ($97.6$ vs. $102.5$, $p = 0.041$). |
| **GMM (`k = 3`, BIC opt)** | 0.1155 | 2.2480 | `526 / 428 / 101` | Achieves lowest test BIC (`15260.00` vs. `15435.61` for `k = 2`), isolating a severe cardiometabolic tail (`n = 101`). |
| **Autoencoder (`k = 2, 2D`)** | 0.1072 | 2.6810 | `C0: 579` / `C1: 476` | Near-zero agreement with k-Means (`ARI = 0.0109`) and GMM (`ARI = -0.0036`). |

### C. Sociodemographic Disparity Audits (BH-FDR Corrected) & Survey Weighting (`n = 1,055`)
* **Clinical Endotypes vs. Confounding:** Physiological effect sizes (`BMI` $H = 401.5$, `HDL` $H = 384.0$, `HbA1c` $H = 226.3$, `Triglycerides` $H = 208.9$) are an order of magnitude larger than sociodemographic associations, confirming that clusters capture genuine biological pathology rather than merely proxying demographics.
* **Statistically Significant Social Gradients (BH-FDR $q < 0.05$):** Despite modest effect sizes, both `k-Means (k = 2)` and `GMM (k = 2)` exhibit statistically significant disparities after Benjamini–Hochberg FDR correction across `Education_Level` (`Cramér's V = 0.1546`, $q = 3.6 \times 10^{-5}$ for k-Means; `V = 0.1716`, $q = 2.3 \times 10^{-6}$ for GMM), `Income_Poverty_Ratio` ($q = 0.0012$ / $q = 0.0043$), and `Race_Ethnicity` (`V = 0.0987`, $q = 0.0115$ / `V = 0.0907`, $q = 0.0222$).
* **Nationally Representative Weighted Prevalence (`WTSAFPRP` / `SDMVSTRA` / `SDMVPSU`):** Incorporating NHANES fasting subsample design weights yields national high-risk endotype (`C1`) prevalence estimates of **$41.27\% \pm 2.03\%$** ($95\%$ CI: $37.29\%–45.26\%$) for `k-Means (k = 2)` and **$16.53\% \pm 1.76\%$** ($95\%$ CI: $13.08\%–19.99\%$) for `GMM (k = 2)`.

---

## 5. Repository Structure

```text
EXSU500_Team_25/
├── README.md                                   <- Project overview, methodology, and empirical results summary
├── CRediT_reproducibility.md                   <- Model card, CRediT author roles, AI disclosure, license & reproduction steps
├── LICENSE                                     <- MIT License
├── pyproject.toml                              <- uv / Python project metadata and dependency specification
├── requirements.txt                            <- Pinned Python dependencies for reproducible execution
├── metadata_data_curation_manifest.txt         <- Automated R curation log, missingness & range audit
├── data/                                       <- Local data directory (git-ignored; holds .csv, .rds, .parquet)
└── src/
    ├── data_curation_pipeline.R                <- R script for CDC NHANES download, 2021 CKD-EPI eGFR & export
    └── project_cluster/
        ├── __init__.py
        ├── 01_prepare_cohort.py                <- Adult fasting filter (N=3,515), CDC survey weights, 70/30 split
        ├── 02_train_and_tune.py                <- 5-fold CV tuning across k=2..6 (k-Means, GMM, Autoencoder)
        ├── 03_stability_and_null.py            <- 100-bootstrap Jaccard stability & permutation null baseline
        └── 04_evaluate_test_and_disparities.py <- Held-out test evaluation (n=1,055), clinical & disparity tests
```

---

## 6. Governance, Model Card & Reproducibility

Full details on the **Model Card**, **CRediT Author Contributions**, **Generative AI Disclosure**, **MIT License**, and **Step-by-Step Reproduction Instructions** are documented in [`CRediT_reproducibility.md`](CRediT_reproducibility.md).