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
* **Downstream Hypothesis Testing:** Statistical evaluation of demographic and socioeconomic divergence across phenotypes using Pearson's $\chi^2$ tests, Cramér's $V$, Kruskal–Wallis tests, and Benjamini–Hochberg False Discovery Rate (FDR) correction, complemented by survey-weighted sensitivity analyses.

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
* **5-Fold Cross-Validation ($k \in \{2, 3, 4, 5, 6\}$):** Cluster granularity and autoencoder bottleneck dimensions are tuned strictly within the training set using out-of-fold Silhouette Coefficient, Davies-Bouldin Index, and Bayesian Information Criterion (BIC).
* **Stability & Permutation Null Testing:** Model robustness is audited on the training set via 100 non-parametric bootstrap resamples (Hungarian-matched Jaccard similarity) and 100 column-wise feature-permutation null trials.

---

## 4. Empirical Results Summary

### A. 5-Fold Cross-Validation Tuning (`n = 2,460` Training Set)
Across $k \in \{2, 3, 4, 5, 6\}$, `k = 2` maximizes out-of-fold geometric Silhouette separation across all three model families, while `GMM` achieves its minimum Bayesian Information Criterion at `k = 3` (`BIC = 7383.57`):

| Model Family | Optimal Configuration | Mean Val Silhouette ($\uparrow$) | SD Val Silhouette | Mean Val Davies–Bouldin ($\downarrow$) | Mean Val BIC ($\downarrow$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **k-Means (Baseline)** | `k = 2, n_init = 50` | 0.1764 | 0.0167 | 1.9854 | — |
| **GMM (Classical ML)** | `k = 2, cov = full` | 0.2633 | 0.0038 | 2.1553 | 7423.79 |
| **GMM (BIC Optimum)** | `k = 3, cov = full` | 0.1304 | 0.0253 | 2.1966 | **7383.57** |
| **Autoencoder (Deep)** | `k = 2, bottleneck = 2D` *(latent)* | 0.6131 | 0.1528 | 0.6939 | — |
| **Autoencoder (Deep)** | `k = 2, bottleneck = 3D` *(latent)* | 0.3252 | 0.0594 | 1.2210 | — |

*(Note: Autoencoder Silhouette scores above reflect separation within the compressed 2D/3D latent bottleneck; ambient 6D Euclidean metrics are reported alongside latent metrics in `03_stability_and_null.py` and `04_evaluate_test_and_disparities.py` for direct comparison against k-Means and GMM).*

### B. Preliminary Full-Cohort Reference Baseline (`N = 3,515`, `k = 3`)
Initial feasibility clustering (`k = 3`) on standardized HbA1c and Triglycerides identified three clinically distinct risk subgroups with strong sociodemographic gradients:

| Cluster Profile | Cohort Share N (%) | Mean HbA1c (%) | Mean Triglycerides (mg/dL) | Mean BMI (kg/m²) | Mean Systolic BP (mmHg) | Mean Income PIR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normoglycemic / Normolipidemic** | 2,931 (83.4%) | 5.61 | 86.4 | 29.5 | 123.2 | 2.67 |
| **Hypertriglyceridemic** | 357 (10.2%) | 5.85 | 277.7 | 31.8 | 125.8 | 2.64 |
| **Severe Hyperglycemic** | 227 (6.5%) | 9.44 | 149.0 | 33.3 | 131.3 | 2.26 |

* **Race/Ethnicity ($\chi^2 = 87.91, p < 0.001$):** Severe hyperglycemia was disproportionately concentrated among Mexican American ($10.25\%$) and Non-Hispanic Black ($7.51\%$) participants, whereas hypertriglyceridemia was elevated in Mexican American ($14.12\%$) and Non-Hispanic Asian ($13.68\%$) participants compared to Non-Hispanic Black ($3.03\%$) participants.
* **Educational Attainment ($\chi^2 = 45.77, p < 0.001$):** Severe hyperglycemia prevalence followed an inverse socioeconomic gradient from $<9\text{th}$ grade ($12.65\%$) and high school/GED ($7.36\%$) to college graduates ($3.10\%$).
* **Income-to-Poverty Ratio (Kruskal–Wallis $H = 11.43, p = 0.003$):** Mean PIR was significantly lower in the severe hyperglycemic group ($2.26$) than in the normoglycemic group ($2.67$).

---

## 5. Repository Structure

```text
EXSU500_Team_25/
├── README.md                                   <- Project overview, methodology, results, model card & reproduction guide
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