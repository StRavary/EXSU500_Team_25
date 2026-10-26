# EXSU500_Team_25

### Problem Class: Clustering/Phenotyping

### Dataset: NHANES

### Problem: Do data-driven cardiometabolic phenotypes found in NHANES capture real clinical differences, or do they mostly reproduce demographic and socioeconomic groups?
# Data-Driven Cardiometabolic Phenotyping vs. Sociodemographic Confounding

**Course:** EXSU 500: Fundamentals of Artificial Intelligence in Medicine  
**Institution:** Department of Surgery, McGill University (Fall 2026)  
**Academic Affiliations:** McGill University, Mila (Quebec AI Institute)  
**Repository Access:** Graded academic repository (Read access granted to course staff @AHooshiar and teaching assistants)

---

## 1. Project Overview and Task Definition

Cardiometabolic diseases, including type 2 diabetes mellitus, dyslipidemia, and arterial hypertension, are commonly evaluated using isolated clinical thresholds. Examples include fasting plasma glucose cutoffs or systolic blood pressure boundaries. While straightforward, single-marker criteria frequently fail to reflect complex, multi-system physiological interactions.

This project formulates an unsupervised machine learning task to discover data-driven cardiometabolic risk endotypes from continuous physiological biomarkers. Using unsupervised clustering, we identify distinct metabolic subgroups and evaluate whether membership in these phenotypes differs significantly across social determinants of health: race/ethnicity, adult educational attainment, and family income-to-poverty ratio.

### Problem Formulation
* **Task Type:** Unsupervised clustering and clinical phenotyping.
* **Input Feature Space:** Six continuous, objective physiological measurements: Glycated Hemoglobin (HbA1c, %), Serum Triglycerides (mg/dL), Total Cholesterol (mg/dL), High-Density Lipoprotein Cholesterol (HDL, mg/dL), Body Mass Index (BMI, kg/m²), and Systolic Blood Pressure (SBP, mmHg).
* **Target Phenotypes:** Data-driven patient clusters characterizing distinct patterns of cardiometabolic risk.
* **Downstream Assessment:** Statistical hypothesis tests measuring demographic and socioeconomic divergence across clusters to identify systemic screening and outcome disparities.

---

## 2. Dataset Provenance and Analytical Domain

### Source Dataset
Data are obtained from the National Health and Nutrition Examination Survey (NHANES) 2017 to March 2020 pre-pandemic release, administered by the National Center for Health Statistics (NCHS) at the Centers for Disease Control and Prevention (CDC). The full pre-pandemic release includes 15,560 individuals across Demographics, Examination, and Laboratory components.

* **NHANES Source Cycle:** 2017 to March 2020 Pre-Pandemic Examination Data
* **Data Custodian:** CDC / NCHS (Hyattsville, MD, USA)
* **Licence and Access:** Open-access public domain data. Free to download without credentialed access, institutional approval, or data use agreements.

### Analytical Domain Formulation
Directly clustering across the entire 15,560 records creates methodological distortion due to structural survey design:
1. **Educational Attainment:** Administered only to adult participants aged 20 years and older (9,232 participants). Missingness in younger age brackets reflects eligibility criteria rather than random nonresponse.
2. **Fasting Lipids (Triglycerides and LDL):** Administered strictly to an assigned morning fasting subsample aged 12 years and older (4,650 participants). This represents planned subsample missingness by design.

To prevent artificial clustering around synthetic data points, whole-table imputation is not performed. The analytical domain is defined as adult participants aged 20 years or older who completed the morning fasting phlebotomy examination and have complete profiles for core cardiometabolic markers. This protocol yields an uncompromised complete-case analytical cohort of N = 3,515 individuals.

### Feature Schema and Transformation Dictionary

| Variable Name | Concept / Measurement | Source Component / NHANES Code | Analytical Role |
| :--- | :--- | :--- | :--- |
| **ID** | Participant Sequence Token | `SEQN` | Relational primary key |
| **HbA1c_Percent** | Glycohemoglobin (%) | `P_GHB` (`LBXGH`) | Clustering feature |
| **Triglycerides** | Serum Triglycerides (mg/dL) | `P_TRIGLY` (`LBXTR`) | Clustering feature |
| **Total_Cholesterol** | Total Serum Cholesterol (mg/dL) | `P_TCHOL` (`LBXTC`) | Clustering feature |
| **HDL_Cholesterol** | High-Density Lipoprotein (mg/dL) | `P_HDL` (`LBDHDD`) | Clustering feature |
| **BMI** | Body Mass Index (kg/m²) | `P_BMX` (`BMXBMI`) | Clustering feature |
| **Systolic_BP** | Mean Systolic Blood Pressure (mmHg) | Engineered from `P_BPXO` | Clustering feature |
| **Diastolic_BP** | Mean Diastolic Blood Pressure (mmHg) | Engineered from `P_BPXO` | Clinical reference |
| **eGFR** | Renal Function (mL/min/1.73m²) | Engineered via 2021 CKD-EPI | Clinical reference |
| **Inflammation_CRP** | High-Sensitivity CRP (mg/L) | `P_HSCRP` (`LBXHSCRP`) | Clinical reference |
| **Age_Years** | Participant Age at Exam | `P_DEMO` (`RIDAGEYR`) | Splitting / Stratification |
| **Gender** | Categorized Biological Sex | `P_DEMO` (`RIAGENDR`) | Demographic variable |
| **Race_Ethnicity** | Self-Reported Ethno-Racial Origin | `P_DEMO` (`RIDRETH3`) | Sociodemographic test variable |
| **Education_Level** | Adult Attainment (5 levels) | `P_DEMO` (`DMDEDUC2`) | Sociodemographic test variable |
| **Income_Poverty_Ratio** | Family Income-to-Poverty Ratio | `P_DEMO` (`INDFMPIR`) | Sociodemographic test variable |

*Note on Blood Pressure Standardization:* Systolic and diastolic values represent the vectorized row-wise mean of up to three consecutive oscillometric readings (`BPXOSY1`, `BPXOSY2`, `BPXOSY3`) to minimize single-reading measurement error.

*Note on Renal Filtration:* Estimated Glomerular Filtration Rate (eGFR) is computed using the 2021 CKD-EPI Creatinine equation without race parameters.

---

## 3. Modelling Strategy and Technical Protocol

### Model Families Under Comparison
To meet course requirements, three distinct model families are evaluated:
1. **Baseline Model (k-Means):** Standard Euclidean k-means with k-means++ initialization across 50 random initializations. This establishes the baseline floor for spherical geometric partitioning.
2. **Classical Machine Learning Model (Gaussian Mixture Models):** Probabilistic Gaussian Mixture Models (GMM) fitted via Expectation-Maximization. GMM relaxes spherical geometry by estimating full covariance matrices, enabling ellipsoidal clusters and soft probabilistic class assignments.
3. **Deep Representation Learning Model (Undercomplete Autoencoder):** A symmetric deep autoencoder compressing the six input biomarkers into a low-dimensional bottleneck (2 to 3 dimensions) using mean squared reconstruction error and weight decay regularization. A clustering layer is applied to the bottleneck representations to capture nonlinear physiological interactions.

### Data Partitioning and Leakage Prevention
* **Cohort Splits:** Complete-case analytical records (N = 3,515) are partitioned into a Training Set (70%, n = 2,460) and an independent Held-Out Test Set (30%, n = 1,055) using stratified sampling across age deciles.
* **Leakage Protocol:** All normalization parameters (z-score means and standard deviations) are computed strictly on the training partition. The held-out test set is transformed using these stored parameters and remains untouched during hyperparameter tuning and model selection.
* **Internal Cluster Validation:** Cluster count selection (k in [2, 6]) is conducted on the training set using the Silhouette Coefficient, Davies-Bouldin Index, and the Bayesian Information Criterion (BIC).
* **Stability Evaluation:** Stability is measured using 100 non-parametric bootstrap resamples of the training data, evaluated via mean Jaccard similarity.
* **Downstream Disparity Tests:** Cluster assignments are tested against demographic variables using Pearson's chi-square tests for categorical variables (Race_Ethnicity and Education_Level) and one-way ANOVA with Kruskal-Wallis tests for Income_Poverty_Ratio.

---

## 4. Preliminary Baseline Results

A baseline k-means model (k = 3) fit to standardized HbA1c and Triglycerides within the adult fasting domain identified three clinically distinct risk phenotypes:

| Cluster Profile | Cohort Share N (%) | Mean HbA1c (%) | Mean Triglycerides (mg/dL) | Mean BMI (kg/m²) | Mean Systolic BP (mmHg) | Mean Income PIR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cluster 1: Normoglycemic / Normolipidemic** | 2,931 (83.4%) | 5.61 | 86.4 | 29.5 | 123.2 | 2.67 |
| **Cluster 0: Hypertriglyceridemic** | 357 (10.2%) | 5.85 | 277.7 | 31.8 | 125.8 | 2.64 |
| **Cluster 2: Severe Hyperglycemic** | 227 (6.5%) | 9.44 | 149.0 | 33.3 | 131.3 | 2.26 |

### Sociodemographic Distribution Differences
* **Race and Ethnicity (Chi-square = 87.91, p < 0.001):** Mexican American (10.25%) and Non-Hispanic Black (7.51%) participants showed the highest concentration within the severe glycemic cluster. Mexican American (14.12%) and Non-Hispanic Asian (13.68%) participants had elevated rates in the hypertriglyceridemic cluster, whereas Non-Hispanic Black participants showed low rates (3.03%), illustrating known population differences in lipid profiles.
* **Educational Attainment (Chi-square = 45.77, p < 0.001):** Severe hyperglycemia prevalence followed a clear inverse education gradient: Less than 9th grade (12.65%), High school/GED (7.36%), and College graduate or higher (3.10%).
* **Income-to-Poverty Ratio (Kruskal-Wallis H = 11.43, p = 0.003):** Mean PIR was significantly lower in the severe hyperglycemic cluster (2.26) compared to the normoglycemic cluster (2.67).

---

## 5. Repository Structure and Reproducibility

```text
├── README.md                                <- Project specification, methodology, and results summary
├── LICENSE                                  <- Open-source licence file (MIT License)
├── requirements.txt                         <- Pinned Python dependencies for modeling pipeline
├── environment.R                            <- R environment specification and package dependencies
├── data/
│   ├── raw/                                 <- Local download directory for raw CDC NHANES XPT files (gitignored)
│   ├── processed/
│   │   ├── nhanes_phenotype_analytical_matrix.csv  <- Curation matrix with semantic headers
│   │   └── nhanes_phenotype_analytical_matrix.rds  <- Native R binary preserving factor types
│   └── manifests/
│       └── metadata_data_curation_manifest.txt     <- Automated curation log and range audits
├── src/
│   ├── 01_data_curation_pipeline.R          <- Ingestion, eGFR calculation, BP averaging, export
│   ├── 02_preprocessing_and_splits.py       <- Domain filtering (Age >= 20, fasting), train/test splits
│   ├── 03_baseline_kmeans.py                <- Baseline k-means fitting and internal metric evaluation
│   ├── 04_classical_gmm.py                  <- Gaussian Mixture Models implementation and BIC selection
│   ├── 05_deep_autoencoder.py               <- Bottleneck latent autoencoder architecture in PyTorch
│   └── 06_statistical_disparity_tests.py    <- Chi-square, ANOVA, and Kruskal-Wallis evaluations
└── docs/
    └── model_card.md                        <- Formal model card documenting intended use and limitations
```

### Reproducing the Pipeline

#### Step 1: Data Ingestion and Curation (R Environment)
The raw survey files can be downloaded and curated directly from the CDC servers using the automated curation pipeline:
```bash
Rscript src/01_data_curation_pipeline.R
```
This script queries the CDC database via `nhanesA`, calculates the 2021 CKD-EPI eGFR equation, averages blood pressure readings, and exports the clean CSV and RDS files alongside an audit manifest.

#### Step 2: Model Training and Evaluation (Python Environment)
Create and activate an isolated environment, install pinned dependencies, and execute the modeling scripts:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

# Run preprocessing and baseline modeling
python src/02_preprocessing_and_splits.py
python src/03_baseline_kmeans.py
```

---

## 6. Model Card Summary

* **Intended Use:** Unsupervised stratification of cardiometabolic risk patterns in epidemiological and population health research.
* **Target Modality:** Tabular adult physiological and lipid lab panels.
* **Training Data:** NHANES 2017 to March 2020 pre-pandemic adult fasting subsample (Train split, n = 2,460).
* **Evaluation Data:** Held-out test split (n = 1,055).
* **Primary Metrics:** Silhouette Coefficient, Davies-Bouldin Index, BIC, and bootstrap Jaccard assignment stability.
* **Known Limitations:** The model is not an approved clinical diagnostic tool and should not be used as a standalone guide for medical interventions. Fasting lipids were collected in a specific morning fasting subpopulation; predictions on non-fasting individuals are structurally invalid.

---

## 7. CRediT Contribution Statement

* **Data Lead (Krisen Jean Julien):** Conceptualization, Data Curation, Software (R Curation Pipeline), Validation (Metadata Manifest and Curation Logging), Writing (Data Engineering Summary).
* **Modelling Lead (Steven Ravary):** Methodology (Clustering Formulation, Partition Protocols), Software (Clustering Models, Feature Scaling), Formal Analysis (Cluster Validation, Feasibility Benchmarks), Writing (Methods and Feasibility Plan).
* **Interface Lead:** Software (Prototype Interface Architecture), Visualization (Disparity Visualizations), Deployment Planning.
* **Writing Lead:** Project Administration, Investigation (Literature Review Synthesis), Writing (Original Manuscript Draft, IEEE Formatting and Assembly).

---

## 8. Use of Generative AI Statement

In accordance with course academic integrity guidelines, generative AI tools (Google Gemini) were used during this project for coding assistance, debugging, and text editing. Specifically, AI assistance supported mapping CDC alphanumeric codes to descriptive column headers, writing vectorized R expressions for the 2021 CKD-EPI formula and blood pressure averaging, configuring baseline scikit-learn clustering workflows, and refining technical prose for clarity. 

All pipeline logic, mathematical derivations, package versions, data manifests, and model evaluations were independently executed, tested, and audited in local R and Python environments. The authors take full intellectual responsibility for all code, results, and text presented in this repository.

---

## 9. Licence

This project is released under the **MIT Licence**. CDC NHANES data components are works of the United States Government and reside in the public domain.
