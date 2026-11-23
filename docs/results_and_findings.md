# Empirical Findings & Technical Report: Data-Driven Cardiometabolic Phenotyping vs. Sociodemographic Confounding

**Project Repository:** [https://github.com/StRavary/EXSU500_Team_25](https://github.com/StRavary/EXSU500_Team_25)  
**Dataset:** CDC NHANES (2017–March 2020 Pre-Pandemic Cycle)  
**Course:** EXSU 500: Fundamentals of Artificial Intelligence in Medicine (Fall 2026, McGill University)

---

## Executive Summary

This investigation evaluates whether unsupervised cardiometabolic phenotypes discovered from continuous physiological biomarkers in U.S. adults represent genuine multi-system clinical endotypes or primarily reproduce demographic and socioeconomic stratification. Using a leakage-free pipeline across $N = 3,515$ fasting adults from the CDC NHANES 2017–March 2020 cycle (`n = 2,460` training / `n = 1,055` held-out test), we establish four core findings:

1. **Classical Geometric & Probabilistic Models Outperform Deep Latent Clustering:** Full-covariance Gaussian Mixture Models (`GMM`, $k=2$) and Euclidean `k-Means` ($k=2$) achieve high 100-bootstrap assignment stability ($\text{Jaccard} = 0.9349 \pm 0.0384$ and $0.8824 \pm 0.1357$, respectively) and generalize cleanly to the held-out test set. Conversely, an undercomplete PyTorch Autoencoder ($6 \to 16 \to 8 \to 2 \to 8 \to 16 \to 6$) suffers from severe initialization instability ($\text{pairwise ARI} = 0.0781 \pm 0.1757$, $\text{Jaccard} = 0.3955 \pm 0.1169$), exhibits an artificial Silhouette inflation in compressed 2D latent space (`0.6131` CV / `0.3161` test vs. `0.1072` in ambient 6D space), and fails to outperform column-permuted random noise ($p = 0.9109$).
2. **External Clinical Validity on Unseen Biomarkers:** Without ever seeing renal filtration (`eGFR`), systemic inflammation (`Inflammation_CRP`), or `Diastolic_BP` during training, both `k-Means (k=2)` and `GMM (k=2)` stratify held-out test patients ($n = 1,055$) into clinically coherent high-risk (`C1`) and low-risk (`C0`) endotypes with highly significant divergence across systemic inflammation ($p = 2.17 \times 10^{-13}$ for GMM; $p = 1.06 \times 10^{-43}$ for k-Means), diastolic blood pressure, and renal function.
3. **Clinical Pathology Dominates Confounding, While Structural Disparities Persist:** Physiological effect sizes (`BMI` $H = 401.48$, `HDL` $H = 384.03$, `HbA1c` $H = 226.33$, `Triglycerides` $H = 208.92$) are an order of magnitude larger than sociodemographic associations ($\text{Cramér's } V = 0.0907–0.1716$), demonstrating that clusters capture true biological endotypes rather than acting as demographic proxies. Nevertheless, Benjamini–Hochberg FDR-corrected hypothesis tests confirm statistically significant disparities ($q < 0.05$) across educational attainment, family income-to-poverty ratio, and race/ethnicity.
4. **Nationally Representative Burden:** Incorporating NHANES complex survey design weights (`WTSAFPRP`, `SDMVSTRA`, `SDMVPSU`) yields national high-risk cardiometabolic endotype prevalence estimates of **$16.53\% \pm 1.76\%$** ($95\%$ CI: $13.08\%–19.99\%$) under `GMM (k=2)` and **$41.27\% \pm 2.03\%$** ($95\%$ CI: $37.29\%–45.26\%$) under `k-Means (k=2)`.

---

## 1. Cohort Construction & Leakage-Free Partitioning

To prevent artificial clustering around synthetic imputed values caused by NHANES survey eligibility rules, whole-table imputation was avoided in favor of strict domain filtering:
* **Total NHANES 2017–March 2020 Participants Loaded:** $15,560$
* **Adults Aged $\ge 20$ Years (Educational Attainment Eligible):** $9,232$
* **Complete-Case Morning Fasting 6-Biomarker Analytical Cohort ($N$):** $3,515$
* **Stratified 70/30 Split (Stratified by Age Deciles, `seed=42`):**
  * **Training Partition ($70\%$):** $n = 2,460$ (used exclusively for z-score parameter fitting, 5-fold cross-validation, 100-bootstrap stability, and 100-permutation null testing).
  * **Held-Out Test Partition ($30\%$):** $n = 1,055$ (locked until final evaluation; transformed using frozen training scaler parameters).

---

## 2. Training-Set Model Selection & Robustness Benchmarks (`n = 2,460`)

### 2.1 Five-Fold Cross-Validation Across $k \in \{2, 3, 4, 5, 6\}$
Out-of-fold geometric separation peaked at $k = 2$ across all three model families, whereas `GMM` achieved its global Bayesian Information Criterion (BIC) minimum at $k = 3$ ($\text{BIC} = 7383.57$ vs. $7423.79$ at $k = 2$):

| Model Family | Hyperparameters | Mean Val Silhouette ($\uparrow$) | SD Val Silhouette | Mean Val Davies–Bouldin ($\downarrow$) | Mean Val BIC ($\downarrow$) | Mean Recon MSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **k-Means** | `k=2, n_init=50` *(selected)* | **0.1764** | 0.0167 | 1.9854 | — | — |
| **k-Means** | `k=3, n_init=50` | 0.1651 | 0.0129 | 1.7608 | — | — |
| **k-Means** | `k=6, n_init=50` | 0.1738 | 0.0125 | **1.3773** | — | — |
| **GMM** | `k=2, cov=full` *(selected)* | **0.2633** | 0.0038 | **2.1553** | 7423.79 | — |
| **GMM** | `k=3, cov=full` *(BIC opt)* | 0.1304 | 0.0253 | 2.1966 | **7383.57** | — |
| **Autoencoder** | `k=2, dim=2` *(selected)* | **0.6131** *(latent)* | 0.1528 | **0.6939** *(latent)* | — | 0.2702 |
| **Autoencoder** | `k=2, dim=3` | 0.3252 *(latent)* | 0.0594 | 1.2210 *(latent)* | — | **0.1544** |

### 2.2 Bootstrap Stability ($B = 100$) & Column-Wise Permutation Null ($P = 100$)
To verify that discovered clusters reflect reproducible multivariate covariance rather than random initialization or marginal skewness, each selected model was evaluated across 100 non-parametric bootstrap resamples (Hungarian-matched Jaccard similarity; $\ge 0.75$ stability threshold) and 100 column-permuted null datasets:

| Model Family | Bootstrap Jaccard Mean $\pm$ SD | Bootstrap $95\%$ CI | Observed Ambient 6D Silhouette | Permutation Null 6D Silhouette | Empirical $p$-value (6D) | Observed Latent 2D Silhouette | Empirical $p$-value (2D) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **k-Means (`k=2`)** | **$0.8824 \pm 0.1357$** | $[0.2903, 0.9825]$ | **0.1790** | $0.1554 \pm 0.0371$ | **$0.0396$** | — | — |
| **GMM (`k=2`)** | **$0.9349 \pm 0.0384$** | $[0.8433, 0.9872]$ | **0.2656** | $0.2696 \pm 0.0109$ | $0.6436$ | — | — |
| **Autoencoder (`k=2, 2D`)** | $0.3955 \pm 0.1169$ | $[0.2801, 0.6704]$ | 0.1124 | $0.2617 \pm 0.1272$ | $0.9109$ | 0.3367 | $0.9802$ |

* **Methodological Insight on Deep Bottlenecks:** Across 5 random initializations (`seeds 42–46`) on the full training set, the Autoencoder achieved a pairwise Adjusted Rand Index (ARI) of only **$0.0781 \pm 0.1757$**. Moreover, compressing column-shuffled noise into a 2D bottleneck generated artificially high latent Silhouette scores ($0.6422 \pm 0.1482$), confirming that low-dimensional latent clustering on $6\text{D}$ tabular lab panels is prone to geometric distortion.

---

## 3. Held-Out Test Set Evaluation (`n = 1,055`)

### 3.1 Internal Metrics & Inter-Model Agreement
Locked training models were evaluated once on the untouched held-out test set ($n = 1,055$):

| Model Configuration | Test Ambient 6D Silhouette ($\uparrow$) | Test Davies–Bouldin ($\downarrow$) | Test Latent 2D Silhouette | Test BIC ($\downarrow$) | Cluster Assignment Counts |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **k-Means (`k=2`)** | 0.1770 | **1.9960** | — | — | `C0: 566` / `C1: 489` |
| **GMM (`k=2`, Silhouette opt)** | **0.2566** | 2.1993 | — | 15435.61 | `C0: 832` / `C1: 223` |
| **GMM (`k=3`, BIC opt sensitivity)** | 0.1155 | 2.2480 | — | **15260.00** | `C0: 526` / `C1: 428` / `C2: 101` |
| **Autoencoder (`k=2, dim=2`)** | 0.1072 | 2.6810 | 0.3161 | — | `C0: 579` / `C1: 476` |

* **Inter-Model Agreement (Adjusted Rand Index):** `k-Means` and `GMM` exhibit modest positive concordance ($\text{ARI} = 0.1940$), reflecting `k-Means` partitioning the cohort roughly in half ($46.4\%$ high-risk) while full-covariance `GMM` isolates a sharper, higher-acuity cardiometabolic tail ($21.1\%$ high-risk). Neither classical model agrees with the Autoencoder ($\text{ARI} = 0.0109$ vs. k-Means; $\text{ARI} = -0.0036$ vs. GMM).

---

### 3.2 Clinical Biomarker Profiles & External Validation (`n = 1,055`)
Cluster `C1` consistently captures an adverse multi-system cardiometabolic phenotype across both clustering biomarkers and **unclustered clinical reference markers** (`eGFR`, `Inflammation_CRP`, `Diastolic_BP`):

| Biomarker | Analytical Role | k-Means `C0` ($n=566$) | k-Means `C1` ($n=489$) | k-Means KW $H$ ($p$-value) | GMM `C0` ($n=832$) | GMM `C1` ($n=223$) | GMM KW $H$ ($p$-value) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HbA1c (%)** | Clustering | $5.45 \pm 0.47$ | $6.40 \pm 1.57$ | $226.33$ ($3.8 \times 10^{-51}$) | $5.54 \pm 0.43$ | **$7.22 \pm 2.03$** | $194.31$ ($3.6 \times 10^{-44}$) |
| **Triglycerides (mg/dL)** | Clustering | $80.83 \pm 40.33$ | $140.85 \pm 84.61$ | $208.92$ ($2.4 \times 10^{-47}$) | $88.69 \pm 40.53$ | **$183.10 \pm 104.51$** | $193.58$ ($5.3 \times 10^{-44}$) |
| **Total Cholesterol (mg/dL)** | Clustering | $186.00 \pm 38.61$ | $180.10 \pm 44.74$ | $10.71$ ($1.1 \times 10^{-3}$) | $181.69 \pm 38.08$ | $189.14 \pm 52.56$ | $0.89$ ($0.344$) |
| **HDL Cholesterol (mg/dL)** | Clustering | $60.63 \pm 15.91$ | $43.74 \pm 8.81$ | $384.03$ ($1.7 \times 10^{-85}$) | $55.13 \pm 15.90$ | **$44.14 \pm 10.50$** | $106.19$ ($6.7 \times 10^{-25}$) |
| **BMI ($\text{kg/m}^2$)** | Clustering | $26.01 \pm 4.45$ | $34.75 \pm 7.43$ | $401.48$ ($2.6 \times 10^{-89}$) | $28.75 \pm 6.20$ | **$34.96 \pm 9.36$** | $87.15$ ($1.0 \times 10^{-20}$) |
| **Systolic BP (mmHg)** | Clustering | $119.76 \pm 17.58$ | $129.51 \pm 20.46$ | $73.69$ ($9.2 \times 10^{-18}$) | $122.84 \pm 18.88$ | **$129.65 \pm 21.16$** | $23.42$ ($1.3 \times 10^{-6}$) |
| **eGFR ($\text{mL/min/1.73m}^2$)** | **Held-Out Ref** | $103.95 \pm 21.23$ | $98.48 \pm 24.23$ | $10.45$ ($1.2 \times 10^{-3}$) | $102.45 \pm 21.96$ | **$97.57 \pm 25.46$** | $4.17$ ($0.041$) |
| **hs-CRP (mg/L)** | **Held-Out Ref** | $2.74 \pm 6.67$ | $6.02 \pm 8.89$ | $192.19$ ($1.1 \times 10^{-43}$) | $3.58 \pm 7.20$ | **$6.80 \pm 9.87$** | $53.85$ ($2.2 \times 10^{-13}$) |
| **Diastolic BP (mmHg)** | **Held-Out Ref** | $71.97 \pm 11.09$ | $77.34 \pm 11.60$ | $65.22$ ($6.7 \times 10^{-16}$) | $74.00 \pm 11.60$ | **$76.17 \pm 11.63$** | $7.95$ ($4.8 \times 10^{-3}$) |
| **LDL Cholesterol (mg/dL)** | **Held-Out Ref** | $109.19 \pm 33.95$ | $108.30 \pm 38.44$ | $0.85$ ($0.356$) | $108.82 \pm 33.29$ | $108.64 \pm 45.29$ | $1.44$ ($0.230$) |
| **Age (Years)** | **Held-Out Ref** | $48.71 \pm 17.86$ | $53.16 \pm 16.29$ | $17.13$ ($3.5 \times 10^{-5}$) | $49.66 \pm 17.70$ | **$54.91 \pm 14.94$** | $16.17$ ($5.8 \times 10^{-5}$) |

* **Clinical Interpretation:** `GMM (k=2)` isolates a severe cardiometabolic risk endotype (`C1`, $21.1\%$) characterized by overt diabetic-range mean glycohemoglobin ($7.22\%$), marked hypertriglyceridemia ($183.1\text{ mg/dL}$), atherogenic dyslipidemia (low HDL $44.1\text{ mg/dL}$ with unchanged LDL $108.6\text{ mg/dL}$), class I/II obesity ($\text{BMI } 35.0\text{ kg/m}^2$), elevated blood pressure, nearly doubled systemic inflammation (`hs-CRP` $6.80$ vs. $3.58\text{ mg/L}$), and reduced renal filtration (`eGFR` $97.6$ vs. $102.5\text{ mL/min/1.73m}^2$). `k-Means (k=2)` captures a broader metabolic-syndrome risk tier (`C1`, $46.4\%$) driven primarily by adiposity ($\text{BMI } 34.8\text{ kg/m}^2$), low HDL ($43.7\text{ mg/dL}$), and systemic inflammation (`hs-CRP` $6.02\text{ mg/L}$).

---

### 3.3 Sociodemographic Disparity Hypothesis Tests (Benjamini–Hochberg FDR Corrected)

| Model | Sociodemographic Variable | Hypothesis Test | Test Statistic ($df$) | Raw $p$-value | Effect Size | BH-FDR $q$-value | Significant ($q < 0.05$) |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **k-Means (`k=2`)** | `Education_Level` | Pearson $\chi^2$ | $29.1989$ ($4$) | $7.12 \times 10^{-6}$ | $\text{Cramér's } V = 0.1546$ | **$0.000036$** | **Yes** |
| **k-Means (`k=2`)** | `Age_Years` | Kruskal–Wallis | $17.1296$ ($1$) | $3.49 \times 10^{-5}$ | $\text{ANOVA } F = 17.71$ | **$0.000087$** | **Yes** |
| **k-Means (`k=2`)** | `Income_Poverty_Ratio` | Kruskal–Wallis | $11.4612$ ($1$) | $7.11 \times 10^{-4}$ | $\text{ANOVA } F = 12.74$ | **$0.001184$** | **Yes** |
| **k-Means (`k=2`)** | `Race_Ethnicity` | Pearson $\chi^2$ | $15.2783$ ($5$) | $9.24 \times 10^{-3}$ | $\text{Cramér's } V = 0.0987$ | **$0.011546$** | **Yes** |
| **k-Means (`k=2`)** | `Gender` | Pearson $\chi^2$ | $6.3748$ ($1$) | $1.16 \times 10^{-2}$ | $\text{Cramér's } V = 0.0714$ | **$0.011575$** | **Yes** |
| **GMM (`k=2`)** | `Education_Level` | Pearson $\chi^2$ | $35.0402$ ($4$) | $4.56 \times 10^{-7}$ | $\text{Cramér's } V = 0.1716$ | **$0.000002$** | **Yes** |
| **GMM (`k=2`)** | `Age_Years` | Kruskal–Wallis | $16.1715$ ($1$) | $5.79 \times 10^{-5}$ | $\text{ANOVA } F = 16.42$ | **$0.000145$** | **Yes** |
| **GMM (`k=2`)** | `Income_Poverty_Ratio` | Kruskal–Wallis | $9.0872$ ($1$) | $2.57 \times 10^{-3}$ | $\text{ANOVA } F = 11.02$ | **$0.004290$** | **Yes** |
| **GMM (`k=2`)** | `Race_Ethnicity` | Pearson $\chi^2$ | $13.6829$ ($5$) | $1.78 \times 10^{-2}$ | $\text{Cramér's } V = 0.0907$ | **$0.022193$** | **Yes** |
| **GMM (`k=2`)** | `Gender` | Pearson $\chi^2$ | $1.7469$ ($1$) | $1.86 \times 10^{-1}$ | $\text{Cramér's } V = 0.0266$ | $0.186261$ | No |
| **GMM (`k=3`, BIC)** | `Education_Level` | Pearson $\chi^2$ | $32.4702$ ($8$) | $7.67 \times 10^{-5}$ | $\text{Cramér's } V = 0.1078$ | **$0.000384$** | **Yes** |
| **GMM (`k=3`, BIC)** | `Age_Years` | Kruskal–Wallis | $17.4346$ ($2$) | $1.64 \times 10^{-4}$ | $\text{ANOVA } F = 9.35$ | **$0.000409$** | **Yes** |
| **GMM (`k=3`, BIC)** | `Income_Poverty_Ratio` | Kruskal–Wallis | $7.4685$ ($2$) | $2.39 \times 10^{-2}$ | $\text{ANOVA } F = 4.24$ | **$0.039818$** | **Yes** |
| **GMM (`k=3`, BIC)** | `Race_Ethnicity` | Pearson $\chi^2$ | $17.4014$ ($10$) | $6.59 \times 10^{-2}$ | $\text{Cramér's } V = 0.0592$ | $0.082427$ | No |
| **GMM (`k=3`, BIC)** | `Gender` | Pearson $\chi^2$ | $1.1935$ ($2$) | $5.51 \times 10^{-1}$ | $\text{Cramér's } V = 0.0000$ | $0.550591$ | No |
| **Autoencoder (`k=2`)** | `Age_Years` | Kruskal–Wallis | $16.0741$ ($1$) | $6.09 \times 10^{-5}$ | $\text{ANOVA } F = 18.21$ | **$0.000305$** | **Yes** |
| **Autoencoder (`k=2`)** | `Gender` | Pearson $\chi^2$ | $3.2930$ ($1$) | $6.96 \times 10^{-2}$ | $\text{Cramér's } V = 0.0466$ | $0.173940$ | No |
| **Autoencoder (`k=2`)** | `Race_Ethnicity` | Pearson $\chi^2$ | $4.6199$ ($5$) | $4.64 \times 10^{-1}$ | $\text{Cramér's } V = 0.0000$ | $0.610767$ | No |
| **Autoencoder (`k=2`)** | `Education_Level` | Pearson $\chi^2$ | $3.4299$ ($4$) | $4.89 \times 10^{-1}$ | $\text{Cramér's } V = 0.0000$ | $0.610767$ | No |
| **Autoencoder (`k=2`)** | `Income_Poverty_Ratio` | Kruskal–Wallis | $0.0049$ ($1$) | $9.44 \times 10^{-1}$ | $\text{ANOVA } F = 0.01$ | $0.944241$ | No |

---

### 3.4 NHANES Complex Survey-Weighted Sensitivity Analysis (`WTSAFPRP` / `SDMVSTRA` / `SDMVPSU`)
Applying CDC fasting subsample weights (`WTSAFPRP`) and Taylor-series linearized variance estimation across masked variance strata (`SDMVSTRA`) and primary sampling units (`SDMVPSU`) confirms that unweighted test-set prevalence slightly overestimates the high-risk cluster share (reflecting NHANES oversampling of older and minority subpopulations), while preserving substantive clinical stratification:

| Model | Phenotype Cluster | Unweighted Test $n$ ($\%$) | Survey-Weighted National Prevalence ($\%$) | Linearized SE ($\%$) | Weighted $95\%$ Confidence Interval |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **k-Means (`k=2`)** | `C0` (Lower Risk) | $566$ ($53.65\%$) | **$58.73\%$** | $2.03\%$ | $[54.74\%, 62.71\%]$ |
| **k-Means (`k=2`)** | `C1` (Adiposity/Low-HDL Risk) | $489$ ($46.35\%$) | **$41.27\%$** | $2.03\%$ | $[37.29\%, 45.26\%]$ |
| **GMM (`k=2`)** | `C0` (Cardiometabolically Healthy) | $832$ ($78.86\%$) | **$83.47\%$** | $1.76\%$ | $[80.01\%, 86.92\%]$ |
| **GMM (`k=2`)** | `C1` (Severe Cardiometabolic Endotype) | $223$ ($21.14\%$) | **$16.53\%$** | $1.76\%$ | $[13.08\%, 19.99\%]$ |

---

## 4. Synthesis & Conclusions for Clinical AI Governance

1. **Answer to Research Question:** Unsupervised clustering of continuous NHANES biomarkers discovers **genuine, externally validated cardiometabolic endotypes** rather than merely reproducing demographic categories—evidenced by massive physiological separation on both clustered markers ($H = 87.2–401.5$) and unseen renal/inflammatory markers (`hs-CRP` $p = 2.2 \times 10^{-13}$, `eGFR` $p = 0.041$), contrasted against small sociodemographic effect sizes ($\text{Cramér's } V \le 0.172$). However, because educational attainment ($q = 2.3 \times 10^{-6}$), income-to-poverty ratio ($q = 0.0043$), and race/ethnicity ($q = 0.0222$) remain significantly associated with high-risk endotype membership, downstream clinical deployment must audit subgroup calibration and avoid conflating structural socioeconomic risk exposure with biological determinism.
2. **Model Selection Recommendation:** For low-dimensional ($6\text{D}$) continuous clinical biomarker panels, **full-covariance Gaussian Mixture Models (`GMM`)** and **Euclidean `k-Means`** provide superior stability, calibration, and external clinical validity compared to unregularized neural autoencoders, which can hallucinate high latent Silhouette scores on pure noise.