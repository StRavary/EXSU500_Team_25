# ==============================================================================
# SCRIPT: data_curation_pipeline.R
# ROLE: Data Curation & Metadata Management (CRediT Taxonomy)
# AUTHOR: [Your Name/Title, Data Lead]
# SOURCE DATA: CDC NHANES 2017–March 2020 Pre-Pandemic Cycle
# TARGET FILE: nhanes_phenotype_analytical_matrix.csv / .rds
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. ENVIRONMENT SETTINGS & ENVIRONMENT CAPTURE (Version Control & Reusability)
# ------------------------------------------------------------------------------
if(!require(nhanesA)) install.packages("nhanesA")
library(nhanesA)

# Ensure data types are kept clean during download
options(stringsAsFactors = FALSE)

# ------------------------------------------------------------------------------
# 2. DATA ACQUISITION & AGGREGATION (Diverse Formats from Diverse Sources)
# ------------------------------------------------------------------------------
message("Downloading NHANES pre-pandemic data components from CDC...")

demo_raw   <- nhanes("P_DEMO")     # Demographics & Socioeconomic Status
bmx_raw    <- nhanes("P_BMX")      # Anthropometrics (BMI)
ghb_raw    <- nhanes("P_GHB")      # Laboratory: Glycohemoglobin (HbA1c)
biopro_raw <- nhanes("P_BIOPRO")   # Laboratory: Biochemistry (Serum Creatinine)
tchol_raw  <- nhanes("P_TCHOL")    # Laboratory: Total Cholesterol
hdl_raw    <- nhanes("P_HDL")      # Laboratory: HDL Cholesterol
trig_raw   <- nhanes("P_TRIGLY")   # Laboratory: Triglycerides & LDL Cholesterol
bp_raw     <- nhanes("P_BPXO")     # Examination: Blood Pressure readings
crp_raw    <- nhanes("P_HSCRP")    # Laboratory: High-sensitivity C-Reactive Protein

# ------------------------------------------------------------------------------
# 3. FEATURE ENGINEERING & DATA CLEANING (Scrubbing & Processing)
# ------------------------------------------------------------------------------
message("Executing variable transformation and feature engineering...")

# A. Standardize Blood Pressure: Aggregate 3 separate readings into row-wise means
bp_clean <- data.frame(SEQN = bp_raw$SEQN)
bp_clean$Mean_Systolic  <- rowMeans(bp_raw[, c("BPXOSY1", "BPXOSY2", "BPXOSY3")], na.rm = TRUE)
bp_clean$Mean_Diastolic <- rowMeans(bp_raw[, c("BPXODI1", "BPXODI2", "BPXODI3")], na.rm = TRUE)

# ------------------------------------------------------------------------------
# 4. RELATIONAL DATABASE JOINING (Integrating Matrix Components)
# ------------------------------------------------------------------------------
message("Merging relational database layers on primary token (SEQN)...")

# Sequential left-joins using Base R to preserve demographic master index boundary
final_dataset <- merge(demo_raw, ghb_raw, by = "SEQN", all.x = TRUE)
final_dataset <- merge(final_dataset, bmx_raw, by = "SEQN", all.x = TRUE)
final_dataset <- merge(final_dataset, biopro_raw, by = "SEQN", all.x = TRUE)
final_dataset <- merge(final_dataset, tchol_raw, by = "SEQN", all.x = TRUE)
final_dataset <- merge(final_dataset, hdl_raw, by = "SEQN", all.x = TRUE)
final_dataset <- merge(final_dataset, trig_raw, by = "SEQN", all.x = TRUE)
final_dataset <- merge(final_dataset, bp_clean, by = "SEQN", all.x = TRUE)
final_dataset <- merge(final_dataset, crp_raw, by = "SEQN", all.x = TRUE)

# B. Calculate eGFR: Implement 2021 CKD-EPI Creatinine Equation (Vectorized Math)
scr  <- final_dataset$LBXSCR
age  <- final_dataset$RIDAGEYR
sex_string <- final_dataset$RIAGENDR # Extracted text representation from nhanesA

# Convert categorical text labels back to binary math flags for the equation constants
is_female <- ifelse(sex_string == "Female", TRUE, FALSE)
kappa     <- ifelse(is_female, 0.7, 0.9)
alpha     <- ifelse(is_female, -0.241, -0.302)
gender_multiplier <- ifelse(is_female, 1.012, 1)

# Execute vectorized calculation
final_dataset$eGFR <- 142 * 
  ((pmin(scr / kappa, 1))^alpha) * 
  ((pmax(scr / kappa, 1))^-1.200) * 
  (0.9938^age) * 
  gender_multiplier

# ------------------------------------------------------------------------------
# 5. COLUMN SCHEMA ISOLATION & SEMANTIC ANNOTATION (Producing Metadata descriptions)
# ------------------------------------------------------------------------------
message("Subsetting schema and executing semantic header injection...")

variables_to_keep <- c(
  "SEQN", "LBXGH", "BMXBMI", "eGFR", "LBXTC", "LBDHDD", "LBXTR", "LBDLDL", 
  "Mean_Systolic", "Mean_Diastolic", "LBXHSCRP", "RIDAGEYR", "RIAGENDR", 
  "RIDRETH3", "INDFMPIR", "DMDEDUC2"
)

# Isolate matrix subset
analytical_matrix <- final_dataset[, variables_to_keep]

# Replace CDC codes with descriptive, clean headers
names(analytical_matrix) <- c(
  "ID", "HbA1c_Percent", "BMI", "eGFR", "Total_Cholesterol", "HDL_Cholesterol", 
  "Triglycerides", "LDL_Cholesterol", "Systolic_BP", "Diastolic_BP", 
  "Inflammation_CRP", "Age_Years", "Gender", "Race_Ethnicity", 
  "Income_Poverty_Ratio", "Education_Level"
)

# ------------------------------------------------------------------------------
# 6. ARCHIVING & DATA PRESERVATION EXPORT (Findable, Accessible, Reusable, Interoperable)
# ------------------------------------------------------------------------------
message("Writing data assets and metadata manifests to storage...")

# Export primary flat CSV package (Superficial storage, high interoperability)
write.csv(analytical_matrix, "nhanes_phenotype_analytical_matrix.csv", row.names = FALSE)

# Export native R binary artifact (Preserves precise metadata structures and types natively)
saveRDS(analytical_matrix, file = "nhanes_phenotype_analytical_matrix.rds")

# Generate Automated Metadata Manifest and Profiling Report (Retention Log)
sink("metadata_data_curation_manifest.txt")
cat("========================================================================\n")
cat("NHANES ANNOTATED DATA CURATION MANIFEST LOG\n")
cat("Generated:", as.character(Sys.time()), "\n")
cat("========================================================================\n\n")
cat("--- ARCHIVE REPLICABILITY ENGINE DETAILS ---\n")
print(version)
cat("\nPackage nhanesA version compiled:", as.character(packageVersion("nhanesA")), "\n")
cat("\n--- PIPELINE TARGET MISSINGNESS ACCOUNTING (Retention Audit) ---\n")
print(colSums(is.na(analytical_matrix)))
cat("\n--- SAMPLE DATA BOUNDS PROFILE (Scrubbing Validation) ---\n")
print(summary(analytical_matrix[, c("BMI", "HbA1c_Percent", "eGFR", "Systolic_BP")]))
sink()

message("Pipeline completed successfully! Curation files exported to work directory.")
# ==============================================================================
