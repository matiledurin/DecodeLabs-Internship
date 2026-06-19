# Project 2 — Medical Insurance Cost Analysis & Predictive Modelling

**Author:** Matile Durin Mohwaduba  
**Organisation:** DecodeLabs Data Science  
**Type:** Exploratory Data Analysis · Regression · Predictive Modelling  

---

## Business Problem

Medical insurance pricing is not random. Insurers use a combination of demographic and lifestyle factors to estimate how much a customer is likely to cost them in healthcare claims — and price their premiums accordingly.

This project analyses a real-world medical insurance dataset to answer two connected business questions:

> **1. Which factors most significantly drive the cost of medical insurance charges?**  
> **2. Can we build a model that accurately predicts what a new customer's insurance charges will be, based on their profile?**

**Why this matters:**
- For insurers: More accurate pricing means fewer underpriced high-risk customers
- For policymakers: Understanding cost drivers enables better public health interventions
- For individuals: Knowing which habits (like smoking) compound financial risk creates actionable incentive

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | Machine Learning with R Datasets (public GitHub repository) |
| Records | 1,338 raw records (1,337 after removing 1 duplicate) |
| Features | 7 columns |
| Target Variable | `charges` — annual medical insurance billed (USD) |

**Feature breakdown:**

| Column | Description |
|--------|-------------|
| `age` | Age of the primary insurance beneficiary |
| `sex` | Gender of the beneficiary |
| `bmi` | Body Mass Index |
| `children` | Number of dependents covered |
| `smoker` | Whether the beneficiary smokes (yes/no) |
| `region` | Geographic region in the USA |
| `charges` | Annual medical insurance charges — **target variable** |

---

## Analytical Pipeline

| Step | Task | Status |
|------|------|--------|
| 1 | Data Collection & Dataset Understanding | ✅ Complete |
| 2 | Data Cleaning & Preprocessing | ✅ Complete |
| 3 | Exploratory Data Analysis (EDA) | ✅ Complete |
| 4 | Data Visualisation | ✅ Complete |
| 5 | Predictive Modelling | ✅ Complete |

---

## Key Findings

### The Financial Cost of Smoking

| Smoker Status | Average Annual Charges |
|---------------|----------------------|
| Non-smoker | $8,440 |
| Smoker | $32,050 |
| **Difference** | **3.8x more expensive** |

Smoking is the single largest cost driver in this dataset by a significant margin. It is not a small surcharge — it represents a fundamentally different pricing tier.

### Age Compounds the Smoking Penalty

The scatter plot reveals three distinct pricing bands:
- **Lower band:** Non-smokers of all ages — charges rise gradually with age but remain relatively contained
- **Middle band:** Older non-smokers with additional risk factors (likely high BMI)
- **Upper band:** Smokers — charges start high in young adulthood and compound dramatically with age

**Key insight:** Smoking does not just add a flat premium — it interacts with age to produce exponentially higher charges over a lifetime. A 60-year-old smoker faces charges that dwarf those of a 60-year-old non-smoker by a factor far exceeding the simple average difference.

---

## Model Performance

| Metric | Result |
|--------|--------|
| Model | Random Forest Regressor |
| R² Score | 86% |
| Mean Absolute Error | $2,500 |
| Training split | 80% train / 20% test |
| Random state | 42 |

**Interpretation:**
- **R² of ~86%** means the model explains 86% of the variation in insurance charges across customers. The remaining 14% is driven by factors not in this dataset (individual medical history, specific diagnoses).
- **MAE of ~$2,500** means predictions are within $2,500 of actual charges on average — strong precision given charges range from ~$1,100 to ~$63,770.

### Why Random Forest?
- Handles mixed numeric and categorical features effectively
- Robust to the outliers present in the `charges` column for high-risk smokers
- Captures the non-linear age-smoking interaction identified in EDA
- Provides feature importance scores for interpretability

---

## Data Cleaning Summary

| Issue | Finding | Action Taken |
|-------|---------|--------------|
| Duplicate rows | 1 exact duplicate found | Removed — 0.07% of data |
| Missing values | Zero nulls across all 7 columns | No action required |
| Structural errors | All categorical columns clean (no typos, no mixed casing) | No action required |
| Logical errors | Minimum age = 18, minimum charge > $0 | No action required |

**Records retained after cleaning:** 1,337 / 1,338 (99.93%)

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data loading, cleaning, groupby analysis |
| `seaborn` | Bar chart and scatter plot visualisations |
| `matplotlib` | Figure layout and rendering |
| `sklearn.ensemble` | `RandomForestRegressor` — predictive model |
| `sklearn.model_selection` | `train_test_split` — 80/20 data split |
| `sklearn.metrics` | `r2_score`, `mean_absolute_error` — model evaluation |

---

## Files

```
├── Matile_Project2_Predictive_Modeling_Narrated.ipynb   # Main notebook
└── Cleaned_Medical_Insurance.csv                         # Cleaned dataset output
```

---

## How to Run

1. Open the notebook in **Jupyter Notebook**, **JupyterLab**, or **Google Colab**
2. Run all cells in order — the dataset loads directly from a public GitHub URL (no local file needed)
3. The cleaned dataset is exported automatically as `Cleaned_Medical_Insurance.csv`

**Dependencies:**
```bash
pip install pandas seaborn matplotlib scikit-learn
```

---

## Limitations

- Dataset contains only 1,337 records — a larger dataset would improve model stability
- No access to medical history, which in reality is a primary insurance pricing factor
- Model predictions should be treated as estimates, not clinical or financial guarantees
- Dataset is US-specific — regional pricing dynamics may differ in other markets

---

## Author

**Matile Durin Mohwaduba**  
Founder, Durin Digital | Data Science Intern, DecodeLabs  
Johannesburg, South Africa
