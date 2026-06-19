# Project 1 — Telco Customer Churn Analysis

**Author:** Matile Durin Mohwaduba  
**Organisation:** DecodeLabs Data Science  
**Type:** Exploratory Data Analysis · Business Intelligence  

---

## Business Problem

Customer churn — when a customer cancels their service — is one of the most costly problems a telecom company faces. Research consistently shows that acquiring a new customer costs **5 to 7 times more** than retaining an existing one.

This project analyses a real-world telecom dataset to answer a core business question:

> **Which customers are most likely to churn, and what factors are driving that behaviour?**

By understanding the patterns behind churn, a business can identify at-risk customers early and design targeted retention strategies — such as offering contract upgrades, personalised discounts, or proactive support outreach.

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | IBM Telco Customer Churn (public GitHub repository) |
| Records | 7,043 customer records (7,032 after cleaning) |
| Features | 21 columns |
| Target Variable | `Churn` (Yes / No) |

**Feature categories:**
- Demographics — gender, senior citizen status, dependents
- Account info — tenure, contract type, billing method
- Services — internet, phone, streaming, security add-ons
- Financials — monthly charges, total charges

---

## Analytical Pipeline

| Step | Task | Status |
|------|------|--------|
| 1 | Data Collection | ✅ Complete |
| 2 | Data Inspection | ✅ Complete |
| 3 | Data Cleaning | ✅ Complete |
| 4 | Exploratory Data Analysis (EDA) | ✅ Complete |
| 5 | Predictive Modelling | 🔄 Planned |

---

## Key Findings

**Churn distribution:** 73% of customers stayed, 27% churned. The dataset is moderately imbalanced — important for future modelling decisions.

**Contract type is the strongest churn driver identified:** Month-to-month customers churn at a dramatically higher rate than customers on one or two-year contracts. Two-year contract customers show near-negligible churn.

**Business implication:** A targeted retention strategy should focus on converting month-to-month customers to longer-term contracts through incentives — for example, offering a discounted annual plan to customers who have been on month-to-month billing for more than 6 months without churning.

**Financial profile:**
- Average tenure: ~32 months (range: 1–72 months)
- Average monthly charges: ~$64.80 (range: $18.25–$118.75)
- ~16% of the customer base are senior citizens

---

## Data Cleaning Summary

| Issue | Finding | Action Taken |
|-------|---------|--------------|
| `TotalCharges` data type | Stored as string due to blank spaces for new customers | Converted to float using `pd.to_numeric(errors='coerce')` |
| Missing values | 11 nulls in `TotalCharges` (customers with `tenure = 0`) | Dropped — 0.16% of data, negligible loss |
| Duplicate records | 0 duplicates found | No action required |
| `SeniorCitizen` encoding | Integer (0/1) vs Yes/No strings in other columns | Documented — consistent within column |

**Records retained after cleaning:** 7,032 / 7,043 (99.84%)

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data loading, inspection, cleaning, and analysis |
| `seaborn` | Visualisation — count plots and distribution charts |
| `matplotlib` | Figure rendering and layout |

---

## Files

```
├── DecodeLabs_-_Project_1_Telco_Churn_Narrated.ipynb   # Main notebook
└── Cleaned_Telco_Churn.csv                              # Cleaned dataset output
```

---

## How to Run

1. Open the notebook in **Jupyter Notebook**, **JupyterLab**, or **Google Colab**
2. Run all cells in order — the dataset loads directly from IBM's public GitHub URL (no local file needed)
3. The cleaned dataset is exported automatically as `Cleaned_Telco_Churn.csv`

**Dependencies:**
```bash
pip install pandas seaborn matplotlib
```

---

## Planned Next Steps

- Feature encoding (Label Encoding / One-Hot Encoding for categorical columns)
- Class imbalance handling using SMOTE or class weighting
- Baseline model: Logistic Regression
- Comparison model: Random Forest Classifier
- Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- Feature importance ranking to identify the top churn drivers

---

## Author

**Matile Durin Mohwaduba**  
Founder, Durin Digital | Data Science Intern, DecodeLabs  
Johannesburg, South Africa
