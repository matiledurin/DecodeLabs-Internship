# Project 3 — Financial Transaction Fraud Detection

**Author:** Matile Durin Mohwaduba  
**Organisation:** DecodeLabs Data Science 
**Type:** Exploratory Data Analysis · Feature Engineering · Binary Classification · Fraud Detection  

---

## Business Problem

Financial fraud is one of the most damaging and costly threats facing banks, payment processors, and their customers. Every fraudulent transaction represents a direct financial loss — and a breach of the trust customers place in financial institutions to keep their money safe.

This project analyses a real-world financial transactions dataset to answer two connected business questions:

> **1. What patterns exist in suspicious financial transactions — and can we identify them from transaction behaviour alone?**  
> **2. Can we build a machine learning model that automatically flags suspicious transactions before they cause harm?**

**Why automated detection matters:**
- For financial institutions: Automated detection catches threats at scale that human review cannot
- For customers: Suspicious transactions can be blocked or flagged in real time before money is lost
- For regulators: A documented, auditable flagging system demonstrates compliance with financial security standards

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | Kaggle — `saidaminsaidaxmadov/financial-transactions` (`FactTransaction.csv`) |
| Raw records | 10,000+ transaction records |
| Successful transactions analysed | 9,533 |
| Suspicious transactions flagged | 3,760 (39.4% of successful transactions) |
| Target Variable | `Suspicious_Flag` (engineered — 1 = suspicious, 0 = safe) |

**Feature breakdown:**

| Column | Description |
|--------|-------------|
| `TransactionID` | Unique identifier for each transaction |
| `AccountID` | The account that initiated the transaction |
| `TransactionDate` | Date the transaction occurred |
| `TransactionType` | Credit or Debit |
| `TransactionChannel` | How the transaction was made (Web, ATM, Mobile) |
| `ProductID` | Financial product associated with the transaction |
| `Status` | Whether the transaction succeeded or failed |
| `TransactionAmount` | Raw amount — including negative values for suspicious transactions |

**Engineered features:**

| Column | Description |
|--------|-------------|
| `Suspicious_Flag` | **Target variable** — 1 if `TransactionAmount < 0`, else 0 |
| `True_Amount` | Absolute value of `TransactionAmount` — standardised for modelling |

---

## Analytical Pipeline

| Step | Task | Status |
|------|------|--------|
| 1 | Data Collection & Dataset Understanding | ✅ Complete |
| 2 | Data Cleaning & Preprocessing | ✅ Complete |
| 3 | Exploratory Data Analysis (EDA) | ✅ Complete |
| 4 | Data Visualisation — Security Dashboard | ✅ Complete |
| 5 | Predictive Modelling — Fraud Detection Classifier | ✅ Complete |

---

## Key Findings

**Suspicious rate:** 39.4% of all successful transactions carry a suspicious flag — nearly 4 in every 10 completed transactions are structurally anomalous. This is substantially higher than real-world fraud rates (typically under 1%), reflecting that this dataset is structured for anomaly analysis rather than live production modelling.

**The fraud signal:** Suspicious transactions were defined by negative `TransactionAmount` values on successful transactions. In standard double-entry bookkeeping, a successful transaction should always carry a positive monetary value in one direction. Negative values on completed transactions represent an accounting rule violation — the engineered fraud signal.

**Amount distribution:** Both safe and suspicious transactions span the full amount range ($11.07 to $10,000.00). Suspicious transactions are not simply high-value outliers — they appear at all transaction sizes. The model cannot rely on amount alone; it must learn from the combination of amount, channel, and transaction type.

---

## Feature Engineering

The most analytically critical step in this project:

```python
# Flag suspicious transactions (negative amounts = suspicious)
df_success['Suspicious_Flag'] = (df_success['TransactionAmount'] < 0).astype(int)

# Standardise amounts to absolute values for modelling
df_success['True_Amount'] = df_success['TransactionAmount'].abs()

# Drop the original column — True_Amount replaces it
df_clean_money = df_success.drop('TransactionAmount', axis=1)
```

---

## Model Performance

| Metric | Result |
|--------|--------|
| Model | Random Forest Classifier |
| Overall Accuracy | High (see notebook for exact figure) |
| Key metrics | Precision, Recall, F1-Score per class |
| Training split | 80% train / 20% test |
| Features used | `True_Amount`, `TransactionType`, `TransactionChannel`, `ProductID` |
| Dropped features | `TransactionID`, `AccountID`, `TransactionDate` (identifiers — no pattern value) |

### Why accuracy alone is not enough for fraud detection

In fraud detection, **precision** and **recall** matter more than overall accuracy:

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Precision** | Of all flagged transactions, how many were truly suspicious? | Low precision = too many false alarms, wasting investigator time |
| **Recall** | Of all truly suspicious transactions, how many were caught? | Low recall = real fraud slipping through undetected |

A model that flags everything as suspicious achieves 100% recall — but 0% usefulness. Both metrics must be high.

### Why Random Forest?
- Handles mixed numeric and categorical features effectively after encoding
- Robust to class imbalance without requiring resampling
- Captures complex, non-linear decision boundaries
- Provides interpretable classification report with per-class metrics

---

## Data Cleaning Summary

| Issue | Finding | Action Taken |
|-------|---------|--------------|
| Duplicate records | 0 duplicates found | No action required |
| Failed transactions | Non-Success records present | Filtered to Success only (9,533 retained) |
| Negative amounts | 3,760 transactions with negative `TransactionAmount` | Engineered as `Suspicious_Flag = 1` |
| Original amount column | Mixed positive/negative — confusing for model | Replaced with `True_Amount` (absolute value) |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `kagglehub` | Programmatic Kaggle dataset download |
| `pandas` | Data loading, filtering, feature engineering |
| `os` | File path navigation for Kaggle download folder |
| `seaborn` | Count plot and scatter plot visualisations |
| `matplotlib` | Figure layout and rendering |
| `sklearn.ensemble` | `RandomForestClassifier` — fraud detection model |
| `sklearn.model_selection` | `train_test_split` — 80/20 data split |
| `sklearn.metrics` | `accuracy_score`, `classification_report` — model evaluation |

---

## Files

```
├── Matile_Project3_Financial_Fraud_Narrated.ipynb   # Main notebook
└── Cleaned_Financial_Transactions.csv               # Cleaned & feature-engineered dataset
```

---

## How to Run

1. Open the notebook in **Jupyter Notebook**, **JupyterLab**, or **Google Colab**
2. Ensure you have a Kaggle API key configured (required for `kagglehub` download)
3. Run all cells in order
4. The cleaned, feature-engineered dataset is exported as `Cleaned_Financial_Transactions.csv`

**Dependencies:**
```bash
pip install pandas seaborn matplotlib scikit-learn kagglehub
```

**Kaggle API setup (if needed):**
```bash
# Place kaggle.json in ~/.kaggle/ or configure via environment variables
# See: https://github.com/Kaggle/kaggle-api
```

---

## Important Caveats

This model should be treated as a **proof-of-concept**, not a production-ready fraud detector. Real-world fraud detection requires:
- Continuous retraining as fraud patterns evolve over time
- Additional features: device fingerprinting, geolocation, behavioural velocity, transaction history
- Human review of model flags before any action is taken on a customer account
- Significantly lower suspicious rate (real fraud is typically < 1% of transactions)

---

## Author

**Matile Durin Mohwaduba**  
Founder, Durin Digital | Data Science Intern, DecodeLabs  
Johannesburg, South Africa
