# Project 4 — IMDB Sentiment Analysis (NLP)

**Author:** Matile Durin Mohwaduba  
**Organisation:** DecodeLabs Data Science 
**Type:** Natural Language Processing · Text Classification · Binary Sentiment Analysis  

---

## Business Problem

Can a machine read emotion?

This project teaches a model to classify movie reviews as positive or negative — using 50,000 real IMDB reviews, text preprocessing, TF-IDF vectorisation, and Naive Bayes classification.

> **Business question: Given a piece of text written by a customer, can we automatically determine whether their sentiment is positive or negative — without a human reading it?**

**Why this matters in the real world:**
- Netflix and streaming platforms use sentiment to understand which content audiences love
- Amazon flags negative product experiences before they go viral
- Banks detect fraud risk from the language in customer complaints
- Brands monitor social media sentiment at scale — thousands of posts per minute
- Customer support systems automatically triage complaints by emotional urgency

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | IMDB Movie Reviews Dataset (Kaggle) |
| Total records | 50,000 reviews |
| Class balance | 25,000 positive / 25,000 negative — perfectly balanced |
| Target variable | `sentiment` (positive / negative) |
| Training set | 40,000 reviews (80%) |
| Test set | 10,000 reviews (20%) |

This is an industry-standard benchmark dataset used by NLP researchers worldwide.

---

## Analytical Pipeline

| Step | Task | Status |
|------|------|--------|
| 1 | Data Loading & Exploration | ✅ Complete |
| 2 | Text Preprocessing | ✅ Complete |
| 3 | TF-IDF Vectorisation | ✅ Complete |
| 4 | Model Training | ✅ Complete |
| 5 | Evaluation & Comparison | ✅ Complete |

---

## Key Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | 87% |
| Precision | 87% |
| Recall | 87% |
| F1-Score | 87% |
| Test set size | 10,000 unseen reviews |

**Confusion matrix (approximate):**

| | Predicted Negative | Predicted Positive |
|--|--|--|
| **Actual Negative** | 4,300 ✅ True Negative | 700 ❌ False Positive |
| **Actual Positive** | 600 ❌ False Negative | 4,400 ✅ True Positive |

The model is well-balanced — it does not systematically favour one class over the other.

---

## Text Preprocessing Pipeline

Every decision in preprocessing was deliberate:

### 1. HTML tag removal
```python
re.sub(r'<.*?>', '', text)
```
IMDB reviews contain HTML formatting from the web scraping process. This removes it before any text analysis.

### 2. Lowercasing & punctuation removal
Ensures "Brilliant" and "brilliant" are treated as the same word.

### 3. Stopword removal — with a critical exception
Common words like "the", "and", "is" are removed. However, **negation words are deliberately kept:**
```python
# These words are EXCLUDED from the stopword removal list
negation_words = {'not', 'never', 'nor', 'no', 'none', 'neither', 'hardly', 'barely'}
```
**Why this matters:** "not good" is the opposite of "good". A model that removes "not" cannot distinguish between these two meanings. Keeping negation words is a precision engineering decision that directly improves accuracy.

### 4. Lemmatisation (not stemming)
```python
lemmatizer = WordNetLemmatizer()
# "running", "ran", "runs" → all become "run"
```
Lemmatisation uses grammatical rules to find the true root form of a word. Stemming is faster but cruder — it often produces non-words ("running" → "runn"). Lemmatisation is the more accurate choice.

---

## TF-IDF Vectorisation

```python
TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),   # unigrams + bigrams
    stop_words=None        # handled in preprocessing
)
```

**What TF-IDF does:** Measures how important a word is in one review relative to the entire dataset.
- Words that appear everywhere ("film", "movie") get **downweighted** — they carry no sentiment signal
- Rare, specific words ("breathtaking", "abysmal") get **upweighted** — they carry strong sentiment signal
- Bigrams (word pairs like "not good", "absolutely brilliant") capture context that single words miss

**Result:** Each review is transformed into a vector of 50,000 numbers — a mathematical representation of its language.

---

## Models Compared

| Model | Description | Notes |
|-------|-------------|-------|
| `MultinomialNB` | Standard Naive Bayes for text | Primary model |
| `ComplementNB` | Variant designed for text classification | Comparison model |

**Why Naive Bayes?**

Naive Bayes assumes all words are independent — which is not strictly true in language. Yet it achieves 87% accuracy on 50,000 real reviews. This demonstrates a core data science principle: **a simpler model with good preprocessing often outperforms a complex model with poor data preparation.**

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Matrix operations |
| `nltk` | Tokenisation, stopword lists, `WordNetLemmatizer` |
| `re` | Regular expressions for HTML removal and text cleaning |
| `sklearn.feature_extraction.text` | `TfidfVectorizer` — text to numeric conversion |
| `sklearn.naive_bayes` | `MultinomialNB`, `ComplementNB` — classifiers |
| `sklearn.model_selection` | `train_test_split` — 80/20 stratified split |
| `sklearn.metrics` | `classification_report`, confusion matrix |
| `matplotlib` | Visualisation — sentiment distribution, TF-IDF feature chart |
| `seaborn` | Visualisation — review length histogram, confidence distribution |
| `kagglehub` | Kaggle dataset download |

---

## Files

```
Project4-Sentiment-Analysis/
├── README.md                              # Project documentation (this file)
├── PROJECT_4_DECODELABS_MATILE_.ipynb    # Main notebook
├── IMDB_Dataset_Sample_1000.csv          # Sample dataset — 500 positive + 500 negative reviews
└── project4_presentation.html            # Interactive presentation with live sentiment demo
```

---

## Dataset

| | |
|---|---|
| **Full dataset** | 50,000 reviews — [Download from Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| **Sample included** | `IMDB_Dataset_Sample_1000.csv` — 1,000 reviews (500 positive, 500 negative) uploaded directly to this repository |

> The full 66MB dataset exceeds GitHub's file size limit so it is hosted on Kaggle. The 1,000 row sample included here lets you explore the exact data structure and format without any downloads. To run the full notebook, download the complete dataset from Kaggle and place it in the same folder as the notebook.

---

## How to Run

1. Download all files from this folder to your computer
2. Download the full dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place `IMDB_Dataset.csv` in the same folder
3. Open `PROJECT_4_DECODELABS_MATILE_.ipynb` in **Jupyter Notebook**, **JupyterLab**, or **Google Colab**
4. Run the NLTK downloads cell first (required on first run):
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```
5. Run all cells in order

**Dependencies:**
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn
```

> **Want to see the project without running any code?** Open `project4_presentation.html` in any web browser — Chrome, Firefox, or Edge. It includes an interactive demo where you can type any text and see the sentiment model respond in real time.

---

## Real-World Applications of This Pipeline

| Industry | Application |
|----------|-------------|
| Retail & E-commerce | Automatic product review classification at scale |
| Finance | Customer complaint sentiment for risk triage |
| Healthcare | Patient feedback analysis for service improvement |
| Marketing | Brand sentiment monitoring across social media |
| HR | Employee survey sentiment for attrition signals |
| Media | Content recommendation driven by audience reaction signals |

---

## Limitations

- Naive Bayes treats all words as independent — it cannot model complex syntactic relationships
- Sarcasm and irony remain difficult for any bag-of-words model to detect
- The model is trained on English movie reviews — performance may degrade on other domains or languages
- State-of-the-art models (BERT, RoBERTa) achieve 95% on this benchmark vs 87% for this approach — the gap represents the cost of simplicity and the benefit of deeper architectures

---

## Author

**Matile Durin Mohwaduba**  
Founder, Durin Digital | Data Science Intern, DecodeLabs  
Johannesburg, South Africa
