# Resume Parser AI — NLP Clustering & Scoring System

A machine learning project that uses Natural Language Processing and unsupervised clustering to automatically parse, group, and score resumes by job category — reducing the manual effort involved in candidate screening.

---

## 📌 Problem Statement

Recruiters spend enormous time manually screening hundreds of resumes for a single job posting. Traditional keyword-matching systems are easily gamed by candidates stuffing their resumes with buzzwords. This project builds a smarter, context-aware system that groups resumes by domain similarity and assigns a relevance score based on meaningful skill patterns.

---

## 📂 Dataset

**Source:** [Kaggle — Resume Dataset by Sneha Bhawal](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/data)

| Detail | Value |
|---|---|
| Total Resumes | 2,484 |
| Job Categories | 24 |
| License | CC0 (Public Domain) |
| Key Column Used | `Resume_str` — plain text resume content |

**Categories include:** HR, Information Technology, Business Development, Finance, Advocate, Chef, Engineering, Healthcare, Fitness, Aviation, Banking, Sales, Consultant, Construction, Public Relations, Designer, Arts, Teacher, Apparel, Digital Media, Agriculture, Automobile, Accountant, BPO

> ⚠️ **Class Imbalance Note:** BPO (22), Automobile (36), and Agriculture (63) are significantly underrepresented compared to IT and Business Development (120 each).

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **NLP:** NLTK (tokenization, stopword removal, Porter Stemming)
- **Feature Extraction:** Scikit-learn TF-IDF Vectorizer
- **Clustering:** K-Means (K=24)
- **Dimensionality Reduction:** PCA (for visualization)
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **Platform:** Kaggle Notebooks

---

## ⚙️ Project Pipeline

```
Raw Resume Text
      │
      ▼
 Text Cleaning (NLTK)
 - Remove URLs, special characters, digits
 - Lowercase conversion
 - Tokenization
 - Stopword removal
 - Porter Stemming
      │
      ▼
 TF-IDF Vectorization
 (2484 resumes × 1500 features)
      │
      ▼
 K-Means Clustering (K=24)
      │
      ▼
 Resume Scoring (0–10)
 based on category keyword match
```

---

## 📊 Results

### Clustering
| Metric | Value |
|---|---|
| TF-IDF Matrix Shape | (2484, 1500) |
| Silhouette Score | 0.054 |
| Number of Clusters | 24 |

The low silhouette score is expected for long-form resume text due to shared vocabulary across categories (e.g., "manage", "develop", "project"). Despite this, clusters showed strong semantic coherence — particularly for domain-specific roles:

| Cluster | Dominant Theme |
|---|---|
| Cluster 3 | Chef / Culinary (food, menu, kitchen, cook) |
| Cluster 5 | Teaching (student, classroom, lesson, school) |
| Cluster 18 | Aviation / Military (aircraft, army, logistics) |
| Cluster 19 | Healthcare (patient, nurse, clinic, physician) |
| Cluster 20 | Fitness (exercise, trainer, wellness) |
| Cluster 23 | Banking (loan, credit, mortgage, branch) |

### Scoring
| Metric | Value |
|---|---|
| Mean Score | 8.22 / 10 |
| Median Score | 8.5 / 10 |
| Min Score | 0.0 |
| Max Score | 10.0 |

Most resumes scored above 7.5, indicating strong alignment between resume content and their declared job category. Near-zero scores highlight generic or poorly targeted resumes.

---

## 🔍 Key Insights

- **Domain-specific categories** (Chef, Fitness, Healthcare, Aviation) clustered most cleanly due to unique vocabulary with minimal overlap.
- **Business-oriented roles** (Consultant, Business Development, Sales) showed higher overlap, as they share common terms like "client", "revenue", and "strategy".
- The **scoring function** successfully differentiates keyword-rich, targeted resumes from generic ones within the same category.
- **TF-IDF outperforms simple keyword matching** by weighting rare, domain-specific terms higher than common filler words.

---

## ▶️ How to Run

1. Open [Kaggle Notebook](https://www.kaggle.com/) and attach the dataset
2. Run cells in order

---

## ⚠️ Limitations

- Class imbalance affects clustering quality for underrepresented categories
- Stemming occasionally merges semantically different words
- Scoring is relative to category keywords — a perfect 10 means strong keyword coverage, not necessarily the "best" candidate
- No deep semantic understanding (word embeddings like Word2Vec/BERT could improve results)

---

## 🚀 Future Improvements

- Use **Word2Vec or BERT embeddings** for richer semantic representation
- Apply **DBSCAN or Hierarchical Clustering** as alternatives to K-Means
- Build a **web interface** where recruiters can upload a resume and receive an instant score and category prediction
- Address class imbalance using **SMOTE or data augmentation**

---

## 📜 License

Dataset: [CC0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)  
Project: MIT License
