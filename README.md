# 🏥 Medical Transcription NLP Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-3.x-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

**An end-to-end NLP pipeline for automated medical specialty classification from clinical transcriptions — combining rule-based NLP, entity extraction, negation detection, and TF-IDF + Logistic Regression classification with GridSearchCV tuning.**

[📓 Notebook](#-project-structure) • [🚀 Quick Start](#-quick-start) • [📊 Results](#-results) • [🧠 Methodology](#-methodology) • [📚 Dataset](#-dataset)

</div>

---

## 📌 Overview

This project tackles the real-world problem of **automated clinical triage** — routing incoming medical transcriptions to the correct specialist department without human intervention. Using the **MTSamples** dataset from Kaggle, the pipeline classifies transcriptions into one of four medical specialties:

| Specialty | Description |
|-----------|-------------|
| 🫀 **Cardiovascular / Pulmonary** | Heart & lung conditions (MI, CHF, COPD) |
| 🧠 **Neurology** | Neurological disorders (stroke, MS, TIA) |
| 🦴 **Orthopedic** | Musculoskeletal injuries & surgeries |
| 🫃 **Gastroenterology** | GI tract disorders (GERD, IBD, colonoscopy) |

---

## 🗂️ Project Structure

```
nlp-medical-classification/
├── NLP_project_1242.ipynb          # Main Jupyter notebook (all 4 phases)
├── NLP_Project_Report_23BLC1242.pdf # Detailed project report
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Files excluded from version control
├── README.md                        # This file
├── METHODOLOGY.md                   # In-depth technical explanation
└── RESULTS.md                       # Detailed evaluation results & analysis
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/nlp-medical-classification.git
cd nlp-medical-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Download the Dataset (Kaggle API)
```bash
# Place your kaggle.json credentials file in the project root
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d tboyle10/medicaltranscriptions
unzip medicaltranscriptions.zip
```

### 4. Run the Notebook
```bash
jupyter notebook NLP_project_1242.ipynb
```

> **Google Colab users:** Open the notebook directly in Colab — it will prompt you to upload `kaggle.json` automatically.

---

## 📊 Results

### Final Model Performance (Tuned Logistic Regression)

| Metric | Score |
|--------|-------|
| **Accuracy** | ~85–88% |
| **Macro F1** | ~0.84–0.87 |
| **Weighted F1** | ~0.85–0.88 |
| **5-Fold CV Macro F1** | Stable across folds |

### Per-Specialty Breakdown

| Specialty | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Cardiovascular / Pulmonary | High | High | High |
| Gastroenterology | High | High | High |
| Neurology | Moderate–High | Moderate–High | Moderate–High |
| Orthopedic | High | High | High |

> 📄 See [`RESULTS.md`](RESULTS.md) for the complete classification report, confusion matrices, and error analysis.

---

## 🧠 Methodology

The pipeline is structured into **4 distinct phases**:

### Phase 1 — Data Loading & Preprocessing
- Load **MTSamples** CSV; filter to 4 target specialties
- Custom **abbreviation expansion** dictionary (24 medical abbreviations: `pt→patient`, `hx→history`, `dm→diabetes`, etc.)
- **Header removal** (stripping boilerplate fields: `Sample Name`, `ID`, `Date`, `Dictated By`)
- **spaCy-based preprocessing**: lemmatization, stopword removal, punctuation filtering, number stripping

### Phase 2 — Entity Extraction & Vocabulary Analysis
- POS-tag-based extraction of **domain nouns** and **adjectives** per specialty
- **Domain stopword filtering** (generic clinical terms suppressed)
- **Word cloud generation** per specialty for visual vocabulary comparison
- Frequency analysis: Top-N discriminative terms per class

### Phase 3 — Negation Detection
- Rule-based **negation detection** using dependency parsing
- Identifies negation triggers: `no`, `not`, `without`, `deny`, `denies`, `denied`
- Dependency tree traversal to determine scope of negation on clinical entities

### Phase 4 — Classification
- **TF-IDF Vectorisation** with domain stopwords, sublinear TF scaling, unigrams + bigrams
- Baseline: Logistic Regression, Naive Bayes, Linear SVC
- **GridSearchCV** hyperparameter tuning (5-fold stratified CV, macro F1 scoring)
- Best params search over: `ngram_range`, `max_features`, regularisation strength `C`
- Evaluation: Confusion matrix (raw + normalised), classification report, per-feature coefficients
- **Error Analysis**: Qualitative inspection of misclassified examples
- **Triage Demo**: End-to-end inference on unseen clinical notes

---

## 📚 Dataset

| Property | Value |
|----------|-------|
| **Name** | MTSamples — Medical Transcription Samples |
| **Source** | [Kaggle — tboyle10](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) |
| **Size** | ~5,000 clinical transcriptions across 40 specialties |
| **Filtered Classes** | 4 specialties (~500–900 samples each) |
| **Split** | 80% train / 20% test (stratified) |
| **License** | CC0: Public Domain |

> ⚠️ The dataset file (`mtsamples.csv`) is **not included** in this repository due to size. Download it via the Kaggle API as described in [Quick Start](#-quick-start).

---

## 🛠️ Tech Stack

| Component | Library / Tool |
|-----------|---------------|
| Language | Python 3.8+ |
| NLP Processing | spaCy (`en_core_web_sm`) |
| Feature Engineering | scikit-learn `TfidfVectorizer` |
| Classification | scikit-learn `LogisticRegression` |
| Hyperparameter Tuning | `GridSearchCV` + `StratifiedKFold` |
| Visualisation | Matplotlib, Seaborn, WordCloud |
| Data Handling | Pandas, NumPy |
| Environment | Google Colab / Jupyter Notebook |

---

## 📁 Key Design Decisions

- **Why Logistic Regression over deep learning?**  
  The dataset size (~500–900 samples per class) is too small for reliable fine-tuning of transformer models. LR with TF-IDF generalises better in this regime and is fully interpretable via coefficient analysis.

- **Why domain-specific stopwords?**  
  Generic clinical terms like `patient`, `procedure`, `history` appear across **all** specialties equally and hurt discrimination — they are removed before vectorisation.

- **Why sublinear TF scaling?**  
  Medical notes repeat the same clinical terms extensively. Log-scaling (`sublinear_tf=True`) prevents high-frequency but low-information terms from dominating feature weights.

- **Why stratified CV?**  
  Class imbalance between specialties requires stratified splits to ensure each fold is representative of the true class distribution.

---

## 👤 Author

| | |
|--|--|
| **Name** | Rishi |
| **Roll No.** | 23BLC1242 |
| **Course** | Natural Language Processing |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check the [issues page](../../issues).

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

</div>
