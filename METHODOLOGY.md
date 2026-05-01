# 🧠 Methodology — Deep Dive

This document provides an in-depth technical explanation of each phase in the NLP pipeline.

---

## Table of Contents

1. [Phase 1 — Preprocessing Pipeline](#phase-1--preprocessing-pipeline)
2. [Phase 2 — Entity Extraction & Vocabulary Analysis](#phase-2--entity-extraction--vocabulary-analysis)
3. [Phase 3 — Negation Detection](#phase-3--negation-detection)
4. [Phase 4 — Classification Pipeline](#phase-4--classification-pipeline)
5. [Design Rationale](#design-rationale)

---

## Phase 1 — Preprocessing Pipeline

### 1.1 Dataset Loading

The raw dataset (`mtsamples.csv`) is loaded with Pandas. Missing `transcription` entries are dropped, and whitespace is stripped from the `medical_specialty` field. The full dataset spans **~40 specialties**; this project filters to **4 clinically distinct** classes:

- `Gastroenterology`
- `Cardiovascular / Pulmonary`
- `Orthopedic`
- `Neurology`

### 1.2 Preprocessing Function — `preprocess_text()`

The full preprocessing pipeline executes in a strictly ordered sequence:

```
Raw Text
   │
   ▼
[1] remove_headers()        — Strip boilerplate (Sample Name, ID, Date, Dictated By)
   │
   ▼
[2] expand_abbreviations()  — Expand 24 clinical abbreviations AFTER lowercasing
   │
   ▼
[3] re.sub(r'\d+', '', …)   — Remove all numeric tokens
   │
   ▼
[4] spaCy pipeline           — Tokenise, POS-tag, parse dependencies
   │
   ▼
[5] Filter tokens            — Keep: not stop-word, not punctuation, alpha-only
   │
   ▼
[6] Lemmatize                — Reduce to base form (e.g., "fractures" → "fracture")
   │
   ▼
Clean Text (space-joined lemmas)
```

> **Order matters:** Abbreviation expansion is applied **after** lowercasing to ensure matching (e.g., `PT` → `pt` → `patient`).

### 1.3 Abbreviation Dictionary

24 common clinical shorthand terms are expanded:

| Abbreviation | Expansion |
|---|---|
| `pt` | patient |
| `hx` | history |
| `dx` | diagnosis |
| `fx` | fracture |
| `sx` | symptom |
| `bp` | blood pressure |
| `htn` | hypertension |
| `dm` | diabetes |
| `sob` | shortness of breath |
| `gi` | gastrointestinal |
| `gerd` | gastroesophageal reflux disease |
| `ibd` | inflammatory bowel disease |
| `egd` | esophagogastroduodenoscopy |
| `ekg` | electrocardiogram |
| `mi` | myocardial infarction |
| `chf` | congestive heart failure |
| `copd` | chronic obstructive pulmonary disease |
| `tia` | transient ischemic attack |
| `cva` | cerebrovascular accident |
| `ms` | multiple sclerosis |
| `rom` | range of motion |
| `oa` | osteoarthritis |
| `ra` | rheumatoid arthritis |

---

## Phase 2 — Entity Extraction & Vocabulary Analysis

### 2.1 POS-Based Entity Extraction

spaCy's `en_core_web_sm` model performs Part-of-Speech tagging. For each cleaned transcription:

- **Nouns** (`token.pos_ == "NOUN"`) → captures clinical objects (organs, conditions, procedures)
- **Adjectives** (`token.pos_ == "ADJ"`) → captures clinical descriptors (severity, quality)

All tokens are stored in their **lemmatized** form to unify surface variants.

### 2.2 Domain Stopword Filtering

A custom medical stopword set suppresses 16 high-frequency but non-discriminative terms that appear proportionally across **all** specialties:

```python
medical_stopwords = {
    "patient", "place", "procedure", "note", "room", "time",
    "study", "report", "exam", "history", "assessment", "plan",
    "finding", "result", "follow", "year", "day", "week",
    "discharge", "admission", "record", "documentation", "review"
}
```

### 2.3 Word Cloud Generation

Per-specialty word clouds are generated from filtered noun frequencies to visually confirm that the vocabulary is clinically discriminative.

### 2.4 Top Discriminative Vocabulary

A frequency analysis using `collections.Counter` identifies the top-N nouns per specialty, confirming that the preprocessing captures domain-specific terminology.

---

## Phase 3 — Negation Detection

### 3.1 Motivation

Clinical negation is critical for medical NLP: "**no** chest pain" and "chest pain" convey opposite clinical findings. Mishandling negation can catastrophically mislabel conditions.

### 3.2 Implementation

Negation detection uses **dependency parsing** (spaCy's syntactic parse tree) rather than simple keyword window matching, which prevents false positives from sentences like "History of **no** prior surgeries — patient has chest pain."

```
Negation Triggers: {"no", "not", "without", "deny", "denies", "denied"}
```

**Algorithm:**
1. For each `NOUN` or `PROPN` token, examine its **head verb's children** for negation triggers.
2. Alternatively, if the noun is in a subject/object dependency role (`nsubj`, `dobj`, `attr`) and its head verb **is** a negation trigger, mark it negated.

This captures:
- `"denies chest pain"` → `pain` is negated (head verb = "denies" ∈ triggers)
- `"no evidence of fracture"` → `fracture` is negated (sibling child "no" ∈ triggers)

---

## Phase 4 — Classification Pipeline

### 4.1 Feature Engineering — TF-IDF

TF-IDF vectorisation converts preprocessed text to a sparse numerical matrix:

```python
TfidfVectorizer(
    ngram_range=(1, 2),      # Unigrams + bigrams
    max_features=10_000,     # Top 10K most informative terms
    min_df=3,                # Prune hapax/near-hapax terms
    max_df=0.90,             # Prune near-universal terms
    sublinear_tf=True,       # Log-scale TF: tf = 1 + log(tf)
    analyzer="word",
    token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",  # Alpha tokens only
    stop_words=DOMAIN_STOPWORDS
)
```

**Key decisions:**

| Parameter | Value | Rationale |
|---|---|---|
| `sublinear_tf` | `True` | Medical notes repeat terms → log-scaling prevents dominance |
| `max_df=0.90` | 90% | Removes near-universal terms like "patient" |
| `min_df=3` | 3 docs | Removes rare jargon that only appears 1–2 times |
| `ngram_range=(1,2)` | Bigrams | Captures clinical phrases ("chest pain", "blood pressure") |

### 4.2 Classification Models Compared

| Model | Vectoriser | Notes |
|---|---|---|
| Logistic Regression (baseline) | TF-IDF | Multinomial, `lbfgs` solver |
| Naive Bayes | TF-IDF | Complement NB for imbalanced data |
| Linear SVC | TF-IDF | Margin-based, no calibration |
| **Logistic Regression + GridSearchCV** | TF-IDF | **Final model** |

### 4.3 Hyperparameter Tuning

GridSearchCV explores the following parameter space with **5-fold Stratified CV**, optimising **macro F1**:

```python
param_grid = {
    "tfidf__ngram_range"  : [(1, 1), (1, 2)],
    "tfidf__max_features" : [8000, 12000, 15000],
    "clf__C"              : [0.1, 1.0, 5.0, 10.0],
}
```

Total combinations: `2 × 3 × 4 = 24` configurations × 5 folds = **120 model fits**.

### 4.4 Evaluation Protocol

| Metric | Purpose |
|---|---|
| **Accuracy** | Overall correctness |
| **Macro F1** | Treats all classes equally, sensitive to minority class |
| **Weighted F1** | Accounts for class imbalance |
| **Confusion Matrix (raw)** | Per-class absolute error analysis |
| **Confusion Matrix (normalised)** | Per-class recall percentages |
| **Per-feature coefficients** | Model interpretability |
| **5-Fold CV** | Stability check — variance across folds |
| **Error Analysis** | Qualitative inspection of misclassifications |

### 4.5 End-to-End Triage Demo

The `triage_report()` function demonstrates the complete inference pipeline:

```
Raw Clinical Note (string)
        │
        ▼
  preprocess_text()          ← Phase 1 pipeline
        │
        ▼
  TF-IDF transform           ← Fitted vectoriser
        │
        ▼
  LR predict()               ← Calibrated classifier
        │
        ▼
  Specialty Label             ← e.g., "Cardiovascular / Pulmonary"
```

---

## Design Rationale

### Why not use a Transformer (BERT, ClinicalBERT)?

| Consideration | Logistic Regression | Transformer |
|---|---|---|
| Dataset size | ~500–900 / class ✅ | Needs 5K+ for fine-tuning |
| Inference speed | Milliseconds ✅ | Seconds (GPU needed) |
| Interpretability | Full (coefficients) ✅ | Black box ❌ |
| Compute requirements | CPU-only ✅ | GPU required |
| Performance gap | Closes significantly with TF-IDF tuning | Small gain |

For a **course project** with limited data and a focus on **classical NLP techniques**, LR + TF-IDF is the appropriate and educationally motivated choice.

### Why Macro F1 as the optimisation target?

Macro F1 weights each class equally regardless of size. Since the specialties are slightly imbalanced (Cardiovascular > Neurology in sample count), optimising accuracy alone could lead to a model that ignores minority classes. Macro F1 forces the model to perform well on **every** specialty.
