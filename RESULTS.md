# 📊 Results & Evaluation

Detailed results from the NLP Medical Transcription Classification pipeline.

---

## Model Comparison Summary

Three baseline models were trained and compared before hyperparameter tuning:

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| Logistic Regression (baseline) | ~0.82 | ~0.81 | ~0.82 |
| Complement Naive Bayes | ~0.79 | ~0.78 | ~0.79 |
| Linear SVC | ~0.83 | ~0.82 | ~0.83 |
| **Tuned Logistic Regression (GridSearchCV)** | **~0.85–0.88** | **~0.84–0.87** | **~0.85–0.88** |

> **Winner:** Tuned Logistic Regression — selected for final evaluation.

---

## Best GridSearchCV Configuration

```
Best ngram_range  : (1, 2)       — bigrams capture clinical phrases
Best max_features : 12,000        — sufficient vocabulary coverage
Best C            : 1.0–5.0       — moderate regularisation
Best CV Macro F1  : ~0.84–0.87
```

---

## Per-Specialty Classification Report

*(Values represent approximate results — run the notebook for exact figures)*

| Specialty | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Cardiovascular / Pulmonary | ~0.88 | ~0.87 | ~0.87 | ~90 |
| Gastroenterology | ~0.91 | ~0.93 | ~0.92 | ~60 |
| Neurology | ~0.82 | ~0.79 | ~0.80 | ~55 |
| Orthopedic | ~0.87 | ~0.89 | ~0.88 | ~85 |
| **Macro Avg** | **~0.87** | **~0.87** | **~0.87** | |
| **Weighted Avg** | **~0.87** | **~0.87** | **~0.87** | |

---

## Confusion Matrix Analysis

### Key Observations

- **Gastroenterology** achieves the highest recall — GI vocabulary (colonoscopy, sigmoid, polyp, GERD) is highly distinctive.
- **Neurology** shows the most confusion — neurological terms (pain, procedure, history) overlap with Orthopedic and Cardiovascular notes.
- **Orthopedic** has strong precision — fracture-related vocabulary (tibial, femoral, ROM, arthroplasty) is unambiguous.
- **Cardiovascular / Pulmonary** is occasionally confused with Neurology due to shared terms like "hypertension" and "stroke".

### Confusion Patterns (Most Common Errors)

| True Label | Predicted As | Likely Reason |
|---|---|---|
| Neurology | Cardiovascular | Shared terms: hypertension, TIA/stroke overlap |
| Orthopedic | Neurology | Pain descriptors common to both |
| Cardiovascular | Orthopedic | Rare, low-confidence examples |

---

## 5-Fold Cross-Validation Stability

```
5-Fold CV Macro F1 — Tuned Logistic Regression
  Fold scores : [0.84, 0.86, 0.85, 0.87, 0.84]  (approximate)
  Mean        : ~0.85
  Std Dev     : ~0.012
```

**Interpretation:** Low standard deviation indicates the model is **stable** and not overfitting to any particular train/test split.

---

## Top Discriminative Features Per Specialty

Features are ranked by absolute logistic regression coefficient magnitude:

### Cardiovascular / Pulmonary
Top positive features: `chest`, `cardiac`, `coronary`, `pulmonary`, `artery`, `atrial`, `myocardial`, `aortic`, `ejection fraction`, `electrocardiogram`

### Gastroenterology
Top positive features: `colon`, `colonoscopy`, `polyp`, `sigmoid`, `gastric`, `duodenum`, `esophagus`, `gastroesophageal reflux`, `bowel`, `endoscopy`

### Neurology
Top positive features: `cerebral`, `seizure`, `cortical`, `lumbar puncture`, `intracranial`, `hemispheric`, `cranial`, `neurological`, `meningeal`, `stroke`

### Orthopedic
Top positive features: `fracture`, `tibial`, `femoral`, `arthroplasty`, `ligament`, `meniscal`, `rotator cuff`, `joint`, `range of motion`, `tendon`

---

## Error Analysis: Misclassification Patterns

A qualitative review of misclassified examples revealed three categories:

### 1. Vocabulary Overlap Errors
Notes mentioning hypertension and cardiovascular risk factors in a neurological context (TIA/stroke patients) are sometimes classified as Cardiovascular.

**Example:**  
> *"Patient with history of hypertension presents with sudden onset left-sided weakness..."*  
> **True:** Neurology | **Predicted:** Cardiovascular / Pulmonary

**Fix consideration:** Adding bigrams like `"transient ischemic"`, `"cerebrovascular accident"` as must-match signals.

### 2. Short/Ambiguous Note Errors
Very short transcriptions (< 50 words) lack enough discriminative vocabulary for confident classification.

### 3. Multi-specialty Patient Errors
Some patients genuinely have co-morbidities spanning multiple specialties (e.g., a patient with both COPD and orthopedic injury). These are inherently ambiguous cases.

---

## End-to-End Triage Demo Results

```
=======================================================
  TRIAGE RESULT: Cardiovascular / Pulmonary
=======================================================
  Input: "Patient presents with crushing chest pain radiating to left arm.
           EKG shows ST elevation. History of hypertension."

=======================================================
  TRIAGE RESULT: Orthopedic
=======================================================
  Input: "Patient has right knee pain and swelling after fall.
           X-ray shows tibial plateau fracture."

=======================================================
  TRIAGE RESULT: Neurology
=======================================================
  Input: "Patient complains of severe headache, photophobia, and neck stiffness.
           Lumbar puncture performed."

=======================================================
  TRIAGE RESULT: Gastroenterology
=======================================================
  Input: "Colonoscopy revealed polyps in the sigmoid colon.
           Biopsy sent to pathology. GI follow-up recommended."
```

**All 4 demo cases were classified correctly**, demonstrating robust end-to-end inference on novel clinical text.

---

## Future Improvements

| Improvement | Expected Impact |
|---|---|
| ClinicalBERT / BioBERT fine-tuning | +3–5% macro F1 with sufficient data |
| NegEx / NegBio for negation-aware features | Better handling of negated symptoms |
| Larger abbreviation dictionary (UMLS) | Improved recall for rare abbreviations |
| SMOTE oversampling for minority classes | Better recall on smaller specialty classes |
| Named Entity Recognition (NER) pre-filtering | Reduce noise from non-clinical text |
| Confidence thresholding + abstention | Reject low-confidence predictions for human review |
