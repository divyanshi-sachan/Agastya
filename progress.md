# Agastya — project progress & handoff notes

This file records **completed analysis** and **decisions that affect the next steps** (feature engineering, modeling, evaluation). It aligns with the coursework rubric columns *Dataset Quality & EDA* → *Feature Engineering* / *Model Application*.

---

## Completed: Part 02 — Dataset quality & EDA (`notebooks/Phase 1/Part_02_Dataset_Quality_EDA.ipynb`)

### Scale and structure
- **Master table:** `data/CUAD_v1/master_clauses.csv` — **510** contracts × **83** columns (`Filename` + **41** context/answer pairs).
- **Long format:** **20,910** rows = 510 × 41 (natural unit for per-category analysis).
- **SQuAD JSON:** `CUAD_v1.json` matches the same scale (**510** documents, **20,910** `qas`), useful for QA-style tasks later; **~67.9%** of `qas` are `is_impossible` (no extractable span in that paragraph view).

### Data quality — integrity
- **No duplicate** `Filename`; **no blank** filenames.
- **Full-contract text join:** After normalizing `.pdf` / `.PDF` → `.txt`, **11 / 510** CSV rows still have **no exact** match under `full_contract_txt/`. Causes include **`&`**, **apostrophes**, and **human-readable titles** differing from on-disk stems. The notebook suggests **closest** `.txt` names via string similarity; **manual mapping** may still be needed before any pipeline that merges labels with full raw text.
- **Yes/No vs placeholder:** For binary categories, **No** consistently pairs with clause text **`[]`**. **Two anomalies:** `answer == Yes` but clause is still `[]` — *Insurance* (GarrettMotionInc…) and *Third Party Beneficiary* (HALITRON,INC…). Decide whether to **drop**, **relabel**, or **audit** against PDFs before training.

### Distributions — modeling implications
- **33** categories behave as **Yes/No**; **8** are **free-form** (entities, dates, etc.).
- **Class imbalance (Yes-rate):** across Yes/No categories, positive prevalence ranges roughly **2.5%–73%** (~**29×** between extremes). **Next part:** prefer **macro-F1**, per-class precision/recall, and/or **class weights**; raw accuracy will be misleading.
- **Clause length (only rows with real text, not `[]`):** strongly **right-skewed** (e.g. skew ~5, high excess kurtosis). Typical median ~**40+** words; **99th percentile** in the **hundreds** of words; max **thousands**. **Next part:** TF-IDF `max_features` / `ngram_range`, optional **length caps** or **trimming**; expect a few **outlier spans** to dominate raw counts if untreated.
- **Per-contract density:** mean ~**13** “real” clause spans per document across 41 categories (max ~**32**). **Next part:** use **document-level** train/validation/test splits to avoid **leakage** (same contract in train and test).

### Rubric link (why this matters for evaluators)
- EDA is tied to **actions:** imbalance → metrics/weights; heavy tails → vectorizer settings; filename gaps → join strategy; placeholders → training row definition; multi-row-per-doc → split strategy.

---

## Next part — what to reuse (checklist)

1. **Define the training row set explicitly**  
   - Option A: only rows with **`has_real_clause`** (substantive span) for multi-class clause typing.  
   - Option B: binary tasks per category with consistent **`[]`** = negative for Yes/No fields.  
   - Handle the **2** Yes/`[]` anomalies explicitly.

2. **Splits**  
   - Group by **`Filename`** (or official CUAD split if adopted). Never shuffle rows without grouping by contract.

3. **Features (Phase 1)**  
   - Start from **TF-IDF + n-grams**; justify settings using the **length distribution** above.  
   - Optional: **log-length** or binned length as an extra feature (EDA motivates this).

4. **Metrics**  
   - Report **macro-F1** (and per-class) alongside accuracy; confusion matrix for category confusion.

5. **Full text**  
   - If OCR or full-document context is needed, resolve the **11** filename mismatches first.

---

## Suggested next notebook / milestone

- **Part 03 — Feature engineering & baseline prep:** build the actual matrix `X` and label vector(s) `y` from the long table, apply document-level split, fit **TF-IDF**, train **Naive Bayes + SVM**, report metrics — grounded in the decisions above.

---

## Environment

- Virtual env: **`.venv`** at repo root; dependencies in **`requirements.txt`**.
