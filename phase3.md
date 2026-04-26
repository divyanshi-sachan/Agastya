# Agastya Phase 3 — Cursor Build Instructions
**Hybrid AI Contract Analysis Platform** | Legal-BERT + Bayesian Network + OCR + Streamlit

---

## Project Context

Agastya is a 3-phase research platform for automated legal contract analysis on the [CUAD v1 dataset](https://www.atticusprojectai.org/cuad) (510 contracts, 41 clause categories).

- **Phase 1** (done): Classical ML baselines — LinearSVC, Naive Bayes, TF-IDF → Macro-F1 ~0.55
- **Phase 2** (done): Legal-BERT fine-tuned on CUAD → Macro-F1 0.746
- **Phase 3** (this build): Neuro-symbolic hybrid — OCR + Interface Layer + Bayesian Network + Streamlit UI → Target Macro-F1 > 0.746

---

## Folder Structure to Create

```
Agastya/
├── src/
│   └── phase3/
│       ├── __init__.py
│       ├── ocr/
│       │   ├── __init__.py
│       │   └── extractor.py
│       ├── interface/
│       │   ├── __init__.py
│       │   ├── evidence_encoder.py
│       │   ├── confidence_mapper.py
│       │   └── feature_extractor.py
│       ├── bayesian/
│       │   ├── __init__.py
│       │   ├── network.py
│       │   ├── cpt_definitions.py
│       │   ├── em_trainer.py
│       │   └── inference.py
│       ├── hybrid_pipeline.py
│       ├── ablation.py
│       └── evaluation.py
├── notebooks/Phase_3/
│   ├── 10_bn_structure_and_cpts.ipynb
│   ├── 11_hybrid_pipeline_demo.ipynb
│   ├── 12_ablation_study.ipynb
│   └── 13_interpretability_report.ipynb
├── app/
│   └── streamlit_app.py
├── tests/
│   └── phase3/
│       ├── test_ocr.py
│       ├── test_bn.py
│       └── test_interface.py
├── reports/phase3/
│   ├── ablation_results.csv
│   └── figures/
├── results/phase3/
│   └── bayesian_network.pkl        ← saved after EM training
├── .github/
│   └── workflows/
│       └── test.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## requirements.txt (Phase 3 additions)

Append these to the existing `requirements.txt` from Phases 1 & 2:

```
pgmpy==0.1.25
streamlit==1.32.0
networkx==3.2.1
pdfplumber==0.10.3
pdf2image==1.17.0
easyocr==1.7.1
pillow==10.2.0
```

**System dependency** (add to README and install before running):
```bash
# Ubuntu/Debian
sudo apt install poppler-utils

# macOS
brew install poppler
```

---

## Pipeline Overview (7 Stages)

| Stage | Module | Input | Output | Tech |
|-------|--------|-------|--------|------|
| 1 | OCR | PDF / Image / TXT | Clean contract string | EasyOCR, pdfplumber |
| 2 | Segmentation | Contract string | List of clause strings | Regex + Phase 2 segmenter |
| 3 | Legal-BERT | Clause strings | type, confidence, embedding (768-d), risk_flags | HuggingFace, PyTorch |
| 4 | Interface Layer | BERT outputs | BN evidence dict + virtual evidence | NumPy, scikit-learn |
| 5 | Bayesian Network | Evidence dict | Posterior P(Risk=High/Med/Low) | pgmpy |
| 6 | Smart Report | All above | Risk score, heatmap, trace, explanation | matplotlib, JSON |
| 7 | Web UI | File upload | Interactive dashboard | Streamlit |

---

## MODULE 1 — `src/phase3/ocr/extractor.py`

Tiered extraction strategy: try digital PDF first, fall back to OCR only for scanned/image inputs.

```python
import pdfplumber
from pdf2image import convert_from_bytes
import easyocr
import numpy as np
from PIL import Image
from pathlib import Path
import re

# Initialise once at module level — avoids reloading weights per call
reader = easyocr.Reader(["en"], gpu=False)  # CPU-safe for reproducibility

def extract_text(file_input) -> str:
    """
    Universal contract text extractor.
    Accepts: digital PDF, scanned PDF, PNG, JPG, TXT.
    Returns: clean contract string ready for segmentation.
    """
    suffix = Path(file_input.name).suffix.lower()

    if suffix == ".txt":
        return _clean_text(file_input.read().decode("utf-8"))

    if suffix == ".pdf":
        # Attempt 1: digital text extraction (fast, accurate)
        with pdfplumber.open(file_input) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if len(text.strip()) > 100:
            return _clean_text(text)
        # Attempt 2: OCR fallback for scanned PDFs
        images = convert_from_bytes(file_input.read())
        text = "\n".join(" ".join(reader.readtext(img, detail=0)) for img in images)
        return _clean_text(text)

    if suffix in [".png", ".jpg", ".jpeg"]:
        img = np.array(Image.open(file_input))
        text = " ".join(reader.readtext(img, detail=0))
        return _clean_text(text)

    raise ValueError(f"Unsupported file type: {suffix}")

def _clean_text(text: str) -> str:
    """Fix common OCR artifacts: hyphenation, whitespace, non-ASCII."""
    text = re.sub(r"-\n", "", text)           # merge hyphenated line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)    # collapse excessive newlines
    text = re.sub(r"[^\x00-\x7F]+", " ", text) # remove non-ASCII artifacts
    text = re.sub(r" {2,}", " ", text)        # collapse multiple spaces
    return text.strip()
```

**Tests to write in `tests/phase3/test_ocr.py`:**
- `extract_text()` returns non-empty string for a sample digital PDF
- `extract_text()` returns non-empty string for a sample PNG image
- `_clean_text()` collapses multiple newlines correctly
- Unsupported file type raises `ValueError`

---

## MODULE 2 — Clause Segmentation & Legal-BERT (Reuse from Phase 2)

No new code needed. Phase 3 only consumes Phase 2 outputs.

**Required output format per clause** (extend Phase 2 BERT inference script if these fields are missing):

```python
{
    "clause_text":     str,          # original clause string
    "clause_type":     str,          # e.g. "Payment", "Termination"
    "confidence":      float,        # 0.0–1.0 softmax max probability
    "embedding":       np.ndarray,   # shape (768,) — CLS token vector
    "risk_indicators": list[str],    # e.g. ["vague_timing", "no_penalty"]
    "logits":          np.ndarray,   # shape (41,) — raw class logits
}
```

**5 clause types that flow into the Bayesian Network:**

| BN Node | CUAD Clause Label |
|---------|-------------------|
| Has_Payment_Clause | Revenue / Royalty / Payment |
| Has_Termination_Clause | Termination for Convenience |
| Has_Liability_Clause | Limitation of Liability |
| Has_Confidentiality_Clause | Non-Compete / NDA |
| Has_Dispute_Resolution_Clause | Governing Law / Dispute |

---

## MODULE 3 — Interface Layer

### `src/phase3/interface/evidence_encoder.py`

Converts BERT clause predictions into hard Boolean evidence for BN nodes. Threshold = 0.5.

```python
CLAUSE_MAP = {
    "Payment":            "Has_Payment_Clause",
    "Termination":        "Has_Termination_Clause",
    "Liability":          "Has_Liability_Clause",
    "Confidentiality":    "Has_Confidentiality_Clause",
    "Dispute Resolution": "Has_Dispute_Resolution_Clause",
}

def encode_evidence(bert_outputs: list[dict]) -> dict:
    evidence = {node: "Absent" for node in CLAUSE_MAP.values()}
    for clause in bert_outputs:
        node = CLAUSE_MAP.get(clause["clause_type"])
        if node and clause["confidence"] > 0.5:
            evidence[node] = "Present"
    return evidence
```

### `src/phase3/interface/confidence_mapper.py`

Converts BERT confidence scores into soft (virtual) evidence vectors. This is what makes the hybrid **genuinely integrated** rather than merely sequential.

```python
from .evidence_encoder import CLAUSE_MAP

def map_confidence_to_virtual_evidence(bert_outputs: list[dict]) -> dict:
    """
    Returns {node_name: [P(Absent), P(Present)]} for each BN presence node.
    High BERT confidence → strong evidence for presence.
    Low BERT confidence → uncertainty preserved in BN inference.
    """
    virtual_evidence = {}
    for clause in bert_outputs:
        node = CLAUSE_MAP.get(clause["clause_type"])
        if node:
            conf = clause["confidence"]
            virtual_evidence[node] = [1 - conf, conf]
    return virtual_evidence
```

### `src/phase3/interface/feature_extractor.py`

Uses cosine similarity between clause embeddings to detect cross-clause conflicts.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_cross_clause_features(embeddings: dict) -> float:
    """
    Computes semantic conflict signal between Payment and Termination clauses.
    Low cosine similarity → potential contradiction → higher conflict probability.
    Returns: conflict_signal float in [0, 1]
    """
    if "Payment" in embeddings and "Termination" in embeddings:
        sim = cosine_similarity(
            embeddings["Payment"].reshape(1, -1),
            embeddings["Termination"].reshape(1, -1)
        )[0][0]
        return float(1 - sim)  # low similarity = high conflict signal
    return 0.5  # default: maximum uncertainty
```

**Tests to write in `tests/phase3/test_interface.py`:**
- `encode_evidence()` returns 5-key dict with `Present`/`Absent` values
- `map_confidence_to_virtual_evidence()` returns vectors summing to 1.0
- `extract_cross_clause_features()` returns float in [0, 1]

---

## MODULE 4 — Bayesian Network

### Network Structure: 8 Nodes, 4 Layers

```
LAYER 1 — Clause Presence (5 nodes, binary: Present / Absent)
  P1: Has_Payment_Clause
  P2: Has_Termination_Clause
  P3: Has_Liability_Clause
  P4: Has_Confidentiality_Clause
  P5: Has_Dispute_Resolution_Clause

LAYER 2 — Risk Factors (2 nodes, binary: Risky / Not_Risky)
  R1: Payment_Or_Termination_Risky       [parents: P1, P2]
  R2: Liability_Or_Confidentiality_Risky [parents: P3, P4]

LAYER 3 — Conflict Detection (1 node, binary: Conflict / No_Conflict)
  D1: Cross_Clause_Conflict              [parents: R1, R2, P5]

LAYER 4 — Final Risk (1 node, 3-state: Low / Medium / High)
  F1: Contract_Risk_Level                [parent: D1]
```

### `src/phase3/bayesian/network.py`

```python
from pgmpy.models import BayesianNetwork

EDGES = [
    ("Has_Payment_Clause",            "Payment_Or_Termination_Risky"),
    ("Has_Termination_Clause",        "Payment_Or_Termination_Risky"),
    ("Has_Liability_Clause",          "Liability_Or_Confidentiality_Risky"),
    ("Has_Confidentiality_Clause",    "Liability_Or_Confidentiality_Risky"),
    ("Payment_Or_Termination_Risky",  "Cross_Clause_Conflict"),
    ("Liability_Or_Confidentiality_Risky", "Cross_Clause_Conflict"),
    ("Has_Dispute_Resolution_Clause", "Cross_Clause_Conflict"),
    ("Cross_Clause_Conflict",         "Contract_Risk_Level"),
]

def build_network() -> BayesianNetwork:
    model = BayesianNetwork(EDGES)
    return model
```

### `src/phase3/bayesian/cpt_definitions.py`

Seed CPTs from legal domain reasoning. Document justifications in Notebook 10.

```python
from pgmpy.factors.discrete import TabularCPD

def get_seed_cpts(model):
    """
    Returns list of TabularCPD objects with hand-crafted seed values.
    Legal justifications must be documented in Notebook 10.
    """
    # Presence nodes — uniform priors (data will refine via EM)
    cpt_payment    = TabularCPD("Has_Payment_Clause", 2,
                                [[0.6], [0.4]],
                                state_names={"Has_Payment_Clause": ["Present", "Absent"]})

    cpt_termination = TabularCPD("Has_Termination_Clause", 2,
                                 [[0.5], [0.5]],
                                 state_names={"Has_Termination_Clause": ["Present", "Absent"]})

    cpt_liability   = TabularCPD("Has_Liability_Clause", 2,
                                 [[0.55], [0.45]],
                                 state_names={"Has_Liability_Clause": ["Present", "Absent"]})

    cpt_confidentiality = TabularCPD("Has_Confidentiality_Clause", 2,
                                     [[0.45], [0.55]],
                                     state_names={"Has_Confidentiality_Clause": ["Present", "Absent"]})

    cpt_dispute     = TabularCPD("Has_Dispute_Resolution_Clause", 2,
                                 [[0.5], [0.5]],
                                 state_names={"Has_Dispute_Resolution_Clause": ["Present", "Absent"]})

    # R1: P(Risky | Payment, Termination)
    # Both present → 0.75 risk (vague payment + easy termination = severe)
    cpt_r1 = TabularCPD(
        "Payment_Or_Termination_Risky", 2,
        [[0.75, 0.60, 0.55, 0.15],   # Risky
         [0.25, 0.40, 0.45, 0.85]],  # Not_Risky
        evidence=["Has_Payment_Clause", "Has_Termination_Clause"],
        evidence_card=[2, 2],
        state_names={
            "Payment_Or_Termination_Risky": ["Risky", "Not_Risky"],
            "Has_Payment_Clause": ["Present", "Absent"],
            "Has_Termination_Clause": ["Present", "Absent"],
        }
    )

    # R2: P(Risky | Liability, Confidentiality)
    cpt_r2 = TabularCPD(
        "Liability_Or_Confidentiality_Risky", 2,
        [[0.80, 0.65, 0.60, 0.20],
         [0.20, 0.35, 0.40, 0.80]],
        evidence=["Has_Liability_Clause", "Has_Confidentiality_Clause"],
        evidence_card=[2, 2],
        state_names={
            "Liability_Or_Confidentiality_Risky": ["Risky", "Not_Risky"],
            "Has_Liability_Clause": ["Present", "Absent"],
            "Has_Confidentiality_Clause": ["Present", "Absent"],
        }
    )

    # D1: P(Conflict | R1, R2, P5)
    cpt_d1 = TabularCPD(
        "Cross_Clause_Conflict", 2,
        [[0.80, 0.92, 0.50, 0.68, 0.45, 0.62, 0.15, 0.30],   # Conflict
         [0.20, 0.08, 0.50, 0.32, 0.55, 0.38, 0.85, 0.70]],  # No_Conflict
        evidence=["Payment_Or_Termination_Risky",
                  "Liability_Or_Confidentiality_Risky",
                  "Has_Dispute_Resolution_Clause"],
        evidence_card=[2, 2, 2],
        state_names={
            "Cross_Clause_Conflict": ["Conflict", "No_Conflict"],
            "Payment_Or_Termination_Risky": ["Risky", "Not_Risky"],
            "Liability_Or_Confidentiality_Risky": ["Risky", "Not_Risky"],
            "Has_Dispute_Resolution_Clause": ["Present", "Absent"],
        }
    )

    # F1: P(Risk_Level | Conflict)
    cpt_f1 = TabularCPD(
        "Contract_Risk_Level", 3,
        [[0.05, 0.55],   # Low
         [0.20, 0.35],   # Medium
         [0.75, 0.10]],  # High
        evidence=["Cross_Clause_Conflict"],
        evidence_card=[2],
        state_names={
            "Contract_Risk_Level": ["Low", "Medium", "High"],
            "Cross_Clause_Conflict": ["Conflict", "No_Conflict"],
        }
    )

    return [cpt_payment, cpt_termination, cpt_liability,
            cpt_confidentiality, cpt_dispute, cpt_r1, cpt_r2, cpt_d1, cpt_f1]
```

### `src/phase3/bayesian/em_trainer.py`

```python
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ExpectationMaximization
import pandas as pd
import pickle

def build_network(edges: list[tuple]) -> BayesianNetwork:
    model = BayesianNetwork(edges)
    return model

def train_with_em(model: BayesianNetwork, train_df: pd.DataFrame,
                  n_iter: int = 100) -> BayesianNetwork:
    """
    train_df columns must match node names.
    Values must be state strings: "Present"/"Absent", "Risky"/"Not_Risky" etc.
    """
    model.fit(
        data=train_df,
        estimator=ExpectationMaximization,
        n_jobs=-1,
        n_iter=n_iter
    )
    return model

def save_model(model, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path: str) -> BayesianNetwork:
    with open(path, "rb") as f:
        return pickle.load(f)

def prepare_bn_training_data(cuad_train_df: pd.DataFrame,
                              bert_risk_labels: pd.DataFrame,
                              conflict_labels: pd.Series,
                              risk_labels: pd.Series) -> pd.DataFrame:
    """
    Converts CUAD train.csv + BERT outputs into BN training DataFrame.
    Each row = one contract; columns = BN node names with correct state strings.
    """
    bn_df = pd.DataFrame()
    bn_df["Has_Payment_Clause"] = cuad_train_df["revenue_royalty_payment"] \
        .apply(lambda x: "Present" if x == 1 else "Absent")
    bn_df["Has_Termination_Clause"] = cuad_train_df["termination_for_convenience"] \
        .apply(lambda x: "Present" if x == 1 else "Absent")
    bn_df["Has_Liability_Clause"] = cuad_train_df["limitation_of_liability"] \
        .apply(lambda x: "Present" if x == 1 else "Absent")
    bn_df["Has_Confidentiality_Clause"] = cuad_train_df["non_compete"] \
        .apply(lambda x: "Present" if x == 1 else "Absent")
    bn_df["Has_Dispute_Resolution_Clause"] = cuad_train_df["governing_law"] \
        .apply(lambda x: "Present" if x == 1 else "Absent")
    bn_df["Payment_Or_Termination_Risky"] = bert_risk_labels["payment_term_risky"] \
        .apply(lambda x: "Risky" if x else "Not_Risky")
    bn_df["Liability_Or_Confidentiality_Risky"] = bert_risk_labels["liability_conf_risky"] \
        .apply(lambda x: "Risky" if x else "Not_Risky")
    bn_df["Cross_Clause_Conflict"] = conflict_labels \
        .apply(lambda x: "Conflict" if x else "No_Conflict")
    bn_df["Contract_Risk_Level"] = risk_labels  # "Low", "Medium", "High"
    return bn_df
```

### `src/phase3/bayesian/inference.py`

```python
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianNetwork

def run_inference(model: BayesianNetwork, evidence: dict,
                  query_var: str = "Contract_Risk_Level") -> dict:
    """
    Run belief propagation given hard evidence.
    Returns posterior distribution over query_var states.
    """
    bp = BeliefPropagation(model)
    result = bp.query(variables=[query_var], evidence=evidence)
    return {
        "distribution": result[query_var],
        "risk_level": result[query_var].state_names[query_var][result[query_var].values.argmax()],
        "probabilities": {
            state: float(result[query_var].get_value(**{query_var: state}))
            for state in ["Low", "Medium", "High"]
        }
    }
```

**Tests to write in `tests/phase3/test_bn.py`:**
- `model.check_model()` returns `True`
- `run_inference()` returns dict with `Low`/`Medium`/`High` keys
- All probabilities sum to 1.0
- CPTs differ from seed values after EM (print before/after in test output)

---

## MODULE 5 — `src/phase3/hybrid_pipeline.py`

The single entry point wiring all modules together.

```python
from src.phase3.ocr.extractor import extract_text
from src.phase3.interface.evidence_encoder import encode_evidence
from src.phase3.interface.confidence_mapper import map_confidence_to_virtual_evidence
from src.phase3.interface.feature_extractor import extract_cross_clause_features
from src.phase3.bayesian.inference import run_inference
from src.phase3.bayesian.em_trainer import load_model
from pgmpy.inference import BeliefPropagation

# Import Phase 2 components
# from src.phase2.segmenter import segment_contract
# from src.phase2.bert_encoder import load_legal_bert

class AgastyaHybridPipeline:
    def __init__(self, bert_model_path: str, bn_model_path: str):
        self.bert      = load_legal_bert(bert_model_path)   # Phase 2
        self.bn        = load_model(bn_model_path)
        self.bp_engine = BeliefPropagation(self.bn)

    def predict(self, contract_text: str) -> dict:
        # Stage 1: Segment
        clauses = segment_contract(contract_text)            # Phase 2

        # Stage 2: BERT Encode
        bert_outputs = [self.bert.predict(c) for c in clauses]

        # Stage 3: Interface Layer
        hard_evidence    = encode_evidence(bert_outputs)
        virtual_evidence = map_confidence_to_virtual_evidence(bert_outputs)
        embeddings       = {o["clause_type"]: o["embedding"] for o in bert_outputs}
        conflict_signal  = extract_cross_clause_features(embeddings)

        # Stage 4: BN Inference
        bn_result = run_inference(self.bn, hard_evidence)

        # Stage 5: Assemble Smart Report
        return {
            "risk_level":         bn_result["risk_level"],
            "risk_probabilities": bn_result["probabilities"],
            "clause_evidence":    hard_evidence,
            "conflict_signal":    conflict_signal,
            "bert_details":       bert_outputs,
            "bn_trace":           str(bn_result["distribution"]),
            "virtual_evidence":   virtual_evidence,
        }
```

---

## MODULE 6 — `src/phase3/ablation.py`

```python
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
from src.phase3.hybrid_pipeline import AgastyaHybridPipeline

class AblationRunner:
    def __init__(self, test_df, bert_path, bn_path, svm_path):
        self.test_df = test_df
        self.pipelines = {
            "ML_Only": load_svm_pipeline(svm_path),        # Phase 1 SVM
            "DL_Only": load_bert_pipeline(bert_path),       # Phase 2 BERT
            "Hybrid":  AgastyaHybridPipeline(bert_path, bn_path),
        }

    def run(self) -> pd.DataFrame:
        results = {}
        for name, pipeline in self.pipelines.items():
            preds  = [pipeline.predict(text) for text in self.test_df["text"]]
            labels = self.test_df["label"].tolist()
            results[name] = {
                "Macro-F1":  f1_score(labels, preds, average="macro"),
                "Accuracy":  accuracy_score(labels, preds),
                "Precision": precision_score(labels, preds, average="macro"),
                "Recall":    recall_score(labels, preds, average="macro"),
            }
        return pd.DataFrame(results).T

    def per_class_analysis(self) -> pd.DataFrame:
        """Per-class F1 for each config — feeds the diagnostic paragraph."""
        class_results = {}
        for name, pipeline in self.pipelines.items():
            preds  = [pipeline.predict(t) for t in self.test_df["text"]]
            labels = self.test_df["label"].tolist()
            f1s    = f1_score(labels, preds, average=None)
            class_results[name] = f1s
        return pd.DataFrame(class_results, index=pipeline.class_names)
```

**Target ablation table** (fill with real numbers after running):

| Configuration | Macro-F1 | Accuracy | Precision | Recall |
|---------------|----------|----------|-----------|--------|
| ML Only (LinearSVC) | ~0.55 | ~0.72 | — | — |
| DL Only (Legal-BERT) | 0.746 | 0.824 | — | — |
| Hybrid (Agastya) | **>0.746** | **>0.824** | — | — |

---

## MODULE 7 — `app/streamlit_app.py`

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.phase3.ocr.extractor import extract_text
from src.phase3.hybrid_pipeline import AgastyaHybridPipeline

st.set_page_config(page_title="Agastya", page_icon="⚖", layout="wide")
st.title("Agastya — Contract Risk Analyzer")
st.caption("Hybrid AI Platform: Legal-BERT + Bayesian Network")

@st.cache_resource
def load_pipeline():
    return AgastyaHybridPipeline(
        bert_model_path="phase2_results/legal_bert_phase2.pt",
        bn_model_path="results/phase3/bayesian_network.pkl"
    )

uploaded = st.file_uploader("Upload contract", type=["pdf", "txt", "png", "jpg"])

if uploaded:
    with st.spinner("Running OCR and analysis..."):
        text     = extract_text(uploaded)
        pipeline = load_pipeline()
        result   = pipeline.predict(text)

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Level", result["risk_level"])
    col2.metric("High Risk Prob", f'{result["risk_probabilities"]["High"]:.1%}')
    col3.metric("Conflict Signal", f'{result["conflict_signal"]:.2f}')

    st.subheader("Risk Distribution")
    probs = result["risk_probabilities"]
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.barh(list(probs.keys()), list(probs.values()), color=["green", "orange", "red"])
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    st.subheader("Clause Evidence (Interface Layer Output)")
    st.dataframe(pd.DataFrame([result["clause_evidence"]]))

    st.subheader("Bayesian Network Inference Trace")
    st.json(result["bn_trace"])

    st.subheader("Per-Clause BERT Analysis")
    bert_df = pd.DataFrame(result["bert_details"])
    bert_df = bert_df[["clause_type", "confidence", "risk_indicators"]]
    st.dataframe(bert_df)
```

**Run with:**
```bash
streamlit run app/streamlit_app.py
```

---

## CI/CD — `.github/workflows/test.yml`

```yaml
name: Agastya Phase 3 CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: sudo apt install -y poppler-utils
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/phase3/ -v
      - run: python -m pytest tests/phase3/test_ocr.py -v
      - run: python -m pytest tests/phase3/test_bn.py -v
      - run: python -m pytest tests/phase3/test_interface.py -v
```

---

## Notebook Content Plan

| Notebook | Title | Must Contain |
|----------|-------|--------------|
| `10_bn_structure_and_cpts.ipynb` | BN Structure & CPTs | DAG definition, seed CPTs with legal justifications, EM before/after CPT comparison, network visualization (networkx) |
| `11_hybrid_pipeline_demo.ipynb` | Hybrid Pipeline Demo | End-to-end trace on 5 real contracts, inference trace with real numbers matching the Step table below, risk heatmap |
| `12_ablation_study.ipynb` | Ablation Study | Full ablation table (ML/DL/Hybrid), per-class F1 heatmap, diagnostic paragraph |
| `13_interpretability_report.ipynb` | Interpretability & Report | BN posterior distributions, attention weights, Phase 1→2→3 progression summary table |

**⚠ Important:** Write notebooks AFTER the code works. Every result must come from actually-executing code.

### Concrete Inference Trace for Notebook 11 (reproduce with real numbers)

| Step | Component | Operation | Expected Output |
|------|-----------|-----------|-----------------|
| 1 | Legal-BERT | Encode "Payment shall be made within 30 days of invoice." | type: Payment, conf: ~0.95, risk_flags: [vague_timing] |
| 2 | Evidence Encoder | conf > 0.5 → Present | hard_evidence["Has_Payment_Clause"] = "Present" |
| 3 | Confidence Mapper | 0.95 → [0.05, 0.95] | virtual_evidence["Has_Payment_Clause"] = [0.05, 0.95] |
| 4 | BN Prior | P(R1=Risky) before evidence | Print prior |
| 5 | BN Update | P(R1=Risky \| Payment=Present) | Should be ~0.75 per seed CPT |
| 6 | BN Propagation | P(D1=Conflict) after update | Should increase |
| 7 | BN Final | P(F1=High \| all evidence) | Should be > 0.70 |
| 8 | Smart Report | Risk: HIGH (XX% confidence) | JSON output to UI |

---

## Determinism Checklist

- `random_state=42` everywhere: SVM, train/test splits, EM initialisation
- All package versions pinned in `requirements.txt`
- Save trained BN to `results/phase3/bayesian_network.pkl` after EM
- `gpu=False` in EasyOCR — GPU paths are not reproducible across machines
- Add poppler install instruction to `README.md`

### `.gitignore` additions

```
.EasyOCR/
results/phase3/models/*.pkl
__pycache__/
*.pyc
```

---

## Build Order (do not skip steps)

| Step | Task | Pass Criterion |
|------|------|----------------|
| 1 | Install dependencies | `import pgmpy; import easyocr` succeeds |
| 2 | Build OCR extractor | `extract_text()` returns non-empty string for 1 PDF + 1 image |
| 3 | Verify OCR → segmenter | Segmenter receives clean string, returns list of clauses |
| 4 | Define CPTs + BN structure | `model.check_model()` returns `True` |
| 5 | Run EM on training data | CPTs change from seed values (print before/after) |
| 6 | Test BN inference with dummy evidence | Returns dict with Low/Med/High probabilities |
| 7 | Build Evidence Encoder | Returns 5-key dict with Present/Absent values |
| 8 | Build Confidence Mapper | Returns virtual evidence vectors summing to 1.0 |
| 9 | Build Feature Extractor | Returns float in [0, 1] |
| 10 | Wire Hybrid Pipeline | `pipeline.predict()` returns full result dict on 3 contracts |
| 11 | Run Ablation Study | Table with 3 rows, all metrics populated |
| 12 | Build Streamlit App | UI loads, file upload works |
| 13 | Write Notebooks 10-13 | All cells run top-to-bottom without errors |
| 14 | Add CI/CD | GitHub Actions green on push |

---

## Rubric Targets

| Criterion | Target | What Earns It |
|-----------|--------|---------------|
| Hybrid Innovation | 5/5 — Synergistic | Legal-BERT embeddings feed BN via Interface Layer; confidence scores become virtual evidence; neuro-symbolic |
| Ablation Studies | 5/5 — Diagnostic | Real table: ML-Only vs DL-Only vs Hybrid; per-class F1 analysis; diagnostic paragraph with specific delta values |
| Architecture Diagram | 5/5 — Publication-Ready | Tensor shapes (N,768), (N,41); colour zones for DL/Interface/BN; fusion mechanism labelled; 300 DPI export |
| Reproducibility | 4/5 — Documented | Clean repo, pinned requirements.txt, README with poppler install, commented code |
| Extra Mile | 4/5 — Strong | Streamlit UI + GitHub Actions CI/CD, both functional |

### Architecture Diagram Requirements (draw.io or matplotlib)
- Purple zone: DL (Legal-BERT)
- Yellow zone: Interface Layer
- Orange zone: Bayesian Network
- Green zone: Output / Smart Report
- Every arrow labelled with data shape: `(N, 768)`, `(N, 41)`, `(5,)`, `scalar`, `dict`
- Fusion mechanism labels: "Evidence Encoding", "Virtual Evidence", "Belief Propagation"
- Export as PNG ≥ 300 DPI or SVG

---

## Final Submission Checklist

- [ ] `model.check_model()` returns `True` for BN structure
- [ ] EM has run and CPTs differ from seed values (print comparison in Notebook 10)
- [ ] `hard_evidence` dict is non-empty for test contracts (print in Notebook 11)
- [ ] Ablation table contains real numbers for all 3 configs
- [ ] Diagnostic paragraph written with per-class delta values
- [ ] Architecture diagram shows tensor shapes and fusion mechanism
- [ ] `streamlit run app/streamlit_app.py` launches and accepts file upload
- [ ] GitHub Actions passes on latest push
- [ ] README includes poppler install instruction
- [ ] All Phase 3 notebooks run top-to-bottom without errors
- [ ] `requirements.txt` has all Phase 3 deps with pinned versions
- [ ] No hardcoded absolute paths anywhere in Phase 3 code

---

*Agastya Phase 3 PRD · Divyanshi Sachan (230069) · Subham Mahapatra (230037)*  
*Built on CUAD v1 · Legal-BERT · pgmpy · EasyOCR · Streamlit*