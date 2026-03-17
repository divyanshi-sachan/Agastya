## 🏛 Agastya: A Hybrid AI Platform for Automated Contract Analysis

### 📌 Overview
Agastya is a hybrid AI system that bridges the gap between complex legal contracts and non-expert users. It transforms static, often non-digital contracts into structured, interpretable insights using a combination of:

- **Deep Learning (DL)** for semantic understanding of legal language
- **Machine Learning (ML)** for probabilistic reasoning and risk assessment

At a high level, Agastya performs end-to-end contract analysis, including clause extraction, classification, risk detection, and logical consistency checking.

### 🚨 Problem Statement
Legal contracts are often:

- **Difficult to understand** due to dense, domain-specific legal language  
- **Non-digital or low quality**, e.g. scanned PDFs or handwritten documents  
- **Risk-prone**, with missing clauses, vague terms, or unfair conditions  

In regions like India, a large fraction of contracts are still non-digital. This creates high information asymmetry between parties and exposes individuals/SMEs to legal and financial risk.

Agastya aims to reduce this asymmetry by making contract structure, risks, and inconsistencies transparent and machine-interpretable.

### 💡 Proposed Solution
Agastya provides an intelligent pipeline that:

1. **Digitizes contracts** using OCR (from PDFs/images)
2. **Extracts and segments clauses** into meaningful units
3. **Classifies clauses** using transformer-based models (e.g. Legal-BERT)
4. **Detects risks and inconsistencies** (missing / vague / unfair clauses)
5. **Performs probabilistic reasoning** over clause-level evidence
6. **Generates an interpretable Smart Report** for end users

The system is designed to be modular so that components (OCR, clause segmentation, models) can be swapped or improved independently.

### ⚙️ 12-Step System Workflow
The full end-to-end workflow is:

1. **Upload Contract** (PDF/Image)  
2. **Document Preprocessing** (denoising, deskewing, basic cleanup)  
3. **OCR** (text extraction from scanned documents)  
4. **Clause Segmentation** (split into clause-level units)  
5. **Clause Classification** (Transformer-based, e.g. Legal-BERT)  
6. **Risk Detection** (missing / vague / unfair clauses)  
7. **Risk Scoring** (Low / Medium / High)  
8. **Logical Consistency Checking** across clauses  
9. **Clause Relationship Mapping** (dependencies and cross-references)  
10. **Risk Phrase Highlighting** in text  
11. **Smart Report Generation** (dashboard + narrative summary)  
12. **Version Comparison** (advanced feature for contract revisions)  

> **Note**: Not all steps may be fully implemented in code yet; this document describes the *target* system design for Agastya.

### 🧠 System Architecture

High-level pipeline:

`Input (PDF/Image) → OCR → Clause Segmentation → DL Model (Legal-BERT) → ML Reasoning (Bayesian Network) → Output (Smart Report)`

- **Perception Stage** → OCR + segmentation  
- **Deep Learning Stage** → Legal-BERT-based clause classification & embedding extraction  
- **Reasoning Stage** → Bayesian Network for probabilistic risk reasoning and consistency checks  
- **Output Stage** → Risk dashboard, Smart Report, and visualizations  

The architecture is explicitly **hybrid**: deep learning provides rich semantic representations, while probabilistic ML models capture dependencies and legal logic.

### 🤖 AI Components

#### 🔹 Deep Learning (DL)
- **Model**: Legal-BERT (Transformer-based model trained for legal text)  
- **Tasks**:
  - Clause classification into predefined legal categories  
  - Semantic feature extraction (embeddings)  
- **Outputs**:
  - Clause type / category  
  - Semantic embeddings (e.g. 768-dimensional vectors)  
  - Confidence scores  
  - Clause-level risk indicators (e.g. vague_timing, missing_party, unfair_term)  

#### 🔹 Machine Learning (ML)

- **Baseline Model (for ablation/comparison)**  
  - TF-IDF + SVM  
  - Used as a simpler baseline for clause classification and risk signals  

- **Probabilistic Model (Core Reasoning Layer)**  
  - **Bayesian Network** over clause-level features and risks  
  - **Functions**:
    - Risk reasoning and aggregation  
    - Dependency modeling between clauses (e.g. payment ↔ termination)  
    - Logical consistency checking (detect contradictions or missing dependencies)  

### 🔗 Hybrid DL–ML Integration
Agastya’s core innovation is the interaction between the DL and ML layers.

**Flow:**

1. DL processes each clause and outputs:
   - Clause type (e.g. Payment, Termination, Confidentiality)  
   - Embedding vector  
   - Local risk indicators  
2. An **interface layer** converts these outputs into structured evidence variables.  
3. The **Bayesian Network** consumes these variables for global reasoning and risk scoring.  
4. The final **risk score** and explanations are surfaced to the user.

**Example**

- **Clause** → "Payment within 30 days"  
- **DL Output**:  
  - Type: `Payment`  
  - Risk: `vague_timing`  
- **ML Reasoning** (Bayesian Network):  
  - Payment clause is vague **and** termination clause is highly protective for the other party  
  - → Overall contract-level **HIGH RISK (e.g. 78%)** for payment-timing related disputes  

### 📊 Dataset

#### Primary Dataset
- **CUAD (Contract Understanding Atticus Dataset)**  
  - 13,000+ labeled clauses  
  - 41 legal categories  

#### Custom Dataset
- Manually collected contracts (e.g. Indian context)  
- Used for:
  - Domain adaptation  
  - Fine-tuning on local contract styles and clause templates  

### 🧪 Evaluation Strategy

We compare three models:

| Model   | Description                      |
|--------|----------------------------------|
| Model A | TF-IDF + SVM (Baseline)         |
| Model B | Transformer (Legal-BERT)        |
| Model C | Hybrid (Agastya: DL + Bayesian) |

**Metrics**

- F1-score  
- Precision / Recall  
- Accuracy  
- (Optionally) Calibration and interpretability metrics  

### 📈 Expected Contributions

- **Improved clause classification** using transformer-based models  
- **Better risk detection** via probabilistic reasoning over clause combinations  
- **Logical consistency checking** across related clauses  
- **Interpretable AI outputs** instead of pure black-box predictions  

### 📊 Output Features

The system is expected to produce:

- **Digitized contract** (OCR output)  
- **Clause categorization** (per-clause labels)  
- **Risk detection** (flags for missing/vague/unfair clauses)  
- **Clause dependency insights** (which clauses influence each other)  
- **Risk heatmap visualization** (clause-level risk scores)  
- **Smart Report dashboard** (human-readable contract summary and risks)  

### 📁 Project Structure

Target high-level layout:

- **`data/`**: Datasets (CUAD, custom contracts, preprocessed text, splits)  
- **`src/`**: Source code for the processing pipeline  
  - `ocr/` – document preprocessing and OCR wrappers  
  - `segmentation/` – clause segmentation and cleaning  
  - `models/` – DL models (Legal-BERT, TF-IDF+SVM)  
  - `reasoning/` – Bayesian Network and risk reasoning logic  
  - `reporting/` – Smart Report and visualization utilities  
- **`models/`**: Saved model weights and checkpoints  
- **`notebooks/`**: Exploration, EDA, and experiment notebooks  
- **`requirements.txt`**: Python dependencies  
- **`README.md`**: Project documentation (this file)  


### 🛠 Getting Started

> This section assumes a Python environment (e.g. 3.9+). Update commands as needed once the codebase is fully implemented.

1. **Clone the repository**

```bash
git clone <your-repo-url>.git
cd agastya
```

2. **Create and activate a virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run experiments / pipeline**

- (To be updated) Example:

```bash
python -m src.pipeline.run_contract_analysis --input path/to/contract.pdf --output reports/contract_001.json
```

Fill this section in as the CLI or notebooks for the project solidify.

### 🚀 Future Work

- Advanced risk prediction models (e.g. graph neural networks over clause graphs)  
- Richer **explainable AI** (highlight not only risky phrases, but also reasoning paths in the Bayesian Network)  
- Contract comparison system (e.g. highlight differences between two versions or against a standard template)  
- Web-based deployment and interactive dashboard for non-technical users  

### 👥 Team

- **Team Agastya**  
  - Divyanshi Sachan  
  - Subham Mahapatra  

### 🧠 Key Idea
Agastya combines the strengths of **Deep Learning** (for understanding legal language) and **Machine Learning / Probabilistic Reasoning** (for modeling dependencies and uncertainty) to build an **interpretable and intelligent legal AI system** for automated contract analysis.

