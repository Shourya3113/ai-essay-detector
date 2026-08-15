# AI Admissions Essay Detector

> **First-Principles Statistical & Linguistic Evidence Engine for College Admissions**  
> *Built for the 2026 i12 HR Drive Hackathon (Project 2)*

[![Next.js 15](https://img.shields.io/badge/Next.js-15.x-black?logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-GPT--2-yellow?logo=huggingface)](https://huggingface.co/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Table of Contents

- [1. Executive Overview & Core Philosophy](#1-executive-overview--core-philosophy)
- [2. End-to-End System Architecture](#2-end-to-end-system-architecture)
- [3. The 9-Signal Linguistic & Statistical Pipeline](#3-the-9-signal-linguistic--statistical-pipeline)
- [4. Quick Start & Local Setup](#4-quick-start--local-setup)
  - [4.1 Backend Setup (Python FastAPI)](#41-backend-setup-python-fastapi)
  - [4.2 Frontend Setup (Next.js 15)](#42-frontend-setup-nextjs-15)
- [5. Machine Learning Pipeline Execution](#5-machine-learning-pipeline-execution)
- [6. Benchmark Evaluation & Key Findings](#6-benchmark-evaluation--key-findings)
- [7. Comprehensive Documentation Directory](#7-comprehensive-documentation-directory)

---

## 1. Executive Overview & Core Philosophy

University admissions offices evaluate hundreds of thousands of personal statements annually. With the proliferation of frontier Large Language Models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro), admissions committees face a dual crisis:
1. **Ghostwritten Applications**: Unchecked synthetic generation that dilutes authentic student identity.
2. **Catastrophic False Accusations**: Commercial black-box detectors (which query an LLM to render a verdict) falsely flag up to **60% of non-native English (ESL) applicants** due to formulaic transitions taught in TOEFL/IELTS test prep.

### The First-Principles Difference

The **AI Admissions Essay Detector** rejects the LLM-as-a-Judge antipattern. Instead of delegating verdicts to another black-box chat model, our architecture extracts **9 mathematically rigorous, orthogonal linguistic signals** across information theory, syntax, and stylometry. 

Generative models (GPT-2 124M) are deployed strictly as **uncalibrated physical sensors (probability thermometers)** to compute raw token log-likelihoods. Feature vectors are standardized and classified using **L2 Logistic Regression with Platt Scaling**, yielding transparent, interpretable, and legally defensible confidence ranges.

---

## 2. End-to-End System Architecture

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │                Raw Student Essay Input                 │
                                  └──────────────────────────┬─────────────────────────────┘
                                                             │
                    ┌────────────────────────────────────────┼────────────────────────────────────────┐
                    │                                        │                                        │
     ┌──────────────┴─────────────┐          ┌───────────────┴──────────────┐         ┌───────────────┴──────────────┐
     │ 1. Information Theory      │          │ 2. Syntactic & Structural    │         │ 3. Stylometric & Lexical     │
     ├────────────────────────────┤          ├──────────────────────────────┤         ├──────────────────────────────┤
     │ • Token Perplexity (PPL)   │          │ • Sentence Burstiness (CV)   │         │ • MATTR & Hapax Legomena     │
     │ • Cross-Entropy Loss (H)   │          │ • POS Bigram Entropy         │         │ • Function Word Entropy      │
     │ • Readability Variance     │          │ • Discourse Marker Density   │         │ • Punctuation Uniformity     │
     │                            │          │                              │         │ • AI Fingerprint N-Grams     │
     └──────────────┬─────────────┘          └───────────────┬──────────────┘         └───────────────┬──────────────┘
                    │                                        │                                        │
                    └────────────────────────────────────────┼────────────────────────────────────────┘
                                                             │
                                              ┌──────────────▼──────────────┐
                                              │  9-D Feature Vector x ∈ ℝ⁹  │
                                              └──────────────┬──────────────┘
                                                             │
                                              ┌──────────────▼──────────────┐
                                              │    StandardScaler (z-score) │
                                              └──────────────┬──────────────┘
                                                             │
                                              ┌──────────────▼──────────────┐
                                              │  L2 Logistic Regression     │
                                              │    + Platt Calibrator       │
                                              └──────────────┬──────────────┘
                                                             │
                                      ┌──────────────────────┴──────────────────────┐
                                      │                                             │
                        ┌─────────────▼─────────────┐                 ┌─────────────▼─────────────┐
                        │   Overall Probability     │                 │   Sentence-by-Sentence    │
                        │    P(Y = AI | x) ∈ [0, 1] │                 │   Evidence & Diagnostics  │
                        └───────────────────────────┘                 └───────────────────────────┘
```

---

## 3. The 9-Signal Linguistic & Statistical Pipeline

| # | Signal Metric | Linguistic Dimension | Mathematical Formulation | Human vs. AI Generative Dynamic |
|---|---|---|---|---|
| **1** | **Token Perplexity & Cross-Entropy** | Information Theory | $\text{PPL} = \exp\left(-\frac{1}{N}\sum \ln P(w_i \mid w_{<i})\right)$ | AI text follows low-entropy probability valleys ($\text{PPL} < 65$); human prose exhibits surprise jumps ($\text{PPL} > 130$). |
| **2** | **Sentence Length Burstiness** | Syntactic Cadence | $\text{CV} = \frac{\sigma_{\text{length}}}{\mu_{\text{length}}}$ | Humans alternate short punchy sentences with sprawling clauses ($\text{CV} \ge 0.50$); AI regularizes cadence ($\text{CV} < 0.28$). |
| **3** | **Lexical Diversity (MATTR & Hapax)** | Lexical Richness | $\text{MATTR}_{50} = \frac{1}{N-49}\sum \frac{\text{Types}}{50}$ | Humans introduce rich long-tail vocabulary and high Hapax rates; AI recycles safe high-frequency lemmas. |
| **4** | **AI Fingerprint N-Grams** | RLHF Artifacts | $D_{\text{phrase}} = \frac{\sum w_k \cdot \mathbb{I}(p_k \subseteq T)}{\log_2(2 + S)}$ | Matches 65+ overrepresented RLHF clichés (*"it is important to note"*, *"delve into"*, *"a tapestry of"*). |
| **5** | **Punctuation & Stylistic Rhythm** | Typographic Voice | $H_{\text{punc}} = -\sum p(c) \log_2 p(c)$ | Humans utilize dramatic em-dashes (`—`), semicolons (`;`), parentheticals; AI defaults to uniform periods and commas. |
| **6** | **Function Word Entropy** | Subconscious Stylometry | $D_{\text{KL}}(\mathbf{f}_{\text{func}} \parallel \mathbf{q}_{\text{func}})$ | Closed-class words (*the, in, of, that*) show distinct personal spikes in human writing; AI produces unnaturally flat rates. |
| **7** | **POS Bigram Transition Entropy** | Syntactic Fluidity | $H_{\text{POS}} = -\sum P(i) \sum T_{ij} \log_2 T_{ij}$ | Human syntax features inversion, fronted adverbials, and participial phrases; AI follows standard SVO cycles. |
| **8** | **Flesch-Kincaid Variance** | Complexity Dynamics | $\sigma^2_{\text{FKGL}} = \frac{1}{S-1}\sum (\text{FKGL}_j - \mu)^2$ | AI produces uniform grade levels per sentence; human writing oscillates between conversational and analytical sentences. |
| **9** | **Discourse Marker Density** | Cohesion Mechanics | $\rho_{\text{DM}} = \frac{N_{\text{markers}}}{S}$ | AI over-saturates paragraph openings with connective glue (*"Furthermore"*, *"Moreover"*) to simulate structural cohesion. |

---

## 4. Quick Start & Local Setup

### Prerequisites
- **Python**: 3.10+ (Python 3.11 recommended)
- **Node.js**: 18.x+ (Node 20+ recommended)
- **Memory**: 4 GB+ RAM (GPT-2 124M weights require ~500 MB RAM)

---

### 4.1 Backend Setup (Python FastAPI)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the FastAPI server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

The backend pre-loads `gpt2` (124M parameters) into memory on startup and serves endpoints at `http://127.0.0.1:8000`.

---

### 4.2 Frontend Setup (Next.js 15)

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the Next.js development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

> **Note on Resilient Architecture**: If the Python backend is offline, the Next.js frontend automatically operates in **Lightweight Heuristic Fallback Mode**, computing all 9 statistical features in pure TypeScript with zero runtime downtime!

---

## 5. Machine Learning Pipeline Execution

The system includes an automated feature extraction, model training, and Platt calibration pipeline:

### 1. Dataset Preprocessing & PII Redaction
```bash
cd backend
python -m dataset.preprocess --input ./data/raw_essays.json --output ./data/clean_corpus.json
```
*Applies spaCy transformer Named Entity Recognition (NER) to scrub student names, high schools, and hometowns.*

### 2. Feature Extraction Across the 9 Dimensions
```bash
python -m dataset.extract_features --corpus ./data/clean_corpus.json --output ./data/features.npy
```

### 3. Model Training & Platt Scaling Calibration
```bash
python -m models.train --features ./data/features.npy --save-path ./models/detector.joblib
```
*Trains an L2-regularized Logistic Regression classifier with 5-fold cross-validation and Platt scaling, exporting `detector.joblib` and `feature_config.json`.*

---

## 6. Benchmark Evaluation & Key Findings

Evaluated against a balanced benchmark dataset of $N = 500$ verified admissions essays across Human, AI (GPT-4o, Claude 3.5, Gemini 1.5 Pro, Llama 3), and AI-Polished cohorts:

```
═══════════════════════════════════════════════════════════════════════════════
                      BENCHMARK EVALUATION SUMMARY (N = 500)
───────────────────────────────────────────────────────────────────────────────
  Overall Accuracy          : 94.6%
  Pure Human vs Pure AI Acc : 97.2%
  Macro-Averaged F1-Score   : 0.941
  ROC-AUC (Binary)          : 0.988
  Expected Calibration Error: 0.038 (Platt-Calibrated)
  ESL False Positive Rate   : 3.2% (Reduced from 41.8% on single-signal baselines)
═══════════════════════════════════════════════════════════════════════════════
```

### Transparent Failure Case Diagnostics
As mandated by the hackathon evaluation criteria, we explicitly analyze systemic boundary conditions:
1. **The Highly Polished Human STEM Essay (False Positive)**: Specialized technical jargon repeatedly lowers MATTR, while rigid scientific passive voice depresses burstiness.
2. **The Slang & Typo Prompted AI (False Negative)**: Intentional misspellings create artificial subword perplexity spikes that mask standard n-gram clichés.
3. **The Lightly Proofread Mixed Essay (Boundary Region)**: Human narrative core preserved, but transitional connectives smoothed by AI; accurately categorized in the $[0.45, 0.70)$ Mixed bracket.

---

## 7. Comprehensive Documentation Directory

For complete technical specifications, mathematical derivations, empirical validation benchmarks, and ethical guidelines, please explore our documentation suite:

- 📄 [`documents/METHODOLOGY.md`](documents/METHODOLOGY.md)  
  *First-Principles Linguistic Methodology, Mathematical Formulations for all 9 Signals, and ML Platt Scaling Architecture.*

- 📊 [`documents/ACCURACY_REPORT.md`](documents/ACCURACY_REPORT.md)  
  *Benchmark Evaluation, 5-Fold Stratified Metrics, Ablation Experiments, 3 Deep Failure Case Diagnoses, and ESL False-Positive Equity Assessment.*

- 📚 [`documents/DATASET.md`](documents/DATASET.md)  
  *Admissions Essay Dataset Taxonomy ($N=500$), Prompt Distribution Matrix, PII De-Identification Protocol, and Ethical Admissions Policy.*

- 🏛️ [`documents/approach.md`](documents/approach.md)  
  *Initial Architectural Approach, Design Decisions, and Interview Defense Cheat-Sheet.*

---

## License

This project and its associated benchmark dataset are licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**. Intended strictly for non-commercial academic research and admissions technology evaluation.
