# TwinLife AI — Autonomous Multi-Domain Digital Twin Platform

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Gemini API](https://img.shields.io/badge/LLM-Google%20Gemini-orange.svg)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TwinLife AI** is an intelligent digital-twin platform that models an individual's **Health**, **Financial**, and **Insurance** profiles. It combines machine learning risk prediction models (Random Forest, SHAP explainability), deterministic clinical and financial formulas, an autonomous LLM tool-use agent, RAG-based guideline retrieval, a LangGraph orchestrator, proactive monitoring, and AI-narrated explanations to deliver a comprehensive wellness advisory system.

---

## 📌 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Module Breakdown](#-module-breakdown)
- [Installation & Setup](#-installation--setup)
- [Quick Start & Command Guide](#-quick-start--command-guide)
- [Autonomous Agent Workflow](#-autonomous-agent-workflow)
- [RAG Pipeline](#-rag-pipeline)
- [Orchestrator Routing](#-orchestrator-routing)
- [Health Score Calculation](#-health-score-calculation)
- [License & Disclaimer](#-license--disclaimer)

---

## 🏗️ Overview & Architecture

TwinLife AI operates on a modular pipeline where deterministic mathematical formulas and trained ML models power higher-level "Digital Twins". An autonomous AI agent interacts with these twin interfaces via function calling, while a LangGraph orchestrator routes queries and a narration layer provides human-friendly explanations grounded in retrieved medical/financial guidelines.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      MODULE 6: ORCHESTRATOR (LangGraph)                 │
│         Routes queries → Twins → RAG → Simulation → Narration          │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
     ┌───────────────────────────────┼───────────────────────────────┐
     ▼                               ▼                               ▼
┌─────────────┐   ┌──────────────────────────────────┐   ┌─────────────┐
│  MODULE 7   │   │      MODULE 4: COMBINED           │   │  MODULE 5   │
│  Narration  │   │      SIMULATOR (Gemini Agent)      │   │  RAG Agent  │
│  Agent      │   └────────────────┬─────────────────┘   │  (ChromaDB) │
└─────────────┘                    │ (4 Deterministic     └─────────────┘
                                   │  Tool Calls)
     ┌─────────────────────────────┼─────────────────────────────┐
     ▼                             ▼                             ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│    MODULE 3A      │   │    MODULE 3B      │   │    MODULE 3C      │
│    HealthTwin     │   │    FinanceTwin    │   │   InsuranceTwin   │
└────────┬──────────┘   └────────┬──────────┘   └────────┬──────────┘
    ┌────┴────┐                  │                       │
    ▼         ▼                  │                       │
┌────────┐┌────────┐             │                       │
│MODULE 2││MODULE 2│             │                       │
│Cardio  ││Diabetes│             │                       │
│Model   ││Model   │             │                       │
└───┬────┘└───┬────┘             │                       │
    └─────────┴──────────┬───────┴───────────────────────┘
                         ▼
            ┌─────────────────────────┐
            │        MODULE 1         │     ┌────────────────────┐
            │ Pure Domain Formulas    │     │ MODULE 8: Monitor  │
            │ (formulas.py)           │     │ (APScheduler)      │
            └─────────────────────────┘     └────────────────────┘
                                            ┌────────────────────┐
                                            │ MODULE 9: Engines  │
                                            │ (Recommendations)  │
                                            └────────────────────┘
```

---

## 📦 Module Breakdown

### Module 1: Domain Formulas (`formulas.py`)
- **Pure, deterministic functions** with 0 external side-effects or I/O.
- **Health**: Body Mass Index (BMI), WHO Obesity Staging, Blood Pressure Staging (AHA/ACC 2017), Framingham 10-Year Cardiovascular Disease Risk, Blood Glucose Diabetes Indicator.
- **Finance**: Debt-to-Income (DTI) Ratio, Savings Rate, Emergency Fund Coverage (Months), Financial Stability Score (0-100), Treatment Affordability Score.
- **Insurance**: Coverage Adequacy Ratio, Premium Affordability Ratio, Composite Insurance Adequacy Score (0-100).
- **Unit Tests**: Full test suite in `test_formulas.py` (70 unit tests passing).

### Module 2: Health ML Models (`train_cardio_model.py` & `train_diabetes_model.py`)
- **Cardiovascular Disease Classifier**:
  - Dataset: `data/cardio_train.csv` (70,000 rows).
  - Preprocessing: Age converted from days to years (`age // 365`), BMI derived, physiological filtering (`ap_lo <= ap_hi`).
  - Model: `RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=20)`.
  - Metrics: **Accuracy: 73.6% | F1-Score: 72.0% | ROC-AUC: 80.3%**. Saved to `models/cardio_model.pkl`.
- **Diabetes Classifier**:
  - Dataset: `data/diabetes_prediction_dataset.csv` (100,000 rows).
  - Preprocessing: Deduplication (3,854 duplicate rows removed), One-hot encoding for categorical variables, class balancing.
  - Model: `RandomForestClassifier(class_weight='balanced')`.
  - Metrics: **Accuracy: 90.2% | F1-Score: 61.7% | ROC-AUC: 97.5%**. Saved to `models/diabetes_model.pkl`.

### Module 3: Digital Twin Classes (`health_twin.py`, `finance_twin.py`, `insurance_twin.py`)
- **`HealthTwin`**: Integrates both ML model predictions, SHAP feature importance tree explainers, and Module 1 formulas into a unified health profile and a 0-100 composite health score.
- **`FinanceTwin`**: Evaluates financial stability, debt obligations, emergency funds, and computes 12-month treatment affordability (`affordability_for(cost)`).
- **`InsuranceTwin`**: Estimates 5-year condition-specific medical costs, checks sum insured limits, and performs rider gap analysis (e.g. Critical Illness, Diabetes Care, Hospital Cash riders).

### Module 4: Autonomous Combined Simulator (`combined_simulator.py`)
- **Agent Architecture**: Autonomous function-calling agent using the Google Gemini API (`google-genai` SDK).
- **Deterministic Tools Registered**:
  1. `check_affordability(cost)`: Evaluates user's financial capacity over 12 months.
  2. `check_insurance_coverage(cost)`: Computes covered amount and out-of-pocket gap.
  3. `get_cheaper_alternative(condition, current_cost)`: Looks up next lower-cost medical tier.
  4. `get_health_context(condition)`: Retrieves clinical risk level and urgency.
- **Explainability**: Every tool execution is recorded in an ordered `tool_trace` for full transparency.

### Module 5: RAG Pipeline (`rag_agent.py`)
- **Document Indexing**: Chunks 5 guideline documents (cardiovascular, diabetes, financial planning, insurance, general wellness) using LangChain's `RecursiveCharacterTextSplitter` (~2000 chars/chunk, 200 overlap).
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no API key needed).
- **Vector Store**: ChromaDB with persistent local storage (`data/chroma_db/`).
- **Retrieval**: `retrieve(topic, top_k=3)` returns the most semantically relevant guideline chunks for any query.
- **Guideline Sources**: AHA/ACC cardiovascular guidelines, ADA diabetes standards, RBI/SEBI financial planning, IRDAI insurance guidelines, WHO/ICMR wellness guidelines.

### Module 6: Orchestrator (`orchestrator.py`, LangGraph)
- **Two-Step Routing**:
  - **Step 1 (Deterministic)**: Keyword matching against domain-specific terms (e.g., "diabetes" → health, "EMI" → finance, "premium" → insurance, "treatment cost" → all three + simulate).
  - **Step 2 (LLM Fallback)**: For ambiguous queries, calls Gemini API to classify into `{twins, mode, simulate}`.
- **LangGraph State Graph**: Conditional edges route through health → finance → insurance → RAG → simulate → narrate → END based on routing decisions.
- **Full Pipeline**: Processes a natural language query end-to-end through twin assessments, RAG retrieval, optional simulation, and narrated explanation.

### Module 7: Narration Layer (`narration_agent.py`)
- **`generate_explanation(twin_outputs, shap_features, rag_chunks)`**: Calls Gemini API to produce a warm, plain-language explanation combining computed scores, SHAP-driven factor analysis, and cited RAG guideline references.
- **`generate_ranked_whatifs(scenarios)`**: Ranks candidate what-if scenarios using `score = risk_reduction×0.5 + (1/cost)×0.3 + insurance_fit×0.2` and narrates trade-offs.
- **Rules**: No invented numbers, natural source citations, disclaimers included, under 300 words.

### Module 8: Monitoring Agent (`monitoring_agent.py`, APScheduler)
- **Scheduled Checks**: Weekly (configurable) profile recomputation using APScheduler's `BackgroundScheduler`.
- **Threshold Alerts**:
  - Health score drop > 10 points
  - DTI ratio rise above 0.40
  - Savings rate fall below 10%
  - Insurance score drop > 15 points
  - New insurance rider gaps detected
- **Persistence**: File-based snapshots (`data/snapshots/`) and notification logs (`data/notifications/`).

### Module 9: Recommendation Engines (`recommendation_engines.py`)
- **`investment_recommendation(finance_profile, age)`**: Rule-based tier logic — emergency fund first → growth allocation → balanced → conservative → no new investment.
- **`purchase_impact_simulator(finance_profile, purchase_cost)`**: Analyzes DTI impact, months to save, emergency fund depletion. Generates ranked alternatives: defer, cheaper tier, EMI split (24 months), expense reallocation.
- **`lifestyle_impact_simulator(health_profile, habit_changes)`**: Projects health score changes from lifestyle modifications (quit smoking, weight loss, exercise, diet). Generates substitute/moderate/offset alternatives for worsening habits.

---

## ⚡ Installation & Setup

### 1. Clone & Environment Creation
```bash
git clone https://github.com/Vajra-Chaitanya/Twin-Life.git
cd Twin-Life

# Create conda environment
conda create -n twin_life python=3.12 -y
conda activate twin_life

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Gemini API Key
Obtain a free API Key from [Google AI Studio](https://aistudio.google.com/apikey). Create or open the `.env` file in the project root:

```env
GEMINI_API_KEY=AIzaSy...your_api_key_here...
```

---

## 🚀 Quick Start & Command Guide

### Run Module 1 Unit Tests
```bash
python -m pytest test_formulas.py
```

### Train Machine Learning Models (Module 2)
```bash
python train_cardio_model.py
python train_diabetes_model.py
```

### Evaluate Digital Twins (Module 3)
```bash
python health_twin.py
python finance_twin.py
python insurance_twin.py
```

### Run Autonomous Treatment Advisor (Module 4)
```bash
python test_combined_simulator.py
```

### Build & Query RAG Pipeline (Module 5)
```bash
python rag_agent.py
```

### Test Orchestrator Routing (Module 6)
```bash
python orchestrator.py
```

### Generate AI Narrations (Module 7)
```bash
python narration_agent.py
```

### Run Monitoring Agent Check (Module 8)
```bash
python monitoring_agent.py
```

### Test Recommendation Engines (Module 9)
```bash
python recommendation_engines.py
```

---

## 🤖 Autonomous Agent Workflow

When evaluating a treatment plan cost (e.g. ₹5,00,000), the Gemini agent reasons step-by-step without hardcoded rules:

```
[Agent Initiated] → Prompt: Evaluate ₹5,00,000 Cardio Treatment Plan

  ├── Call Tool 1: get_health_context("cardio") → { risk_level: "High", urgency: "Urgent" }
  ├── Call Tool 2: check_affordability(500000) → { affordable: false, score: 0.45 }
  ├── Call Tool 3: check_insurance_coverage(500000) → { covered_amount: 300000, gap: 200000 }
  ├── Call Tool 4: get_cheaper_alternative("cardio", 500000) → { new_cost: 200000, tier: "Moderate" }
  └── Call Tool 5: check_insurance_coverage(200000) → { covered_amount: 200000, gap: 0 }

[Agent Synthesis] → Generates detailed recommendation explaining optimal insurance & self-funding path.
```

---

## 📚 RAG Pipeline

The RAG system indexes 5 authoritative guideline documents into ChromaDB:

| Document | Source | Topics Covered |
|----------|--------|---------------|
| `cardiovascular_guidelines.txt` | AHA/ACC 2019 | BP staging, cholesterol, Framingham risk, lifestyle |
| `diabetes_management_guidelines.txt` | ADA 2024 | Glucose classification, HbA1c targets, pharmacology |
| `financial_planning_guidelines.txt` | RBI/SEBI | DTI management, savings benchmarks, investment allocation |
| `insurance_planning_guidelines.txt` | IRDAI | Sum insured adequacy, premium ratios, rider analysis |
| `general_wellness_guidelines.txt` | WHO/ICMR | BMI, physical activity, nutrition, sleep, screening |

Each query retrieves the top-3 most semantically relevant chunks using cosine similarity over `all-MiniLM-L6-v2` embeddings.

---

## 🔀 Orchestrator Routing

The LangGraph orchestrator uses two-step routing:

| Query Example | Routed To | Mode |
|--------------|-----------|------|
| "What is my blood pressure?" | Health | Single-domain |
| "Am I saving enough?" | Finance | Single-domain |
| "How much is my coverage?" | Insurance | Single-domain |
| "Can I afford ₹5L surgery?" | Health + Finance + Insurance | Cross-domain + Simulate |
| "Treatment for diabetes Rs.3L" | Health + Finance + Insurance | Cross-domain + Simulate |

---

## 🧮 Health Score Calculation

The `overall_health_score` (0-100) in `HealthTwin` is calculated using a 5-component weighted model:

$$\text{Overall Score} = 0.25(\text{BMI Score}) + 0.25(\text{BP Score}) + 0.25(\text{Cardio Score}) + 0.15(\text{Diabetes Score}) + 0.10(\text{Framingham Score})$$

---

## 📜 License & Disclaimer

This software is developed for academic and demonstration purposes only. All financial figures, insurance coverage rules, and health estimates do not constitute licensed medical or financial advice.

Distributed under the MIT License. See `LICENSE` for more information.
