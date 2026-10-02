<div align="center">

# 🧬 TwinLife AI

### Autonomous Multi-Domain Digital Twin Platform for Personalized Health, Finance & Insurance

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-646CFF?logo=vite&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?logo=google&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00)
![Status](https://img.shields.io/badge/Status-Demo%20%2F%20Research-blueviolet)

**One twin. Three domains. One answer.**

*"Can I afford a ₹5 lakh treatment, and does my insurance cover it?"*

</div>

---

## 📖 Overview

**TwinLife AI** is an intelligent digital-twin platform that builds a unified representation of an individual's **health**, **financial situation**, and **insurance profile**.

It combines:

| Layer | Technologies / Capabilities |
|---|---|
| 🧮 **Deterministic Engine** | Health, finance & insurance formulas |
| 🤖 **Machine Learning** | Random Forest risk prediction |
| 🔍 **Explainability** | SHAP feature attribution |
| 🧠 **Autonomous Agent** | Gemini AI with function calling |
| 📚 **RAG** | Guideline retrieval with ChromaDB |
| 🔀 **Orchestration** | LangGraph workflow routing |
| 💬 **Narration** | AI-generated, grounded explanations |
| 📊 **Dashboard** | Interactive React + Vite UI |

> 🎯 **Main goal:** connect domains that are normally handled in isolation, and produce a **cross-domain analysis** from the user's health context, financial capacity, and insurance coverage.

### 🗺️ High-Level Flow

```text
                    ┌─────────────────────┐
                    │      User Input     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   React Frontend    │
                    │       + Vite        │
                    └──────────┬──────────┘
                               │
                         REST API / JSON
                               │
                               ▼
                    ┌─────────────────────┐
                    │   FastAPI Backend   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   TwinLife Service  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ LangGraph           │
                    │ Orchestrator        │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   │ Health Twin │      │ Finance Twin│      │ Insurance   │
   │             │      │             │      │ Twin        │
   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
        ┌──────────────┐              ┌──────────────┐
        │ RAG Agent    │              │ Gemini Agent │
        │ ChromaDB     │              │ Simulation   │
        └──────┬───────┘              └──────┬───────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
                   ┌─────────────────────┐
                   │  Gemini Narration   │
                   │  + Recommendations  │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Interactive Results │
                   │      Dashboard      │
                   └─────────────────────┘
```

---

## 📑 Table of Contents

- [🎯 Problem Statement](#-problem-statement)
- [🏁 Objectives](#-objectives)
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🧩 Module Breakdown](#-module-breakdown)
- [🛠️ Technology Stack](#️-technology-stack)
- [🚀 Installation & Setup](#-installation--setup)
- [🔐 Environment Variables](#-environment-variables)
- [▶️ Running the Project](#️-running-the-project)
- [🔌 API Endpoints](#-api-endpoints)
- [💡 Example Queries](#-example-queries)
- [📂 Project Structure](#-project-structure)
- [🧪 Testing](#-testing)
- [🖥️ Frontend Highlights](#️-frontend-highlights)
- [⚠️ Current Limitations](#️-current-limitations)
- [🔮 Future Enhancements](#-future-enhancements)
- [📜 Disclaimer](#-disclaimer)
- [📄 License](#-license)

---

## 🎯 Problem Statement

Health, finance, and insurance decisions are usually handled **independently**:

- 🏥 A **health app** may flag a medical risk without checking whether the person can afford treatment.
- 💰 A **finance app** may analyze income, expenses, and savings without accounting for future medical costs.
- 🛡️ An **insurance app** may show coverage without considering the individual's actual health and financial situation.

**TwinLife AI closes this gap** by combining all three domains into a single digital-twin system.

---

## 🏁 Objectives

1. Build a digital representation of a user's **health profile**.
2. Analyze **financial stability** and affordability.
3. Evaluate **insurance adequacy**.
4. Predict health risks using **Machine Learning**.
5. Explain ML predictions using **SHAP**.
6. Retrieve relevant health and financial guidelines using **RAG**.
7. Route user queries through an **intelligent orchestrator**.
8. Simulate **treatment affordability** and insurance coverage.
9. Generate **personalized AI explanations**.
10. Provide **recommendations** and **monitoring** capabilities.

---

## ✨ Key Features

### 🩺 Health Analysis

- BMI calculation & classification
- Blood pressure staging
- Blood glucose & HbA1c analysis
- Cardiovascular risk prediction
- Diabetes risk prediction
- Smoking & physical activity analysis
- Overall health score
- SHAP-based feature importance

### 💰 Finance Analysis

- Monthly income analysis
- Fixed & variable expense analysis
- EMI analysis
- Disposable income
- Debt-to-income (DTI) ratio
- Savings rate
- Emergency fund estimation
- Financial stability score

### 🛡️ Insurance Analysis

- Sum insured analysis
- Coverage adequacy ratio
- Premium affordability ratio
- Insurance adequacy score
- Rider gap identification
- Insurance coverage simulation

### 🤖 Autonomous Treatment Simulation

The system can answer questions such as:

> **"Can I afford a Rs.5 lakh surgery?"**

The autonomous agent evaluates:

| Factor | Description |
|---|---|
| 💵 Treatment cost | The estimated cost entered by the user |
| 📈 Financial affordability | Can the user self-fund it? |
| 🛡️ Insurance coverage | What the policy actually pays |
| 🕳️ Coverage gap | The remaining out-of-pocket amount |
| 🩺 Health context | Risk profile relevant to the treatment |
| ⏱️ Urgency | How time-critical the treatment is |
| 💡 Alternatives | Cheaper options and recommendations |

### 📚 RAG-Based Knowledge

The system retrieves relevant guideline chunks from a local knowledge base. The frontend displays:

- **Guidelines Used**
- **Retrieved Knowledge**
- **Top 3 relevant guideline chunks**

This makes the RAG process **transparent** during demonstrations.

### 🔍 Explainable AI

Health predictions include **SHAP-based top contributing factors**, for example:

```text
1. HbA1c
2. Blood Glucose
3. Age
4. Systolic BP
5. BMI
```

### 📊 Interactive Dashboard

The React dashboard shows:

- Health, Finance & Insurance scores
- Health metrics, financial breakdown & insurance details
- Treatment simulation
- RAG sources
- AI explanation

### ✅ Validation & Error Handling

The frontend validates:

- Health value ranges
- Financial values
- Insurance values
- Expense-to-income consistency
- Premium-to-income consistency

It also provides a **loading state**, **backend connection error handling**, and **invalid input feedback**.

---

## 🏗️ System Architecture

TwinLife AI follows a **modular architecture**.

```text
                     USER
                       │
                       ▼
              ┌─────────────────┐
              │  React + Vite   │
              │    Frontend     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     FastAPI     │
              │    REST API     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ TwinLifeService │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    LangGraph    │
              │   Orchestrator  │
              └────────┬────────┘
                       │
       ┌───────────────┼────────────────┐
       │               │                │
       ▼               ▼                ▼
  Health Twin    Finance Twin    Insurance Twin
       │               │                │
       └───────────────┼────────────────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        ML + SHAP              RAG
             │                   │
             │               ChromaDB
             │                   │
             └─────────┬─────────┘
                       ▼
                 Gemini Agent
                       │
                       ▼
                 AI Narration
                       │
                       ▼
                   Dashboard
```

---

## 🧩 Module Breakdown

<details>
<summary><b>📐 Module 1 — Deterministic Formulas</b></summary>

<br>

Deterministic calculations for all three domains.

**Health**
- BMI
- WHO BMI classification
- Blood pressure staging
- Framingham cardiovascular risk
- Blood glucose indicators

**Finance**
- Debt-to-income ratio
- Savings rate
- Emergency fund
- Financial stability
- Treatment affordability

**Insurance**
- Coverage adequacy
- Premium affordability
- Composite insurance adequacy
- Rider gap analysis

</details>

<details>
<summary><b>🤖 Module 2 — Machine Learning & Explainability</b></summary>

<br>

Two **Random Forest** models are used for health-risk prediction.

#### ❤️ Cardiovascular Risk Model

| Property | Value |
|---|---|
| Dataset | `data/cardio_train.csv` |
| Size | 70,000 rows |
| Model | Random Forest |
| Trees | 200 |
| Max depth | 10 |
| Min samples per leaf | 20 |

| Metric | Score |
|---|---|
| Accuracy | **73.6%** |
| F1 Score | **72.0%** |
| ROC-AUC | **80.3%** |

#### 🩸 Diabetes Risk Model

| Property | Value |
|---|---|
| Dataset | `data/diabetes_prediction_dataset.csv` |
| Size | 100,000 rows |
| Preprocessing | Duplicate records removed |
| Model | Random Forest |
| Class balancing | Applied during training |

| Metric | Score |
|---|---|
| Accuracy | **90.2%** |
| F1 Score | **61.7%** |
| ROC-AUC | **97.5%** |

#### 🔍 SHAP

SHAP explains model predictions and identifies the most influential features. The dashboard displays the **top contributing factors**.

</details>

<details>
<summary><b>👥 Module 3 — Digital Twins</b></summary>

<br>

| Twin | Responsibilities |
|---|---|
| 🩺 **Health Twin** | Health scoring, BMI, blood pressure, cardiovascular risk, diabetes risk, health indicators, SHAP explanations |
| 💰 **Finance Twin** | Financial stability, DTI, savings, emergency fund, disposable income, affordability |
| 🛡️ **Insurance Twin** | Insurance adequacy, coverage ratio, premium affordability, rider analysis, coverage gaps |

</details>

<details>
<summary><b>🧠 Module 4 — Autonomous Treatment Simulator</b></summary>

<br>

The treatment simulator uses a **Gemini-based tool-use agent** with access to deterministic tools:

```text
check_affordability
check_insurance_coverage
get_cheaper_alternative
get_health_context
```

**Example**

> **User:** Can I afford a Rs.5 lakh surgery?

```text
Treatment Cost
      ↓
Financial Affordability
      ↓
Insurance Coverage
      ↓
Coverage Gap
      ↓
Health Context
      ↓
Alternative / Recommendation
```

The frontend exposes the tool results through the **Treatment Simulation** dashboard.

</details>

<details>
<summary><b>📚 Module 5 — RAG Knowledge System</b></summary>

<br>

TwinLife AI uses **Retrieval-Augmented Generation** to ground AI explanations in guideline documents.

#### Knowledge Sources

| Document | Source | Main Topics |
|---|---|---|
| `cardiovascular_guidelines.txt` | AHA/ACC | BP, cardiovascular risk, lifestyle |
| `diabetes_management_guidelines.txt` | ADA | Glucose, HbA1c, diabetes |
| `financial_planning_guidelines.txt` | RBI/SEBI | Financial planning |
| `insurance_planning_guidelines.txt` | IRDAI | Insurance planning |
| `general_wellness_guidelines.txt` | WHO/ICMR | BMI, exercise, nutrition, wellness |

#### Pipeline

```text
Guideline Documents
        ↓
Text Chunking
        ↓
Sentence Transformer (all-MiniLM-L6-v2)
        ↓
ChromaDB
        ↓
Semantic Search
        ↓
Top 3 Relevant Chunks
        ↓
Gemini Narration
```

The frontend displays the retrieved chunks so users can see exactly which knowledge was used.

</details>

<details>
<summary><b>🔀 Module 6 — LangGraph Orchestrator</b></summary>

<br>

The orchestrator coordinates the complete TwinLife workflow:

```text
User Query
    ↓
Query Classification / Routing
    ↓
Health Assessment
    ↓
Finance Assessment
    ↓
Insurance Assessment
    ↓
RAG Retrieval
    ↓
Treatment Simulation (when required)
    ↓
AI Narration
    ↓
Final Response
```

It connects the individual modules into one **end-to-end pipeline**.

</details>

<details>
<summary><b>💬 Module 7 — AI Narration</b></summary>

<br>

The narration layer converts technical results into understandable explanations by combining:

- Twin outputs
- SHAP features
- RAG chunks
- Simulation results

It is designed to **avoid inventing numerical values** and to use the actual computed results.

For treatment simulations, narration distinguishes between:

| Metric | Meaning |
|---|---|
| **Overall Insurance Adequacy** | General strength of the user's policy relative to income |
| **Specific Treatment Coverage** | What the policy covers for *this* treatment |

> This prevents the overall insurance ratio from being incorrectly presented as the treatment-specific coverage gap.

</details>

<details>
<summary><b>📡 Module 8 — Monitoring</b></summary>

<br>

The monitoring module uses **APScheduler** for scheduled checks. It can evaluate changes in:

- Health score
- DTI ratio
- Savings rate
- Insurance score
- Insurance rider gaps

Snapshots and notification-related data are handled through the project's monitoring implementation.

</details>

<details>
<summary><b>🎯 Module 9 — Recommendation Engines</b></summary>

<br>

Rule-based recommendation and simulation engines:

| Engine | What it does |
|---|---|
| **Investment Recommendation** | Uses emergency fund, financial stability, and age |
| **Purchase Impact Simulation** | Analyzes DTI impact, savings time, and emergency fund impact |
| **Lifestyle Impact Simulation** | Evaluates potential effects of lifestyle changes on health outcomes |

</details>

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| **🎨 Frontend** | React, Vite, HTML, CSS, JavaScript |
| **⚙️ Backend** | Python, FastAPI, Uvicorn, REST APIs |
| **🤖 AI / ML** | Scikit-learn, Random Forest, SHAP, Google Gemini API, LangGraph, Agentic AI, Function Calling |
| **📚 RAG & Knowledge** | ChromaDB, Sentence Transformers, `all-MiniLM-L6-v2`, Guideline documents |
| **📡 Monitoring** | APScheduler |
| **🗃️ Data Processing** | Pandas, NumPy |

---

## 🚀 Installation & Setup

### 1️⃣ Clone the repository

```bash
https://github.com/Thahaseena0517/TwinLife1.git
cd Twin-Life
```

### 2️⃣ Create the Conda environment

```bash
conda create -n twin_life python=3.12 -y
conda activate twin_life
```

### 3️⃣ Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Install frontend dependencies
new anaconda command prompt open
```bash
cd frontend
npm install
```

---

## 🔐 Environment Variables

Create a `.env` file in the **project root**:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> ⚠️ **Never commit your real API key to GitHub.** Make sure `.env` is listed in `.gitignore`.

---

## ▶️ Running the Project

### 🔹 Start the Backend

From the project root:

```bash
conda activate twin_life
uvicorn main:app --reload
```

| Resource | URL |
|---|---|
| Backend | http://127.0.0.1:8000 |
| API Docs (Swagger) | http://127.0.0.1:8000/docs |

### 🔹 Start the Frontend

Open a **second terminal**:

```bash
cd frontend
npm run dev
```

| Resource | URL |
|---|---|
| Frontend | http://localhost:5173 |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|:---:|---|---|
| `GET` | `/` | Health check |
| `POST` | `/query` | Run a natural-language query through the orchestrator |
| `GET` | `/affordability/{cost}` | Treatment affordability analysis |
| `GET` | `/coverage/{cost}` | Insurance coverage simulation |
| `GET` | `/recommendations` | Get recommendations |
| `GET` | `/monitor` | Run monitoring checks |

<details>
<summary><b>📥 Request / Response examples</b></summary>

<br>

**`GET /`**

```json
{
  "message": "TwinLife AI API is running"
}
```

**`POST /query`**

```json
{
  "query": "What is my health status?"
}
```

**`GET /affordability/500000`**

```text
GET /affordability/500000
```

**`GET /coverage/500000`**

```text
GET /coverage/500000
```

</details>

---

## 💡 Example Queries

| Domain | Query |
|:---:|---|
| 🩺 **Health** | What is my health status? |
| 🩺 **Health** | What is my cardiovascular risk? |
| 🩺 **Health** | What is my diabetes risk? |
| 💰 **Finance** | How is my financial situation? |
| 💰 **Finance** | Can I afford a Rs.5 lakh treatment? |
| 🛡️ **Insurance** | What is my insurance coverage? |
| 🛡️ **Insurance** | Is my insurance adequate? |
| 🔗 **Cross-Domain** | Can I afford a Rs.5 lakh surgery? |

A cross-domain query combines:

```text
Health + Finance + Insurance + RAG + Autonomous Simulation + AI Narration
```

---

## 📂 Project Structure

```text
Twin-Life/
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── App.jsx
│   ├── App.css
│   ├── package.json
│   └── vite.config.js
│
├── src/
│   ├── __init__.py
│   └── services/
│       ├── __init__.py
│       └── twinlife_service.py
│
├── data/
│   ├── cardio_train.csv
│   ├── diabetes_prediction_dataset.csv
│   ├── chroma_db/
│   └── ...
│
├── models/
│   ├── cardio_model.pkl
│   └── diabetes_model.pkl
│
├── health_twin.py
├── finance_twin.py
├── insurance_twin.py
├── combined_simulator.py
├── rag_agent.py
├── orchestrator.py
├── narration_agent.py
├── monitoring_agent.py
├── recommendation_engines.py
├── main.py
├── requirements.txt
├── README.md
└── CHANGES.md
```

---

## 🧪 Testing

### Formula tests

```bash
python -m pytest test_formulas.py
```

### Full test suite

```bash
python -m pytest
```

### Individual module tests

```bash
python health_twin.py
python finance_twin.py
python insurance_twin.py
python rag_agent.py
python orchestrator.py
python monitoring_agent.py
python recommendation_engines.py
```

### Backend smoke test

Start the backend:

```bash
uvicorn main:app --reload
```

Then run:

```bash
python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/').read().decode())"
```

**Expected output:**

```json
{"message":"TwinLife AI API is running"}
```

---

## 🖥️ Frontend Highlights

<details>
<summary><b>🩺 Health Dashboard</b></summary>

<br>

Health score · BMI · BMI category · Blood pressure · BP category · HbA1c · Blood glucose · Cardiovascular risk · Diabetes risk · Smoking · Physical activity · SHAP factors

</details>

<details>
<summary><b>💰 Finance Dashboard</b></summary>

<br>

Financial stability score · Monthly income · Monthly expenses · EMIs · Disposable income · DTI · Savings rate · Emergency fund · Monthly financial breakdown

</details>

<details>
<summary><b>🛡️ Insurance Dashboard</b></summary>

<br>

Insurance adequacy score · Sum insured · Annual premium · Annual income · Coverage ratio · Premium ratio · Rider gaps

</details>

<details>
<summary><b>🏥 Treatment Simulation</b></summary>

<br>

Treatment cost · Insurance covered amount · Coverage gap · Self-funding status · Affordability score · Health risk · Urgency · Agent recommendation

</details>

<details>
<summary><b>📚 RAG Transparency</b></summary>

<br>

```text
Guidelines Used
      ↓
Retrieved Knowledge
      ↓
Top 3 Relevant Guideline Chunks
```

</details>

<details>
<summary><b>⏳ Loading State & Error Handling</b></summary>

<br>

While processing, the interface shows **"Analyzing your digital twin..."** with progress across:

`Health` → `Finance` → `Insurance` → `RAG` → `AI Analysis`

If the backend is unreachable, the frontend shows a **clear error message** instead of failing silently.

</details>

---

## ⚠️ Current Limitations

- 🔸 User profile persistence is **not** currently implemented.
- 🔸 The frontend sends profile information, but the backend request model currently focuses on the `query` field. Full per-request profile propagation should be completed before treating custom profiles as persistent backend state.
- 🔸 The application is intended for **local development / demo** usage.
- 🔸 Medical and financial outputs are **advisory** and should not replace qualified professional advice.
- 🔸 RAG quality depends on the documents available in the local knowledge base.
- 🔸 Model predictions depend on the training datasets and their limitations.

---

## 🔮 Future Enhancements

- [ ] Persistent user profiles
- [ ] Database-backed user management
- [ ] Authentication
- [ ] Historical assessment tracking
- [ ] More advanced visualization
- [ ] Deployment to cloud infrastructure
- [ ] More healthcare datasets
- [ ] More financial planning scenarios
- [ ] Additional insurance products
- [ ] Real-time monitoring notifications
- [ ] Mobile application
- [ ] Voice interaction
- [ ] External health / wearable integrations

---

## 📜 Disclaimer

> **TwinLife AI is an educational and research-oriented project.**
>
> The health information, risk predictions, financial calculations, insurance analysis, and AI-generated recommendations provided by this system are for **informational purposes only**.
>
> They should **not** be considered a substitute for advice from qualified medical, financial, insurance, or other professional advisors.

---

## 📄 License

Add your license here (e.g., MIT). If you haven't chosen one yet, see [choosealicense.com](https://choosealicense.com).

---

<div align="center">

**Built with ❤️ to connect health, finance, and insurance decisions.**

⭐ If you find this project useful, consider giving it a star!

</div>
