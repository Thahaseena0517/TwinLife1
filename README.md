# TwinLife AI

### Autonomous Multi-Domain Digital Twin Platform for Personalized Health, Finance & Insurance

TwinLife AI is an intelligent digital-twin platform that creates a unified representation of an individual's **health, financial situation, and insurance profile**.

The system combines:

- Deterministic health, finance, and insurance calculations
- Machine Learning risk prediction
- Random Forest models
- SHAP-based explainability
- Autonomous Gemini AI agent with function calling
- RAG-based guideline retrieval
- ChromaDB vector storage
- LangGraph orchestration
- AI-generated explanations
- Treatment affordability and insurance simulation
- Monitoring and recommendation engines
- React-based interactive dashboard

The main goal is to connect domains that are normally handled separately.

For example:

> "Can I afford a ₹5 lakh treatment, and does my insurance cover it?"

TwinLife AI can combine the user's **health context, financial capacity, and insurance coverage** to produce a cross-domain analysis.

---

# Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Module Breakdown](#module-breakdown)
- [Technology Stack](#technology-stack)
- [Machine Learning Models](#machine-learning-models)
- [RAG Pipeline](#rag-pipeline)
- [Autonomous Agent Workflow](#autonomous-agent-workflow)
- [Frontend](#frontend)
- [Backend](#backend)
- [Installation & Setup](#installation--setup)
- [Environment Variables](#environment-variables)
- [Running the Project](#running-the-project)
- [API Endpoints](#api-endpoints)
- [Example Queries](#example-queries)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Current Limitations](#current-limitations)
- [Future Enhancements](#future-enhancements)
- [Disclaimer](#disclaimer)
- [License](#license)

---

# Overview

TwinLife AI models three interconnected domains:

```text
                    ┌─────────────────────┐
                    │      User Input     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   React Frontend    │
                    │     + Vite          │
                    └──────────┬──────────┘
                               │
                         REST API / JSON
                               │
                               ▼
                    ┌─────────────────────┐
                    │     FastAPI         │
                    │      Backend        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ TwinLife Service    │
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
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │ Health Twin │     │ Finance Twin│     │ Insurance   │
   │             │     │             │     │ Twin        │
   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
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
                    │ Gemini Narration    │
                    │ + Recommendations   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Interactive Results │
                    │ Dashboard            │
                    └─────────────────────┘

Problem Statement
Health, finance, and insurance decisions are often handled independently.
A health application may identify a medical risk without considering whether the person can financially manage the treatment.
A financial application may analyze income, expenses, and savings without considering future medical expenses.
An insurance application may show coverage without considering the individual's actual health and financial situation.
TwinLife AI addresses this gap by combining these three domains into one digital-twin system.
Objectives
The main objectives of TwinLife AI are:
1. Build a digital representation of a user's health profile.
2. Analyze financial stability and affordability.
3. Evaluate insurance adequacy.
4. Predict health risks using Machine Learning.
5. Explain ML predictions using SHAP.
6. Retrieve relevant health and financial guidelines using RAG.
7. Route user queries through an intelligent orchestrator.
8. Simulate treatment affordability and insurance coverage.
9. Generate personalized AI explanations.
10. Provide recommendations and monitoring capabilities.
Key Features
Health Analysis
- BMI calculation
- BMI classification
- Blood pressure staging
- Blood glucose analysis
- HbA1c analysis
- Cardiovascular risk prediction
- Diabetes risk prediction
- Smoking analysis
- Physical activity analysis
- Health score
- SHAP-based feature importance
Finance Analysis
- Monthly income analysis
- Fixed expense analysis
- Variable expense analysis
- EMI analysis
- Disposable income
- Debt-to-income ratio
- Savings rate
- Emergency fund estimation
- Financial stability score
Insurance Analysis
- Sum insured analysis
- Coverage adequacy ratio
- Premium affordability ratio
- Insurance adequacy score
- Rider gap identification
- Insurance coverage simulation
Autonomous Treatment Simulation
The system can analyze questions such as:
Can I afford a Rs.5 lakh surgery?

The autonomous agent can evaluate:
- Treatment cost
- Financial affordability
- Insurance coverage
- Coverage gap
- Health context
- Urgency
- Alternative/recommendation
RAG-Based Knowledge
The system retrieves relevant guideline chunks from a local knowledge base.
The frontend displays:
- Guidelines Used
- Retrieved Knowledge
- Top 3 relevant guideline chunks
This makes the RAG process transparent during demonstrations.
Explainable AI
Health predictions include SHAP-based top contributing factors.
Example:
HbA1c
Blood Glucose
Age
Systolic BP
BMI

Interactive Dashboard
The React dashboard displays:
- Health score
- Finance score
- Insurance score
- Health metrics
- Financial breakdown
- Insurance details
- Treatment simulation
- RAG sources
- AI explanation
Validation and Error Handling
The frontend validates:
- Health ranges
- Financial values
- Insurance values
- Expense-to-income consistency
- Premium-to-income consistency
The interface also provides:
- Loading state
- Backend connection error handling
- Invalid input feedback
System Architecture
TwinLife AI follows a modular architecture.
                     USER
                       │
                       ▼
              ┌─────────────────┐
              │ React + Vite    │
              │ Frontend        │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ FastAPI         │
              │ REST API        │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ TwinLifeService │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ LangGraph       │
              │ Orchestrator    │
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
        ML + SHAP             RAG
             │                   │
             │             ChromaDB
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

Module Breakdown
Module 1 — Deterministic Formulas
The first module contains deterministic calculations for the three domains.
Health
- BMI
- WHO BMI classification
- Blood pressure staging
- Framingham cardiovascular risk
- Blood glucose indicators
Finance
- Debt-to-income ratio
- Savings rate
- Emergency fund
- Financial stability
- Treatment affordability
Insurance
- Coverage adequacy
- Premium affordability
- Composite insurance adequacy
- Rider gap analysis
Module 2 — Machine Learning & Explainability
Two Random Forest models are used for health risk prediction.
Cardiovascular Risk Model
Dataset:
data/cardio_train.csv

Dataset size:
70,000 rows

Model:
Random Forest

Configuration includes:
200 trees
Maximum depth = 10
Minimum samples leaf = 20

Reported evaluation:
Accuracy  = 73.6%
F1 Score  = 72.0%
ROC-AUC   = 80.3%

Diabetes Risk Model
Dataset:
data/diabetes_prediction_dataset.csv

Dataset size:
100,000 rows

Duplicate records were removed before training.
Model:
Random Forest

Class balancing was applied during training.
Reported evaluation:
Accuracy  = 90.2%
F1 Score  = 61.7%
ROC-AUC   = 97.5%

SHAP
SHAP is used to explain model predictions and identify the most influential features.
The dashboard displays the top contributing factors.
Module 3 — Digital Twins
TwinLife AI contains three domain-specific digital twins.
Health Twin
Responsible for:
- Health scoring
- BMI
- Blood pressure
- Cardiovascular risk
- Diabetes risk
- Health-related indicators
- SHAP explanations
Finance Twin
Responsible for:
- Financial stability
- DTI
- Savings
- Emergency fund
- Disposable income
- Affordability
Insurance Twin
Responsible for:
- Insurance adequacy
- Coverage ratio
- Premium affordability
- Rider analysis
- Coverage gaps
Module 4 — Autonomous Treatment Simulator
The treatment simulator uses a Gemini-based tool-use agent.
The agent has access to deterministic tools including:
check_affordability
check_insurance_coverage
get_cheaper_alternative
get_health_context

Example:
User:
Can I afford a Rs.5 lakh surgery?

The agent can evaluate:
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

The frontend exposes the tool results through the Treatment Simulation dashboard.
Module 5 — RAG Knowledge System
TwinLife AI uses Retrieval-Augmented Generation to ground AI explanations in guideline documents.
Knowledge Sources
The knowledge base contains documents covering:
Document	Source	Main Topics
cardiovascular_guidelines.txt	AHA/ACC	BP, cardiovascular risk, lifestyle
diabetes_management_guidelines.txt	ADA	Glucose, HbA1c, diabetes
financial_planning_guidelines.txt	RBI/SEBI	Financial planning
insurance_planning_guidelines.txt	IRDAI	Insurance planning
general_wellness_guidelines.txt	WHO/ICMR	BMI, exercise, nutrition, wellness


RAG Pipeline
Guideline Documents
        ↓
Text Chunking
        ↓
Sentence Transformer
        ↓
all-MiniLM-L6-v2
        ↓
ChromaDB
        ↓
Semantic Search
        ↓
Top 3 Relevant Chunks
        ↓
Gemini Narration

The frontend also displays the retrieved chunks so that users can see the knowledge used by the system.
Module 6 — LangGraph Orchestrator
The orchestrator coordinates the complete TwinLife workflow.
A typical flow is:
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

The orchestrator connects the individual modules into one end-to-end pipeline.
Module 7 — AI Narration
The narration layer converts technical results into understandable explanations.
It combines:
- Twin outputs
- SHAP features
- RAG chunks
- Simulation results
The narration layer is designed to avoid inventing numerical values and to use the actual computed results.
For treatment simulations, the narration distinguishes between:
Overall Insurance Adequacy

and
Specific Treatment Coverage

This prevents the overall insurance ratio from being incorrectly presented as the treatment-specific coverage gap.
Module 8 — Monitoring
The monitoring module uses APScheduler for scheduled checks.
Monitoring can evaluate changes in:
- Health score
- DTI ratio
- Savings rate
- Insurance score
- Insurance rider gaps
Snapshots and notification-related data are handled through the project's monitoring implementation.
Module 9 — Recommendation Engines
The recommendation module contains rule-based recommendation and simulation engines.
Examples include:
Investment Recommendation
Uses financial conditions such as:
- Emergency fund
- Financial stability
- Age
Purchase Impact Simulation
Analyzes:
- DTI impact
- Savings time
- Emergency fund impact
Lifestyle Impact Simulation
Evaluates potential effects of lifestyle changes on health outcomes.
Technology Stack
Frontend
- React
- Vite
- HTML
- CSS
- JavaScript
Backend
- Python
- FastAPI
- Uvicorn
- REST APIs
AI / ML
- Scikit-learn
- Random Forest
- SHAP
- Google Gemini API
- LangGraph
- Agentic AI
- Function Calling
RAG & Knowledge
- ChromaDB
- Sentence Transformers
- all-MiniLM-L6-v2
- Guideline documents
Monitoring
- APScheduler
Data Processing
- Pandas
- NumPy
Installation & Setup
1. Clone the Repository
git clone https://github.com/Vajra-Chaitanya/Twin-Life.git
cd Twin-Life

2. Create the Conda Environment
conda create -n twin_life python=3.12 -y
conda activate twin_life

3. Install Python Dependencies
pip install -r requirements.txt

4. Install Frontend Dependencies
cd frontend
npm install

Return to the project root:
cd ..

Environment Variables
Create a .env file in the project root.
GEMINI_API_KEY=your_gemini_api_key_here

Never commit your actual API key to GitHub.
Running the Project
Start the Backend
From:
D:\3\Twin-Life

run:
conda activate twin_life
uvicorn main:app --reload

The backend will run at:
http://127.0.0.1:8000

API documentation is available at:
http://127.0.0.1:8000/docs

Start the Frontend
Open another terminal:
cd frontend
npm run dev

The Vite development server will normally run at:
http://localhost:5173

API Endpoints
Root
GET /

Returns:
{
  "message": "TwinLife AI API is running"
}

Query
POST /query

Example:
{
  "query": "What is my health status?"
}

Affordability
GET /affordability/{cost}

Example:
GET /affordability/500000

Insurance Coverage
GET /coverage/{cost}

Example:
GET /coverage/500000

Recommendations
GET /recommendations

Monitoring
GET /monitor

Example Queries
Health
What is my health status?

What is my cardiovascular risk?

What is my diabetes risk?

Finance
How is my financial situation?

Can I afford a Rs.5 lakh treatment?

Insurance
What is my insurance coverage?

Is my insurance adequate?

Cross-Domain
Can I afford a Rs.5 lakh surgery?

This type of query can combine:
Health
+
Finance
+
Insurance
+
RAG
+
Autonomous Simulation
+
AI Narration

Project Structure
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

Testing
Formula Tests
python -m pytest test_formulas.py

Full Test Suite
python -m pytest

Individual Module Tests
python health_twin.py
python finance_twin.py
python insurance_twin.py
python rag_agent.py
python orchestrator.py
python monitoring_agent.py
python recommendation_engines.py

Backend Smoke Test
Start the backend:
uvicorn main:app --reload

Then test:
python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/').read().decode())"

Expected:
{"message":"TwinLife AI API is running"}

Frontend Improvements
The current frontend includes several validation and visualization improvements.
Health Dashboard
Displays:
- Health score
- BMI
- BMI category
- Blood pressure
- BP category
- HbA1c
- Blood glucose
- Cardiovascular risk
- Diabetes risk
- Smoking
- Physical activity
- SHAP factors
Finance Dashboard
Displays:
- Financial stability score
- Monthly income
- Monthly expenses
- EMIs
- Disposable income
- DTI
- Savings rate
- Emergency fund
- Monthly financial breakdown
Insurance Dashboard
Displays:
- Insurance adequacy score
- Sum insured
- Annual premium
- Annual income
- Coverage ratio
- Premium ratio
- Rider gaps
Treatment Simulation
Displays:
- Treatment cost
- Insurance covered amount
- Coverage gap
- Self-funding status
- Affordability score
- Health risk
- Urgency
- Agent recommendation
RAG Transparency
Displays:
Guidelines Used
        ↓
Retrieved Knowledge
        ↓
Top 3 Relevant Guideline Chunks

Loading State
The interface displays:
Analyzing your digital twin...

Health
Finance
Insurance
RAG
AI Analysis

Error Handling
The frontend handles backend connection failures and displays a clear error message instead of silently failing.
Current Limitations
- User profile persistence is not currently implemented.
- The current frontend sends profile information, but the backend request model currently focuses on the query field; full per-request profile propagation should be completed before treating custom profiles as fully persistent backend state.
- The application is currently intended for local development/demo usage.
- Medical and financial outputs are advisory and should not replace qualified professional advice.
- RAG quality depends on the documents available in the local knowledge base.
- Model predictions depend on the training datasets and their limitations.
Future Enhancements
Potential future improvements include:
- Persistent user profiles
- Database-backed user management
- Authentication
- Historical assessment tracking
- More advanced visualization
- Deployment to cloud infrastructure
- More healthcare datasets
- More financial planning scenarios
- Additional insurance products
- Real-time monitoring notifications
- Mobile application
- Voice interaction
- External health/wearable integrations
Disclaimer
TwinLife AI is an educational and research-oriented project.
The health information, risk predictions, financial calculations, insurance analysis, and AI-generated recommendations provided by the system are intended for informational purposes only.
They should not be considered a substitute for advice from qualified medical, financial, insurance, or other professional advisors.