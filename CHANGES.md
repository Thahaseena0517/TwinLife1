# TwinLife AI — Changes & Development History

This document records the major changes made to the original TwinLife AI repository to reach the current integrated version.

---

## 1. Backend Integration

### Added Unified TwinLife Service

Created:

```text
src/services/twinlife_service.py



The service acts as the main backend integration layer.
It connects:
- Health Twin
- Finance Twin
- Insurance Twin
- LangGraph Orchestrator
- RAG Agent
- Treatment Simulation
- Narration
- Monitoring
- Recommendations
The service exposes a unified interface for the frontend.
2. Health Twin Integration
The existing Health Twin was integrated into the unified backend flow.
Health inputs include:
- Age
- Gender
- Height
- Weight
- Systolic Blood Pressure
- Diastolic Blood Pressure
- HbA1c
- Blood Glucose
- Cholesterol
- Glucose Level
- Smoking
- Alcohol Consumption
- Physical Activity
- Hypertension
- Heart Disease
The Health Twin calculates:
- BMI
- BMI category
- Blood pressure category
- Cardiovascular risk
- Diabetes risk
- Overall health score
- SHAP-based contributing factors
3. Finance Twin Integration
Finance Twin was connected to the unified service.
Financial inputs include:
- Monthly income
- Fixed expenses
- Variable expenses
- EMIs
- Savings balance
The Finance Twin calculates:
- Debt-to-income ratio
- Savings rate
- Emergency fund
- Financial stability score
- Treatment affordability
4. Insurance Twin Integration
Insurance Twin was integrated with the Health and Finance Twins.
Insurance inputs include:
- Sum insured
- Annual premium
- Annual income
- Existing riders
The Insurance Twin calculates:
- Coverage adequacy
- Premium affordability
- Insurance adequacy score
- Rider gaps
- Treatment coverage
5. Twin Assessment Initialization
The unified service was updated to calculate the three twin assessments during initialization.
The flow is:
Health Inputs
      ↓
Health Twin Assessment

Finance Inputs
      ↓
Finance Twin Assessment

Insurance Inputs
      ↓
Insurance Twin Assessment
      ↑
Health Profile + Finance Profile

This ensures that downstream orchestration has access to the calculated twin profiles.
6. LangGraph Orchestration
The LangGraph orchestrator was integrated into the backend.
The orchestration pipeline is:
User Query
    ↓
Intent Detection
    ↓
Health Twin
    ↓
Finance Twin
    ↓
Insurance Twin
    ↓
RAG
    ↓
Treatment Simulation
    ↓
AI Narration
    ↓
Final Response

The orchestrator supports deterministic keyword routing and Gemini-based fallback routing.
7. Treatment Cost Extraction Fix
The orchestrator previously had an issue parsing treatment costs expressed using Indian currency formats.
Examples:
Rs.5 lakh
₹5 lakh
5 lakh

The extraction logic was updated so that:
5 lakh → 500000

and similarly handles crore-based amounts.
The lakh/crore patterns are evaluated before generic currency-number patterns.
8. Autonomous Treatment Simulation
The treatment simulation agent was integrated using Gemini tool calling.
Available tools include:
check_affordability
check_insurance_coverage
get_cheaper_alternative
get_health_context

The agent can determine:
1. Treatment cost
2. Whether the treatment is affordable
3. Insurance coverage
4. Coverage gap
5. Health context
6. Alternative/recommendation
Tool execution is recorded in:
simulation.tool_trace

This trace is exposed to the frontend for transparency.
9. Narration Improvements
The narration agent was updated to provide more accurate treatment explanations.
Previously, there was ambiguity between:
- Overall insurance adequacy
- Specific treatment coverage
The narration logic was updated to use the treatment simulation values directly.
For a treatment simulation, the explanation now uses:
Treatment Cost
Covered Amount
Coverage Gap

instead of incorrectly using the overall insurance coverage ratio.
The narration prompt was also updated to explicitly distinguish these values.
10. RAG Integration
The RAG pipeline was integrated using:
- ChromaDB
- Sentence Transformers
- all-MiniLM-L6-v2
The system retrieves the most relevant guideline chunks for the user's query.
Current knowledge sources include:
cardiovascular_guidelines.txt
diabetes_management_guidelines.txt
financial_planning_guidelines.txt
insurance_planning_guidelines.txt
general_wellness_guidelines.txt

The system retrieves the top relevant chunks and passes them to the downstream AI explanation.
11. RAG Transparency in Frontend
The frontend was updated to display:
Guidelines Used
Shows the guideline categories identified from retrieved knowledge.
Retrieved Knowledge
Displays the top 3 retrieved guideline chunks.
This makes the RAG process more transparent to the user.
12. FastAPI Backend
FastAPI was added as the backend API layer.
Installed:
pip install fastapi uvicorn

The backend exposes:
GET  /
POST /query
GET  /affordability/{cost}
GET  /coverage/{cost}
GET  /recommendations
GET  /monitor

The backend runs using:
uvicorn main:app --reload

13. CORS Configuration
FastAPI CORS middleware was added to allow communication between:
React Frontend
        ↓
FastAPI Backend

The current frontend development origin is:
http://localhost:5173

14. React + Vite Frontend
A new frontend was created using:
- React
- Vite
- JavaScript
- CSS
Created using:
npm create vite@latest frontend -- --template react

The frontend communicates with the FastAPI backend.
15. User Profile Forms
The frontend was expanded from a simple query interface into a profile-driven dashboard.
Health Profile
Users can enter:
- Age
- Gender
- Height
- Weight
- Blood Pressure
- HbA1c
- Blood Glucose
- Cholesterol
- Glucose Level
- Alcohol Consumption
- Hypertension
- Heart Disease
- Smoking
- Physical Activity
Finance Profile
Users can enter:
- Monthly income
- Fixed expenses
- Variable expenses
- EMIs
- Savings balance
Insurance Profile
Users can enter:
- Sum insured
- Annual premium
- Annual income
- Existing riders
16. Frontend Input Validation
Validation was added before sending a request to the backend.
Health Validation
Examples:
Age: 18–100
Height: 50–250 cm
Weight: 20–300 kg
Systolic BP: 70–250
Diastolic BP: 40–150
HbA1c: 3–20
Blood glucose: 40–600

Finance Validation
Checks include:
- Income must be greater than zero
- Expenses cannot be negative
- EMIs cannot be negative
- Savings cannot be negative
- Total monthly expenses cannot exceed income
Insurance Validation
Checks include:
- Sum insured must be greater than zero
- Premium cannot be negative
- Annual income must be greater than zero
- Premium cannot exceed annual income
Invalid data is rejected before the API request is sent.
17. Health Dashboard Improvements
The Health dashboard was expanded to display:
- Overall health score
- BMI
- BMI category
- Blood pressure
- Blood pressure category
- HbA1c
- Blood glucose
- Cardiovascular risk
- Diabetes risk
- Smoking status
- Physical activity
The dashboard also displays the top SHAP contributing factors.
18. SHAP Visualization
Top SHAP features are displayed as horizontal bars.
Example:
HbA1c          █████████████
Blood Glucose  ███████████
Age            ████████
Systolic BP    ███████

This provides a visual explanation of which features contributed most to the ML prediction.
19. Finance Dashboard Improvements
The Finance dashboard was expanded to display:
- Financial stability score
- Monthly income
- Monthly expenses
- EMIs
- Disposable income
- DTI ratio
- Savings rate
- Emergency fund
A monthly financial breakdown was also added.
The disposable income calculation is:
Disposable Income =
Income - Fixed Expenses - Variable Expenses - EMIs

20. Insurance Dashboard Improvements
The Insurance dashboard was expanded to display:
- Insurance adequacy score
- Sum insured
- Annual premium
- Annual income
- Rider gaps
- Coverage ratio
- Premium affordability ratio
Identified rider gaps are displayed separately.
21. Treatment Simulation Dashboard
A treatment simulation summary card was added.
It displays:
Treatment Cost
Insurance Covered
Coverage Gap
Self-Funding
Final Patient Cost

The detailed simulation section also displays:
Financial Affordability
Insurance Coverage
Health Context
Agent Recommendation

22. Loading State
The frontend now displays a loading state while TwinLife processes a request.
The UI communicates the processing stages:
Health
Finance
Insurance
RAG
AI Analysis

The Ask TwinLife button is disabled while processing.
23. Error Handling
Frontend error handling was added for backend connection failures.
The UI can display a user-friendly message when the backend cannot be reached.
24. Backend Testing
The unified backend service was tested after integration.
The service smoke test verified:
Query processing
Affordability
Insurance coverage
Monitoring

The integration test completed successfully.
25. Parsing and Integration Fixes
Several integration issues were resolved during development:
Orchestrator Constructor
Updated service initialization to use the actual Orchestrator constructor.
Twin Assessment
Health, Finance and Insurance assessments are now performed before orchestration.
Monitoring Constructor
Monitoring initialization was corrected to match the actual MonitoringAgent interface.
Treatment Cost Parsing
Indian currency expressions such as lakh and crore are now parsed correctly.
Narration Treatment Coverage
Specific treatment coverage is now separated from overall insurance adequacy.
26. Current End-to-End Flow
The current system follows:
                    ┌──────────────────┐
                    │   React + Vite   │
                    │    Frontend      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     FastAPI      │
                    │     Backend      │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌───────────────────────┐
                  │   TwinLife Service    │
                  └───────────┬───────────┘
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
        Health Twin      Finance Twin    Insurance Twin
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                     ┌────────────────┐
                     │   Orchestrator │
                     │   LangGraph    │
                     └───────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
             RAG       Simulation       Narration
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                     Final AI Response

27. Validation Results
The integrated system was manually tested with user-entered profile values.
Example test results included:
Health Score: 80/100
Finance Score: 64/100
Insurance Score: 83/100

A treatment simulation for:
Rs. 5,00,000

returned:
Treatment Cost: Rs. 5,00,000
Insurance Covered: Rs. 5,00,000
Coverage Gap: Rs. 0

The system also correctly identified that paying the full amount personally could be unaffordable despite insurance covering the treatment cost.
28. Current Limitations
Profile Persistence
The current frontend collects user profile values during the session, but persistent user profile storage has not yet been implemented.
Backend Profile API
The current /query API primarily accepts the query field. Full structured profile transmission from the frontend to the backend should be completed as a future integration step.
Production Deployment
The current system is configured primarily for local development.
Production deployment, authentication, database persistence and cloud infrastructure can be added later.
29. Future Improvements
Potential future improvements include:
- Persistent user profiles
- User authentication
- Database integration
- Structured profile API
- Cloud deployment
- Better SHAP visualization
- More medical and financial knowledge sources
- More treatment simulation scenarios
- Advanced monitoring
- User history
- Personalized recommendations
- Production-grade security
30. Final Integrated Stack
Frontend
React
Vite
HTML
CSS
JavaScript

Backend
Python
FastAPI
Uvicorn

AI / ML
Scikit-learn
Random Forest
SHAP
Google Gemini API
LangGraph
Agentic AI
Function Calling

RAG
ChromaDB
Sentence Transformers
all-MiniLM-L6-v2

Monitoring
APScheduler

Summary
The original TwinLife AI modules were progressively integrated into a unified end-to-end application.
The current version combines:
Digital Twins
+
Machine Learning
+
RAG
+
LangGraph
+
Agentic AI
+
Treatment Simulation
+
AI Narration
+
FastAPI
+
React

The result is a unified TwinLife AI platform capable of analyzing health, finance and insurance information and using those results together for personalized scenario analysis.