# 08. Master Build Prompt for TwinLife AI
### Paste this into Claude Code / Antigravity / Cursor to start implementation

---

```
PROJECT: TwinLife AI — Orchestrator-Driven Agentic Multi-Domain Digital Twin
for Personalized Health and Financial Well-Being

CONTEXT: This is a final-year academic capstone project. Build it incrementally,
module by module, so each piece is independently testable before moving to the
next. Prioritize clean, well-commented, defensible code over cleverness — I
need to explain every design decision in a viva.

=== ARCHITECTURE OVERVIEW ===
An Orchestrator coordinates three "Twins" (Health, Finance, Insurance) plus a
Monitoring Agent, a RAG Agent, a Simulation Engine, and a Narration layer.
Pipeline order: Twins compute raw scores (model/formulas) → SHAP explains the
model's prediction → Orchestrator decides which twin(s) a query needs → RAG
retrieves grounding guideline text → Claude API narrates the final, cited
explanation → Simulation Engine can re-run the whole pipeline on hypothetical
inputs for "what-if" comparisons.

=== TECH STACK (use exactly this, don't substitute) ===
- Health model: scikit-learn (RandomForestClassifier) + SHAP for explainability
- Formulas/rules: plain Python functions, no libraries needed
- Orchestrator: LangGraph
- LLM: Claude API (Anthropic SDK)
- RAG: LangChain + ChromaDB + sentence-transformers (all-MiniLM-L6-v2)
- Backend: FastAPI + PostgreSQL + SQLAlchemy
- Scheduler: APScheduler (for the proactive Monitoring Agent)
- Frontend: React + Vite + Tailwind + Recharts
- Simulation: NumPy (Monte Carlo for finance forecasting)
- Daily-use layer: Twilio WhatsApp API, Web Speech API (voice), Claude API
  vision (receipt/prescription photo scanning)

=== MODULE 1: FORMULAS (formulas.py) ===
Implement these as pure, independently unit-tested functions:
- bmi(height_cm, weight_kg) -> float, and who_obesity_class(bmi) -> str
- bp_stage(systolic, diastolic) -> str  (ACC/AHA staging: Normal/Elevated/
  Stage 1/Stage 2)
- framingham_cvd_risk(age, gender, cholesterol, systolic_bp, smoker) -> float
  (10-year risk %, use a simplified published Framingham-style formula)
- diabetes_indicator(glucose) -> str  (Low/Medium/High via threshold rule)
- dti_ratio(fixed_expenses, emis, income) -> float
- savings_rate(income, fixed_expenses, variable_expenses) -> float
- emergency_fund_months(savings_balance, fixed_expenses, variable_expenses)
  -> float
- financial_stability_score(dti, savings_rate, emergency_fund_months) -> int
  (0-100, weighted: 30% inverse DTI + 30% savings rate + 20% emergency fund
  + 20% debt-to-income)
- affordability_score(disposable_income, months_available, cost) -> float
- coverage_adequacy_ratio(sum_insured, estimated_future_cost) -> float
- premium_affordability_ratio(annual_premium, annual_income) -> float
- insurance_adequacy_score(coverage_ratio, premium_ratio, rider_gaps) -> int

=== MODULE 2: HEALTH MODEL (train_health_model.py) ===
- Load the Kaggle "Cardiovascular Disease" dataset (cardio_train.csv)
- Clean known outliers in height/weight/ap_hi/ap_lo (this dataset has
  documented data-entry errors — filter unrealistic values, e.g. diastolic
  > systolic, extreme height/weight z-scores)
- Derive BMI from height/weight
- Features: age, gender, bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco,
  active
- Target: cardio (0/1)
- Train RandomForestClassifier, 80/20 split
- Evaluate and print: accuracy, F1-score, ROC-AUC
- Run SHAP TreeExplainer on the trained model, save a feature-importance
  summary plot as shap_summary.png
- Save the trained model with joblib as model.pkl

=== MODULE 3: TWIN CLASSES ===
health_twin.py:
  class HealthTwin:
    def assess(self, inputs: dict) -> dict:
      # loads model.pkl, runs prediction + SHAP values, combines with
      # formulas.py functions (bmi, bp_stage, framingham, diabetes_indicator)
      # returns a health_profile dict matching this exact shape:
      # {"bmi":.., "obesity_class":.., "hypertension_stage":..,
      #  "cardio_risk":{"probability":.., "label":..},
      #  "diabetes_indicator":.., "overall_health_score":..,
      #  "shap_top_features": [...]}

finance_twin.py:
  class FinanceTwin:
    def assess(self, inputs: dict) -> dict:
      # uses formulas.py only, no model
      # returns finance_profile matching:
      # {"dti_ratio":.., "savings_rate":.., "emergency_fund_months":..,
      #  "financial_stability_score":..}
    def affordability_for(self, cost: float) -> float
      # used by the combined simulator

insurance_twin.py:
  class InsuranceTwin:
    def assess(self, inputs: dict, health_profile: dict,
               finance_profile: dict) -> dict:
      # rule-based: estimated_future_cost lookup keyed to health_profile's
      # risk labels, coverage_adequacy_ratio, premium_affordability_ratio,
      # rider_gaps, insurance_adequacy_score
      # returns insurance_profile matching the shape from the spec

=== MODULE 4: COMBINED SIMULATOR — AUTONOMOUS TOOL-USE AGENT
(combined_simulator.py) ===
Do NOT implement this as hardcoded if/else. Build it as an autonomous
agent using Claude API tool use (function calling):

1. Define these tools as plain deterministic Python functions (each wraps
   the twin/formula functions from Modules 1 and 3 — do not let the LLM
   compute these numbers itself):
   - check_affordability(cost: float) -> {"affordable": bool, "score": float}
     (wraps finance_twin.affordability_for)
   - check_insurance_coverage(cost: float) -> {"covered_amount": float,
     "gap": float} (wraps insurance_twin.coverage_for)
   - get_cheaper_alternative(condition: str, current_cost: float) ->
     {"new_cost": float, "tier": str} (wraps health_twin's cost lookup
     table, next tier down)
   - get_health_context(condition: str) -> {"risk_level": str,
     "urgency": str} (wraps health_twin.assess output)

2. Register these as tools in a Claude API tool-use call. Use this system
   prompt (goal + tools, NOT the exact steps — let the agent decide the
   order and combination):
   "You are a treatment-plan advisor agent evaluating a ₹{cost} treatment
   plan. Your objective: determine the most affordable, appropriate way
   for the user to proceed, minimizing financial strain while addressing
   their health need. You have the tools listed above. Reason step by
   step, call tools as needed, in whatever order you judge necessary, and
   stop once you reach a clear recommendation. Prefer self-funding if
   genuinely affordable; otherwise explore insurance coverage and
   lower-cost alternatives before concluding the plan is currently out of
   reach."

3. Run the standard tool-use loop: send the prompt -> if Claude requests a
   tool call, execute the corresponding Python function and return the
   result -> repeat until Claude returns a final text recommendation
   (no more tool calls requested).

4. Log every tool call and its result in order — this trace is your
   explainability evidence for this agent and should be stored alongside
   the final recommendation (e.g., in the chat_history table).

5. Test this agent function across at least 10-15 varied sample cases
   (different costs, coverage levels, affordability levels) and record
   the tool-call sequence for each, to confirm it behaves sensibly before
   the demo.

Also implement individual what-if simulators:
- health_twin.simulate(modified_inputs) -> before/after health_profile pair
- finance_twin.monte_carlo_forecast(months=12, iterations=1000,
  modified_inputs) -> probability distribution of key metrics using NumPy
- insurance_twin.simulate(modified_inputs) -> before/after insurance_profile

=== MODULE 5: RAG PIPELINE (rag_agent.py) ===
- Chunk 3-5 provided guideline PDFs (~500 tokens/chunk) using LangChain's
  RecursiveCharacterTextSplitter
- Embed with sentence-transformers/all-MiniLM-L6-v2
- Store in ChromaDB (persistent local collection)
- Implement: retrieve(topic: str, top_k=3) -> list[str] of relevant chunks

=== MODULE 6: ORCHESTRATOR (orchestrator.py, LangGraph) ===
- Define nodes: health_node, finance_node, insurance_node, rag_node,
  narration_node
- Implement route_query(query: str) -> dict with this two-step logic:
  STEP 1 (rule-based, no API call): keyword match against
  ["diabetes","weight","blood pressure"] -> health only;
  ["save","afford","loan","emi"] -> finance only;
  ["coverage","premium","claim"] -> insurance only;
  ["treatment","medication","surgery"] + any cost/money word -> all three,
  triggers combined_simulator
  STEP 2 (LLM fallback for anything ambiguous): call Gemini API with a
  system prompt instructing it to respond ONLY with JSON:
  {"twins": [...], "mode": "single-domain"|"cross-domain", "simulate": bool}
- Wire the graph so only the twins flagged by routing actually execute

=== MODULE 7: NARRATION LAYER (narration_agent.py) ===
Implement generate_explanation(twin_outputs: dict, shap_features: list,
rag_chunks: list) -> str that calls Claude API with a prompt combining:
  1. The computed score(s)
  2. SHAP's top contributing features (if health-related)
  3. The retrieved RAG guideline chunks
Instruct Claude to write a short, warm, plain-language explanation citing
the retrieved source, and to NOT invent any numeric claims not provided
in the input.

Also implement generate_ranked_whatifs(scenarios: list[dict]) -> str that
ranks 3-4 candidate scenarios using:
  score = risk_reduction*0.5 + (1/cost)*0.3 + insurance_fit*0.2
and narrates them ranked with trade-offs.

=== MODULE 8: MONITORING AGENT (monitoring_agent.py, APScheduler) ===
- A scheduled job (weekly) that: loads each user's latest twin inputs,
  recomputes all three profiles, compares to the last saved snapshot in a
  `snapshots` table, and if any score crosses a defined threshold (health
  score drop >10, DTI rise above 0.40, new insurance rider gap), calls
  narration_agent to generate an alert and stores it in a `notifications`
  table.

=== MODULE 9: FINANCE/HEALTH RECOMMENDATION ENGINES ===
- investment_recommendation(finance_profile, age) -> str : rule-based tier
  logic (build emergency fund first if <3 months; growth allocation if
  stability_score>=70 and age<40; balanced if >=70 and age>=40; conservative
  if 40-70; no new investment if <40). Always include the disclaimer
  "illustrative guidance, not licensed financial advice."
- purchase_impact_simulator(finance_profile, purchase_cost) -> dict with
  ranked alternatives (defer, cheaper tier, EMI split, reallocate) if the
  purchase would breach a healthy DTI/savings threshold
- lifestyle_impact_simulator(health_profile, habit_change) -> dict with
  ranked alternatives (substitute, moderate, offset) if the habit change
  would worsen risk band

=== MODULE 10: BACKEND API (FastAPI) ===
Endpoints: POST /onboard, POST /assess, POST /query (routes through
Orchestrator), POST /simulate, POST /whatif, GET /history,
GET /notifications
PostgreSQL tables: users, health_inputs, finance_inputs, insurance_inputs,
snapshots, notifications, chat_history

=== MODULE 11: FRONTEND (React + Tailwind + Recharts) ===
- Onboarding forms for Health/Finance/Insurance inputs
- Dashboard: radar chart (3 scores), forecast timeline chart, recommendation
  cards, notification feed
- Chat interface calling /query, with a mic button (Web Speech API) and a
  photo-upload button (sends image to a backend endpoint that forwards to
  Claude API vision for receipt/prescription extraction)
- What-if slider panel calling /simulate and /whatif, with before/after
  comparison view

=== BUILD ORDER ===
Build and test Module 1 first (pure functions, no dependencies), then 2, then
3, verifying each module works standalone with hardcoded sample inputs before
wiring modules together. Do not skip ahead to the Orchestrator/frontend until
Modules 1-4 are individually tested and passing.

=== CONSTRAINTS ===
- Only ONE trained ML model in the entire project (Module 2). Everything
  else (diabetes indicator, hypertension stage, obesity class, all Finance
  and Insurance scores) must be formulas/rules, not additional models.
- The combined_simulator IS autonomous (an LLM tool-use agent decides the
  order/branching) — but every underlying NUMBER it relies on (affordability,
  coverage, cost) must come from a deterministic tool call to formulas.py/
  twin functions, never estimated or invented by the LLM itself.
- Every recommendation involving money or health must include an
  appropriate disclaimer.
- Keep all functions independently unit-testable with clear docstrings,
  since I need to explain and defend each one individually.

Start with Module 1. Show me the code for formulas.py first, with basic
test cases for each function, before moving to Module 2.
```

---
*This prompt assumes the coding agent has access to your project repo and can install packages. Paste it as your first message to Claude Code (or the equivalent agent in Antigravity/Cursor), then work through the modules in order — reviewing and testing each before telling the agent to proceed to the next.*
