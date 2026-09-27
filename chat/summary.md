# TwinLife AI — Conversation & Development Summary

This document captures a summary of the development sessions, discussions, and decisions made while completing Modules 5 through 9 for the TwinLife AI project.

## 1. Context & Starting Point
Before this session, Modules 1 through 4 were already implemented and tested:
- **Module 1**: Deterministic domain formulas.
- **Module 2**: Health ML Models (Cardio & Diabetes).
- **Module 3**: Digital Twin Classes (Health, Finance, Insurance).
- **Module 4**: Autonomous Combined Simulator Agent (using Gemini API).

The objective of this session was to implement the remaining modules (5-9) sequentially, update the project documentation, and resolve any environmental or dependency issues along the way.

## 2. Module implementations

### Module 5: RAG Pipeline (`rag_agent.py`)
- We implemented a Retrieval-Augmented Generation (RAG) pipeline to pull context from 5 medical and financial guideline documents.
- **Technologies Used**: LangChain (`RecursiveCharacterTextSplitter`), HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`), and ChromaDB for persistent vector storage.
- **Issue Resolved**: Migrated from the deprecated `langchain_community.embeddings` to `langchain_huggingface`. Added the required `langchain-text-splitters` package when testing threw a `ModuleNotFoundError`.

### Module 6: Orchestrator (`orchestrator.py`)
- Built a LangGraph-based state machine to route user queries efficiently.
- **Routing Logic**: Implemented a two-step routing process.
  1. **Deterministic Keyword Match**: First checks against domain-specific keywords.
  2. **LLM Fallback**: If ambiguous, uses the Gemini API to determine routing.
- This ensures that only the necessary "Twins" are executed based on the user's input.

### Module 7: Narration Agent (`narration_agent.py`)
- Designed to generate warm, natural-language explanations for users.
- It combines:
  1. The user's computed scores.
  2. SHAP feature importances (e.g., explaining that blood pressure is driving their risk).
  3. Citations from the RAG guidelines.
- Also generates and ranks "what-if" wellness scenarios (e.g., quitting smoking vs. weight loss).
- **Issue Resolved**: The initially planned model (`gemini-2.0-flash-lite`) was deprecated. We updated the API calls to use `gemini-3.5-flash-lite`. We also experienced some HTTP 429 rate limits due to the free tier, which we handled gracefully.

### Module 8: Monitoring Agent (`monitoring_agent.py`)
- Added proactive monitoring using `apscheduler` to run periodic (e.g., weekly) checks on the user's profile.
- Saves snapshots to `data/snapshots/` and logs alerts to `data/notifications/`.
- Configured thresholds for alerts (e.g., DTI rising above 0.40, health score dropping by 10 points).

### Module 9: Recommendation Engines (`recommendation_engines.py`)
- Implemented three distinct, rule-based (non-ML) recommendation engines:
  1. **Investment Recommendation**: Age and stability-based allocation tiers.
  2. **Purchase Impact Simulator**: DTI/savings impact analysis offering ranked alternatives (e.g., deferring a purchase vs. taking an EMI).
  3. **Lifestyle Impact Simulator**: Projects health score improvements based on clinical guidelines (e.g., quitting smoking).
- **Issue Resolved**: Fixed a `UnicodeEncodeError` when printing box-drawing characters on Windows by instructing the terminal to use UTF-8 (`$env:PYTHONIOENCODING="utf-8"`).

## 3. Documentation Updates
We extensively updated the core project files to reflect the completed modules:
- **`requirements.txt`**: Uncommented and added the dependencies for the new modules (`langchain`, `chromadb`, `sentence-transformers`, `langgraph`, `apscheduler`).
- **`README.md`**: Restructured with a comprehensive architecture diagram, module breakdown, RAG sources table, and orchestrator routing table.
- **`instruction.txt`**: Added detailed execution and testing commands (Steps 4.5 through 4.9) and a troubleshooting section based on the issues we encountered.

## 4. Git & Final Testing
- Validated all tests individually inside the `twin_life` conda environment.
- Formatted a detailed git commit message summarizing the `feat: implement Modules 5-9` changes.
- Addressed questions regarding how the Machine Learning models (Module 2) are utilized deep inside the simulations and Twin generation processes.
