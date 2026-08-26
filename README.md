# Loan Performance Intelligence Engine

Production-grade Machine Learning and Intelligence Engine for loan performance forecasting, risk analytics, servicer anomaly detection, macroeconomic stress testing, explainability, and generative RAG-driven advisory for structured finance and loan servicing portfolios.

---

##  Project Overview

The **Loan Performance Intelligence Engine** is an end-to-end analytical and predictive system designed to monitor loan portfolios, predict multi-horizon delinquency, default, and prepayment risks, model credit state transitions, detect data/servicer anomalies, simulate macroeconomic stress scenarios, and provide explainable AI insights alongside interactive generative RAG assistance.

### Key Capabilities
- **Multi-Horizon Risk Prediction**: Binary and multiclass classification for 3-month/6-month delinquency, 12-month default, 12-month prepayment, and next-state transitions.
- **Survival Analysis**: Time-to-event modeling (Kaplan-Meier and Cox Proportional Hazards) for default and prepayment trajectories.
- **Anomaly Detection & Servicer Reconciliation**: Hybrid anomaly scoring combining unsupervised Isolation Forests with deterministic business/servicer validation rules.
- **Macro Stress Testing & Scenario Engine**: Deterministic and stochastic scenario modeling (Baseline, Mild Recession, Severe Stagflation, High Interest Rate Shock).
- **Explainable AI (XAI)**: Global and local feature attributions using TreeSHAP, summary plots, waterfall charts, and risk driver extractions.
- **RAG-Powered Loan Intelligence Agent**: Context-aware retrieval-augmented generation for loan-level inquiries, compliance checks, and automated servicer exception reports (with built-in offline rule-based fallback).
- **MLflow Tracking**: Experiment tracking, parameter logging, metric recording, and model registry support.

---

##  Directory Structure

```text
Intain/
 data/                               # Data files and datasets
    loan_monthly_performance_train.csv
    loan_monthly_performance_test.csv
    loan_static_attributes.csv
    servicer_updates.csv
    validation_rules.json
    macro_scenarios.csv
    submission_template.csv
    data_dictionary.md
 models/                             # Trained model artifacts, encoders, and FAISS indices
    xgb_next_3m_delinquency.joblib
    xgb_next_12m_default.joblib
    isolation_forest.joblib
    faiss_index.bin
 notebooks/                          # Interactive Jupyter Notebooks
    pipeline.ipynb                  # Complete end-to-end pipeline demonstration
 reports/                            # Generated reports, audit logs, and figures
    data_quality_report.md
    explainability_report.md
    scenario_report.md
    demo_script.md
    llm_logs.json
 src/                                # Modular production Python source code
    __init__.py
    config.py                       # Single source of truth for config, paths, & seeds
    data_generator.py               # Synthetic data fallback generator
    data_loader.py                  # Ingestion, validation, and reconciliation
    feature_engineering.py          # Temporal aggregations, rolling stats, & encodings
    models.py                       # Multi-task ML trainers & predictors
    survival.py                     # Survival analysis (Kaplan-Meier & Cox PH)
    anomaly.py                      # Rule-based & Isolation Forest anomaly engine
    explainability.py               # SHAP feature importance & driver extraction
    scenario.py                     # Macroeconomic stress testing & scenario simulations
    rag_agent.py                    # RAG retrieval & LLM/offline advisory agent
    evaluate.py                     # Evaluation metrics (ROC-AUC, PR-AUC, Brier, F1)
    pipeline.py                     # Master orchestration pipeline entry point
 mlruns/                             # MLflow tracking artifacts
 requirements.txt                    # Python package dependencies
 submission.csv                      # Final formatted submission output
 README.md                           # Project documentation
```

---

##  Installation & Setup

### Prerequisites
- Python 3.10 to 3.12 recommended.
- A virtual environment is recommended to manage dependencies cleanly.

### 1. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Execution

### Option A: Run via Command-Line Pipeline
To run the full end-to-end pipeline (data validation, feature engineering, model training, survival analysis, anomaly detection, SHAP explainability, stress testing, RAG agent initialization, and report generation):

```bash
python src/pipeline.py
```

### Option B: Run via Jupyter Notebook
You can also run and inspect the entire pipeline step-by-step using the provided notebook:

```bash
jupyter notebook notebooks/pipeline.ipynb
```

---

##  Data Ingestion & Synthetic Data Fallback

- The engine looks for loan performance datasets in the `data/` folder as configured in [`src/config.py`](file:///c:/Users/jetty/OneDrive/Desktop/Intain/src/config.py).
- **Synthetic Data Fallback**: If raw training or static datasets (`loan_monthly_performance_train.csv`, `loan_static_attributes.csv`, etc.) are not present in `data/`, the pipeline automatically generates realistic, statistically correlated synthetic mortgage panel data and validation rules so that all modules, training workflows, and report generators execute seamlessly without manual intervention.

---

##  LLM & RAG Advisory Agent

- **Optional API Key**: The retrieval-augmented intelligence assistant supports OpenAI models if an `OPENAI_API_KEY` environment variable is present:
  ```bash
  # Optional: set OpenAI API Key for live LLM responses
  set OPENAI_API_KEY=your_api_key_here          # Windows CMD
  $env:OPENAI_API_KEY="your_api_key_here"      # Windows PowerShell
  export OPENAI_API_KEY="your_api_key_here"    # Linux/macOS
  ```
- **Offline Fallback**: Setting `OPENAI_API_KEY` is **strictly optional**. If no API key is provided, the agent automatically falls back to an offline rule-based and template-driven semantic synthesis engine, ensuring 100% deterministic, offline execution with zero runtime errors.

---

##  Reproducibility & Central Configuration

All global parameters, random seeds (`config.RANDOM_SEED = 42`), canonical column names, target definitions, and model hyperparameters are centrally defined in [`src/config.py`](file:///c:/Users/jetty/OneDrive/Desktop/Intain/src/config.py). All components import from this single source of truth to prevent configuration drift.
