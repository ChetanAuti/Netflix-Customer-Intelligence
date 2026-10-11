# Netflix Customer Intelligence, Churn Prediction & AI Analytics Copilot

An end-to-end customer analytics and AI platform built around a Netflix-style subscription dataset. The project combines data engineering, exploratory analysis, PostgreSQL business analytics, machine learning, customer segmentation, customer lifetime value (CLV), and an AI analytics copilot into one production-oriented workflow.

> **Dataset size:** 8,000 subscribers across customer, subscription, viewing, payment, support, feedback, content, and churn datasets.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Objectives](#business-objectives)
- [Key Capabilities](#key-capabilities)
- [End-to-End Architecture](#end-to-end-architecture)
- [Dataset](#dataset)
- [Data Pipeline](#data-pipeline)
- [Feature Engineering](#feature-engineering)
- [Machine Learning](#machine-learning)
- [Segmentation](#segmentation)
- [Customer Lifetime Value](#customer-lifetime-value)
- [SQL Business Analytics](#sql-business-analytics)
- [AI Analytics Copilot](#ai-analytics-copilot)
- [Streamlit Application](#streamlit-application)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [PostgreSQL Setup](#postgresql-setup)
- [AI / LLM Setup](#ai--llm-setup)
- [Running the Notebooks](#running-the-notebooks)
- [Testing](#testing)
- [Model Results](#model-results)
- [Important Modeling Limitations](#important-modeling-limitations)
- [Outputs](#outputs)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Project Overview

**Netflix Customer Intelligence, Churn Prediction & AI Analytics Copilot** is designed as a complete customer intelligence workflow rather than a standalone machine learning notebook.

The platform takes raw customer data through:

```text
Raw CSV Data
     ↓
Data Cleaning & Validation
     ↓
EDA & Business Exploration
     ↓
PostgreSQL + SQL Business Questions
     ↓
Customer-Level Feature Engineering
     ↓
Churn Prediction
     ↓
Customer Segmentation
     ↓
CLV Prediction
     ↓
AI Analytics Copilot
     ↓
Streamlit Customer Intelligence Dashboard
```

The project supports both **analytical decision-making** and **customer-level intelligence**.

---

## Business Objectives

The platform focuses on questions such as:

- Which customers are at higher churn risk?
- Which subscription plans generate the most revenue?
- Which acquisition channels have higher churn or retention?
- How engaged are customers with the platform?
- Which customers generate repeated support interactions?
- What are the most common customer complaints?
- Which customers have high predicted lifetime value?
- Which customer segments require different retention strategies?
- What are the main operational and payment-related issues?
- Can business users ask questions in natural language and receive evidence-backed answers?

---

## Key Capabilities

### Data & Analytics

- Data cleaning and validation
- Missing-value handling
- Duplicate detection
- Exploratory data analysis
- Customer-level aggregation
- PostgreSQL relational database
- 37 SQL business questions
- Revenue, churn, retention and support analytics

### Machine Learning

- Customer churn propensity modeling
- Logistic Regression
- Random Forest
- Gradient Boosting
- Customer segmentation using K-Means
- Customer Lifetime Value prediction
- Leakage-aware feature engineering
- Forward-looking leakage validation

### AI Analytics Copilot

- Natural-language to SQL
- SQL validation
- Query audit logging
- Evidence-based explanations
- Customer-level explanations
- Retrieval-augmented analysis (RAG)
- Root-cause analysis
- Retention recommendations
- Automated reports
- Offline deterministic fallback mode

### Application

A multi-page Streamlit application provides:

1. Customer Analytics
2. Churn Prediction
3. Customer Segmentation
4. AI Copilot
5. Support Intelligence
6. Automated Reports

---

## End-to-End Architecture

```text
                     ┌──────────────────────┐
                     │      Raw CSV Data     │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │  Cleaning Pipeline   │
                     │ clean.py             │
                     └──────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
          ┌─────────────────┐     ┌─────────────────┐
          │ EDA / Figures   │     │ PostgreSQL      │
          └─────────────────┘     │ + SQL Q1-Q37    │
                                  └────────┬────────┘
                                           │
                                           ▼
                                ┌────────────────────┐
                                │ Feature Engineering│
                                └─────────┬──────────┘
                                          │
                         ┌────────────────┼────────────────┐
                         ▼                ▼                ▼
                  ┌────────────┐   ┌────────────┐   ┌────────────┐
                  │ Churn      │   │ Segmentation│  │ CLV        │
                  │ Prediction │   │ K-Means     │  │ Regression │
                  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
                        │                 │                │
                        └─────────────────┼────────────────┘
                                          ▼
                                ┌────────────────────┐
                                │ AI Analytics Copilot│
                                │ SQL / RAG / RCA /  │
                                │ Retention / Reports│
                                └─────────┬──────────┘
                                          │
                                          ▼
                                ┌────────────────────┐
                                │ Streamlit Dashboard│
                                └────────────────────┘
```

---

## Dataset

The project uses 9 core CSV datasets:

| Dataset | Purpose |
|---|---|
| `customers.csv` | Customer profile and acquisition information |
| `subscriptions.csv` | Subscription history and status |
| `subscription_plans.csv` | Plan metadata and pricing |
| `viewing_activity.csv` | Viewing sessions and engagement |
| `content.csv` | Content metadata |
| `support_tickets.csv` | Customer support interactions |
| `customer_feedback.csv` | Customer ratings and sentiment |
| `churn_labels.csv` | Customer churn target |
| `payments.csv` | Payment transactions and failures |

The original raw data is preserved under `data/raw/`.

---

## Data Pipeline

The cleaning pipeline performs:

- Schema validation
- Data type normalization
- Missing-value analysis
- Duplicate detection
- Invalid-date handling
- Invalid transaction handling
- Expected-null validation
- PostgreSQL-ready exports
- Cleaning decision logging

Cleaning decisions are recorded in:

```text
reports/logs/cleaning_log.json
```

---

## Exploratory Data Analysis

The EDA stage examines:

- Customer demographics
- Acquisition channels
- Subscription plans
- Subscription status
- Auto-renew behavior
- Viewing engagement
- Device usage
- Payment success/failure
- Support ticket categories
- Resolution time
- Customer ratings
- Sentiment
- Overall churn
- Churn by customer and subscription characteristics

EDA figures are stored under:

```text
reports/figures/
```

---

## Feature Engineering

Customer-level features are constructed from multiple behavioral sources.

### Customer Features

- Age
- Gender
- Country
- City
- Acquisition channel
- Customer segment
- Preferred language
- Tenure

### Subscription Features

- Total subscriptions
- Current plan
- Subscription status indicators
- Subscription auto-renew behavior

### Viewing Features

- Total sessions
- Total watch minutes
- Average watch duration
- Average completion percentage
- Active viewing days
- Watch minutes per session
- Sessions per active day

### Payment Features

- Total payments
- Total spend
- Average payment amount
- Payment status counts
- Failed payment rate
- Failed-payment indicator

### Support Features

- Total support tickets
- Average resolution time
- Average support satisfaction
- Support intensity

### Feedback Features

- Total feedback records
- Average feedback rating
- Sentiment counts
- Low-feedback indicator

The project uses a fixed snapshot date and avoids direct post-churn information in the intended churn modeling workflow.

---

## Machine Learning

### Churn Prediction

The churn pipeline evaluates:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Optional XGBoost when available

Evaluation includes:

- ROC-AUC
- Precision
- Recall
- F1 score
- Confusion matrix
- Risk-tier analysis
- Out-of-fold validation

The trained model is stored as:

```text
models/churn_model.joblib
```

### Customer Risk Scoring

Customers can be transformed into risk-oriented outputs such as:

- Churn probability
- Risk tier
- Customer-level explanations
- Retention recommendations

---

## Segmentation

Customer segmentation is implemented using **K-Means clustering**.

The segmentation workflow groups customers using behavioral and value-oriented features and produces interpretable segments for business analysis.

The trained segmentation model is stored as:

```text
models/segmentation_model.joblib
```

Example segment interpretation includes groups such as high-value loyal customers and customers with billing/payment problems or weaker engagement.

---

## Customer Lifetime Value

The project also predicts customer lifetime value using a regression model.

The CLV workflow uses customer-level value and behavioral features and produces predicted CLV for downstream customer intelligence.

Model artifact:

```text
models/clv_model.joblib
```

---

## SQL Business Analytics

The PostgreSQL database contains the relational customer intelligence model and **37 business questions**.

The SQL analysis covers topics including:

- Active customers by country
- Revenue by plan
- Overall churn rate
- Acquisition channel performance
- Average customer age by plan
- Most-watched content
- Content completion
- Failed payments
- Support resolution time
- Customer satisfaction
- Device usage
- Feedback sentiment
- Registrations by channel
- Auto-renew behavior
- High-support customers
- Plan revenue ranking
- Latest support ticket per customer
- Moving averages
- Cohort retention
- Country churn analysis
- Running payment totals
- Engagement decline
- Monthly churn rate
- Retention by acquisition channel
- Customer lifetime spend quartiles
- Country-level complaint analysis
- Failed-payment risk candidates
- Subscription price-change analysis
- Country-level customer health
- At-risk revenue
- Complaint analysis for completed quarters
- Win-back candidates
- Plan migration
- WAU/MAU and stickiness
- Negative feedback after payment/support problems
- Customer health scoring

SQL source:

```text
database/sql_business_questions.sql
```

Database schema:

```text
database/schema.sql
```

---

## AI Analytics Copilot

The AI Copilot is the project's intelligence layer for business users.

### Text-to-SQL

Users can ask questions in natural language and the system maps them to SQL queries.

The SQL pipeline includes:

- Query generation
- SQL validation
- Allowlist-based safety checks
- Audit logging
- Business-question templates
- Deterministic offline fallback

### RAG

The RAG layer can retrieve relevant customer/project information before generating an answer.

### Explanations

The explanation layer converts analytical outputs into business-friendly summaries while keeping the response evidence-based.

### Root-Cause Analysis

RCA combines customer, payment, support, engagement and churn information to identify measurable drivers associated with customer risk.

### Retention Recommendations

The retention module maps customer signals to practical intervention suggestions.

### Automated Reports

The report generator creates structured customer/business reports from the available analytical evidence.

LLM functionality can use Claude through an API key, but the project also supports deterministic offline behavior for core functionality.

---

## Streamlit Application

Run the dashboard with:

```bash
streamlit run app/dashboard.py
```

The application contains six functional pages plus the main dashboard:

### 1. Customer Analytics

Explore customer profiles, behavior, revenue, engagement and customer-level metrics.

### 2. Churn Prediction

Inspect churn probabilities, risk tiers and customer-level churn insights.

### 3. Customer Segmentation

Explore K-Means customer segments and segment-level characteristics.

### 4. AI Copilot

Ask natural-language business questions and access SQL, explanations, RCA and retention insights.

### 5. Support Intelligence

Analyze support categories, complaints, resolution patterns and customer satisfaction.

### 6. Automated Reports

Generate structured analytical reports from the project outputs.

---

## Repository Structure

```text
customer-intelligence-platform/
│
├── app/
│   ├── dashboard.py
│   ├── ui.py
│   └── pages/
│       ├── 1_Customer_Analytics.py
│       ├── 2_Churn_Prediction.py
│       ├── 3_Customer_Segmentation.py
│       ├── 4_AI_Copilot.py
│       ├── 5_Support_Intelligence.py
│       └── 6_Automated_Reports.py
│
├── database/
│   ├── schema.sql
│   ├── load_data.py
│   └── sql_business_questions.sql
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── netflix.db
│
├── models/
│   ├── churn_model.joblib
│   ├── segmentation_model.joblib
│   └── clv_model.joblib
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_churn_model.ipynb
│   ├── 05_segmentation_and_clv.ipynb
│   └── 06_ai_copilot_demo.ipynb
│
├── src/
│   ├── data_processing/
│   ├── features/
│   ├── models/
│   ├── database/
│   ├── llm/
│   └── utils/
│
├── tests/
│
├── reports/
│   ├── figures/
│   ├── logs/
│   └── PROJECT_REPORT.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd customer-intelligence-platform
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

The project supports a local/offline workflow without requiring a PostgreSQL server or LLM API key.

```bash
pip install -r requirements.txt
python run_pipeline.py
streamlit run app/dashboard.py
```

Run the automated tests:

```bash
python -m pytest -q
```

The local pipeline creates/updates processed data, features, model outputs, a SQLite demo database, and analytical figures.

---

## PostgreSQL Setup

PostgreSQL is the intended production-style database setup.

### Create database

```bash
createdb netflix
```

### Configure connection

Linux/macOS:

```bash
export DATABASE_URL=postgresql://user:password@localhost:5432/netflix
```

Windows PowerShell:

```powershell
$env:DATABASE_URL="postgresql://user:password@localhost:5432/netflix"
```

### Run the pipeline

```bash
python run_pipeline.py
```

### Load PostgreSQL

```bash
python database/load_data.py --target postgres
```

### Run SQL business questions

```bash
psql netflix -f database/sql_business_questions.sql
```

### Start the application

```bash
streamlit run app/dashboard.py
```

---

## AI / LLM Setup

The AI features can use Claude through an Anthropic API key.

Linux/macOS:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

Windows PowerShell:

```powershell
$env:ANTHROPIC_API_KEY="your_api_key_here"
```

Optionally configure the model:

```bash
export LLM_MODEL=your_model_name
```

Without an API key, the project still provides deterministic/offline behavior for core AI Copilot functions, including built-in Text-to-SQL templates and evidence-based explanations/reports/RCA.

**Never commit API keys to GitHub.**

---

## Running the Notebooks

The notebooks follow the same analytical sequence as the production pipeline.

### 01 — Data Cleaning

```text
notebooks/01_data_cleaning.ipynb
```

Loads raw data, validates schemas, handles data quality issues and creates cleaned datasets.

### 02 — EDA

```text
notebooks/02_eda.ipynb
```

Performs customer, subscription, viewing, payment, support, feedback and churn analysis.

### 03 — Feature Engineering

```text
notebooks/03_feature_engineering.ipynb
```

Creates customer-level behavioral features for ML.

### 04 — Churn Model

```text
notebooks/04_churn_model.ipynb
```

Trains and evaluates churn models.

### 05 — Segmentation & CLV

```text
notebooks/05_segmentation_and_clv.ipynb
```

Builds customer segments and CLV predictions.

### 06 — AI Copilot Demo

```text
notebooks/06_ai_copilot_demo.ipynb
```

Demonstrates Text-to-SQL, RAG, explanations, RCA, retention and automated reports.

---

## Testing

The project includes automated tests covering:

- Data cleaning
- Feature/model sanity checks
- SQL validation
- AI Copilot behavior
- Supporting application logic

Run:

```bash
python -m pytest -q
```

The project currently contains **39 tests**.

---

## Model Results

### Churn Model

The selected Gradient Boosting churn classifier achieved:

| Metric | Result |
|---|---:|
| Held-out ROC-AUC | 0.617 |
| Out-of-fold ROC-AUC | 0.64 |
| High-risk tier churn | 38.8% |
| Low-risk tier churn | 18.9% |
| Top-risk decile lift | 1.46× base churn rate |

### Segmentation

The project uses **K-Means with K=5** for customer segmentation.

The resulting segments are designed for customer strategy, including value, loyalty, engagement and billing-related behavior.

### CLV Model

The Gradient Boosting CLV model achieved approximately:

| Metric | Result |
|---|---:|
| R² | 0.979 |
| MAE | $9.61 |

The project report notes that the strong CLV performance is largely driven by the relationship between tenure and price, so the result should be interpreted accordingly.

---

## Important Modeling Limitations

### Churn prediction is a propensity/risk-ranking model

The dataset's churn signal is relatively weak. The project therefore emphasizes **risk ranking and customer prioritization** rather than claiming a highly accurate future churn forecast.

A stricter forward-looking leakage experiment produced an AUC around **0.558**, which indicates that a very high AUC on the easier/static setup should not automatically be interpreted as strong real-world forecasting ability.

### Avoid data leakage

Features that directly reveal churn or occur after the churn event should not be used when claiming future churn prediction.

### CLV interpretation

The CLV model has very high fit because the synthetic/project dataset contains strong structural relationships such as tenure and plan price. It should therefore be treated as a project-level analytical model rather than a production financial forecast.

### PostgreSQL verification

The SQL scripts and loader are included, but production deployment should still execute Q1-Q37 against the target PostgreSQL instance and verify the resulting business outputs.

### LLM mode

Real LLM responses require an API key. Offline mode uses deterministic templates and evidence-based logic instead.

---

## Outputs

Important generated artifacts include:

```text
reports/
├── PROJECT_REPORT.md
├── model_metrics.json
├── figures/
└── logs/
    └── cleaning_log.json
```

Processed customer intelligence outputs include:

```text
data/processed/
├── cleaned datasets
├── customer_features.csv
└── customer_scores.csv
```

Model artifacts:

```text
models/
├── churn_model.joblib
├── segmentation_model.joblib
└── clv_model.joblib
```

---

## Future Improvements

Potential next steps for a production implementation include:

- Use real time-based churn labels for 30/60/90-day forecasting.
- Add time-aware train/validation/test splits.
- Monitor model drift and feature drift.
- Add model explainability with SHAP.
- Add automated model retraining pipelines.
- Add PostgreSQL connection pooling and production secrets management.
- Add authentication and role-based access to the dashboard.
- Add monitoring and structured application logging.
- Expand the Copilot with semantic SQL retrieval and richer metadata grounding.
- Add real-time customer health scoring.
- Integrate campaign/CRM actions for retention workflows.

---

## Author

**Chetan Ashok Auti**

Data Science / Data Analytics / Machine Learning

- GitHub: `ChetanAuti`
- LinkedIn: `chetan-auti-814662334`

---

## Project Summary

**Netflix Customer Intelligence, Churn Prediction & AI Analytics Copilot** brings together:

**Python + Pandas + PostgreSQL + SQL + Machine Learning + K-Means + CLV + RAG + Text-to-SQL + Streamlit + AI Analytics**

The goal is to demonstrate a complete analytics workflow from raw customer data to actionable customer intelligence through a single integrated platform.
