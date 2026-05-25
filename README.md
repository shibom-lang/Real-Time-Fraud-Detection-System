# 🔍 Fraud Detection System — IEEE-CIS

## Project Overview
End-to-end fraud detection pipeline with Explainable AI and a live Streamlit dashboard.
Built as part of the AI & Data Analytics Internship Capstone.

## Structure
```
FraudDetection/
├── analysis.ipynb          # Full Jupyter notebook (all 8 tasks)
├── dashboard/
│   └── app.py              # Streamlit multi-page dashboard
├── charts/                 # All generated charts (20 PNG files)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup
```bash
pip install -r requirements.txt
```

## Running the Notebook
```bash
jupyter notebook analysis.ipynb
```

## Running the Dashboard
```bash
# Ensure models.pkl and preprocessed.pkl are in the working directory
streamlit run dashboard/app.py
```

## Results Summary
| Model            | ROC-AUC | PR-AUC | F1    |
|------------------|---------|--------|-------|
| LightGBM         | 0.9146  | 0.6107 | 0.5814|
| XGBoost          | 0.9064  | 0.5842 | 0.5726|
| Isolation Forest | 0.7298  | 0.1166 | 0.1484|

**Best model:** LightGBM  
**Optimal threshold:** 0.399 (F1 = 0.5939)

## Dashboard (Streamlit Cloud)
Deploy via [Streamlit Community Cloud](https://streamlit.io/cloud) and submit the live URL.
# Real-Time-Fraud-Detection-System
