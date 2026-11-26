# Classroom Facility Issue Pattern Mining and Maintenance Prioritization Dashboard

**Using Text Classification and Association Rule Mining with an Interactive Decision Support Dashboard for Agusan del Sur State University (ADSSU)**

Capstone Project 1 prototype.

## Problem
Classroom facility problems (aircon, lights, outlets, projectors, furniture, leaks, and others) are reported often, but the reports are not analyzed. The school cannot easily see which problems repeat, which rooms need attention first, or which problems occur together.

## Intended users
ADSSU facilities and maintenance office, college deans, and administrators.

## Data mining techniques
- **Text classification** (TF-IDF + Logistic Regression): predicts the category of a report from its description.
- **Association rule mining** (Apriori): finds facility items that are reported together.

## Dashboard modules
1. Data Exploration
2. Text Classification (accuracy, precision, recall, F1)
3. Association Rules (support, confidence, lift)
4. Decision Support (room priority ranking, downloadable report)

## Dataset
`data/classroom_issues.csv` contains 400 reports with these columns:
`report_id, timestamp, campus_location, building_name, room_number, reporter_role, issue_description, category, urgency_level, resolution_status, days_to_resolve`

> **Note:** This is a **sample dataset for prototype purposes**. Part of it is simulated. It is not an official ADSSU record, and its results must not be presented as real findings. No personal names are stored. Replace it with real, authorized data in the same format.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Technologies
Python, Pandas, Scikit-learn, mlxtend, Plotly, Streamlit, GitHub, Streamlit Community Cloud.
