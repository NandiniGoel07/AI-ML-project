# AI-ML-project
# Intelligent Network Intrusion Detection System with Explainable AI

A machine learning system that classifies network traffic as **Normal**, **DoS**, **Probe**, **R2L**, or **U2R**, with SHAP-based explanations for every prediction. Built on the CICIDS2017 dataset and deployed as an interactive Streamlit web application.

## 🔗 Live Demo

https://ai-ml-project-cse2-igdtuw-08901012025-12301012025.streamlit.app/

## Problem Statement

Traditional intrusion detection systems rely on static, rule-based or signature-based methods that fail to detect novel and zero-day attacks, and offer no explanation for their decisions. This project applies supervised machine learning to network flow data and integrates Explainable AI (SHAP) — an aspect largely missing from prior NIDS research (see Literature Review below).

## Objective

- Classify network traffic into 5 categories using supervised ML.
- Compare multiple algorithms and identify the best-performing model.
- Explain every prediction using SHAP (SHapley Additive exPlanations).
- Deploy the complete system as a real-time, interactive web app.

## Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset
Source: https://www.unb.ca/cic/datasets/ids-2017.html

- Raw merged dataset: 2,830,743 rows × 79 columns (8 CSV files, Monday–Friday captures)
- After cleaning (duplicates, missing/infinite values removed): 2,520,798 rows × 78 columns
- Classes: Normal, DoS, Probe, R2L, U2R (mapped from 15 fine-grained CICIDS2017 labels)

## Data Preprocessing

- Merged 8 CSV files; stripped whitespace from column names
- Removed rows with NaN/Infinity values and duplicate rows
- Mapped fine-grained attack labels into 5 broad categories
- Encoded labels, applied an 80/20 stratified train-test split
- Scaled features using `StandardScaler` (fit on training data only)
- Balanced severe class imbalance using **SMOTE** (training set only)

## Models Implemented
Model                Decision Tree         Random Forest         XGBoost     
|
 Accuracy               99.86%                 99.81%             99.85%  
|
 Precision              99.87%                 99.82%             99.86%  
|
 Recall                 99.86%                 99.81%             99.85% 
| 
 F1-Score               99.86%                 99.81%             99.85%  

 Full comparison, confusion matrices, and per-class metrics available in `reports/`.

## Explainability

SHAP's `TreeExplainer` is used to generate:
- **Global** feature importance summary plots for each of the 5 classes
- **Local** per-prediction explanations (force plots + bar charts) showing exactly why the model made a specific prediction

## Streamlit App

- Upload your own traffic CSV, or try a pre-loaded sample row
- Choose between Random Forest, Decision Tree, and XGBoost
- View the prediction and a live SHAP explanation of the top contributing features
- View the full model comparison table/chart from the sidebar

## Project Structure
AI-ML-project/
│
├── data/
│ ├── raw/ # Original CICIDS2017 CSVs (not tracked in git)
│ └── processed/ # Cleaned/split/SMOTE-balanced data (not tracked in git)
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_model_training.ipynb
│ └── 04_shap_explainability.ipynb
│
├── models/ # Saved trained models, scaler, encoder, SHAP explainer
├── app/
│ ├── app.py # Streamlit application
│ └── sample_traffic.csv
├── reports/
│ ├── figures/ # Confusion matrices, SHAP plots, comparison charts
│ └── model_comparison.csv
│
├── requirements.txt
└── README.md

## ⚙️ How to Run Locally

```bash
git clone https://github.com/NandiniGoel07/AI-ML-project.git
cd AI-ML-project

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

streamlit run app/app.py
```

**Note:** raw CICIDS2017 CSVs and processed `.npy` files are not included in this repo (too large for git). Place raw CSVs in `data/raw/` and run the notebooks in order (`01` → `04`) to regenerate all intermediate files locally.

## Literature Review

A review of 15 recent research papers on network/host intrusion detection identified explainability as the most consistently missing element — 13 of 15 papers flagged the lack of Explainable AI (XAI) as an open research gap. This project's core contribution is directly addressing that gap by integrating SHAP throughout the pipeline.

## Team

- Nandini Goel — EDA, Preprocessing, SHAP Explainability
- Khushi — Model Training, Streamlit App & Deployment

## Tech Stack

Python · pandas · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · SHAP · Streamlit · Matplotlib/Seaborn
