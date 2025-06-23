# GDPR Compliance Scoring for Privacy Policies
Koblmiller Julia, Prakash Umakanth, Wadhwani Amar, Willisits Lena​

## Problem Statement

Many companies publish privacy policies—but these are often **not GDPR compliant**.  
Manual verification of such policies is:

- Time-consuming
- Error-prone
- Requires expert-level legal knowledge

This project addresses the need for **automated GDPR compliance checks**.

---

## Project Motivation

The goal is to develop an **NLP-based tool** that:
- Parses and analyzes privacy policy texts
- Compares them against the **General Data Protection Regulation (GDPR)** law
- Scores each policy based on its **coverage of GDPR articles**
- Identifies **compliance gaps** and missing clauses

This project contributes to regulatory technology and automated auditing solutions.

---

## Research Questions

- **Which GDPR clauses are most often missing or weakly covered** in company privacy policies?
- **What is the average compliance percentage** with GDPR law across Austrian companies?

---

## How to Run the App

### 1. Install Python dependencies
Make sure you have Python 3.8 or higher installed. Then install required packages:
```bash
pip install -r requirements.txt
```

### 2. Start the Streamlit app
From the terminal (in the app's folder):
```bash
streamlit run app-st.py
```

## Required Files

Place the following files in the app folder:
- `app-st.py` – the main Streamlit app
- `requirements.txt` – Python dependencies
- `logistic_regression_model.joblib`
- `logistic_regression_vectorizer.joblib`
- `multinomialNB_model.joblib`
- `multinomialNB_vectorizer.joblib`

## Supported Models

You can choose one of the following models in the sidebar:
- **Logistic Regression** (TF-IDF + classifier)
- **MultinomialNB** (Naive Bayes + TF-IDF)
- **LegalBERT (Eurlex)** (Transformer embeddings + cosine similarity)
- **Knowledge Graphs** (Not implemented yet)

## Input Files

**Required inputs in sidebar:**
- A `.json` file containing the full GDPR articles structure (e.g. `gdpr_articles_baseline.json` -> provided in the repository)
- A `.txt` file with your company's privacy policy

## Output

- Overall compliance score (0–100%)
- Breakdown of relevant articles matched
- Similarity score and sample matched text for each article

---


