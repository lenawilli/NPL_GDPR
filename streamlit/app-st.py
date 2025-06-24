import streamlit as st
import json
import numpy as np
import joblib
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re

# ---------------------------
# LegalBERT-based compliance checker
# ---------------------------
class GDPRComplianceChecker:
    def __init__(self, model_name="nlpaueb/bert-base-uncased-eurlex"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def get_embeddings(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                output = self.model(**inputs)
                embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding[0])
        return np.array(embeddings)

    def chunk_policy_text(self, text, chunk_size=500):
        paragraphs = re.split(r'\n{2,}|\.\s+', text)
        chunks, current = [], ""
        for para in paragraphs:
            if len(current) + len(para) < chunk_size:
                current += " " + para
            else:
                chunks.append(current.strip())
                current = para
        if current:
            chunks.append(current.strip())
        return [chunk for chunk in chunks if len(chunk) > 50]

    def load_gdpr_articles(self, gdpr_json):
        gdpr_map, texts = {}, []
        for article in gdpr_json:
            number, title = article["article_number"], article["article_title"]
            body = " ".join([f"{k} {v}" for sec in article["sections"] for k, v in sec.items()])
            full_text = f"Article {number}: {title}. {body}"
            gdpr_map[number] = {"title": title, "text": full_text}
            texts.append(full_text)
        embeddings = self.get_embeddings(texts)
        return gdpr_map, embeddings

    def calculate_compliance_score(self, policy_text, gdpr_map, gdpr_embeddings):
        chunks = self.chunk_policy_text(policy_text)
        if not chunks:
            return {"error": "Policy has no meaningful chunks."}
        chunk_embeddings = self.get_embeddings(chunks)
        sim_matrix = cosine_similarity(gdpr_embeddings, chunk_embeddings)

        article_scores = {}
        presence_threshold = 0.35
        total_score, counted_articles = 0, 0

        for i, (art_num, art_data) in enumerate(gdpr_map.items()):
            max_sim = np.max(sim_matrix[i])
            best_idx = np.argmax(sim_matrix[i])

            if max_sim < presence_threshold:
                continue

            score_pct = min(100, max(0, (max_sim - presence_threshold) / (1 - presence_threshold) * 100))
            article_scores[art_num] = {
                "article_title": art_data["title"],
                "compliance_percentage": round(score_pct, 2),
                "similarity_score": round(max_sim, 4),
                "matched_text_snippet": chunks[best_idx][:300] + "..."
            }
            total_score += score_pct
            counted_articles += 1

        overall = round(total_score / counted_articles, 2) if counted_articles else 0
        return {
            "overall_compliance_percentage": overall,
            "relevant_articles_analyzed": counted_articles,
            "total_policy_chunks": len(chunks),
            "article_scores": article_scores
        }


def chunk_policy_text(text, chunk_size=500):
    import re
    paragraphs = re.split(r'\n{2,}|\.\s+', text)
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += " " + para
        else:
            chunks.append(current.strip())
            current = para
    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if len(chunk) > 50]


# ---------------------------
# Streamlit interface
# ---------------------------
st.set_page_config(page_title="GDPR Compliance Checker", layout="wide")
st.title("🛡️ GDPR Compliance Checker")

with st.sidebar:
    st.header("Upload Files")
    gdpr_file = st.file_uploader("GDPR JSON File", type=["json"])
    policy_file = st.file_uploader("Company Policy (.txt)", type=["txt"])

if gdpr_file and policy_file:
    model_choice = st.selectbox(
        "Choose the model to use:",
        ["Logistic Regression", "MultinomialNB", "LegalBERT (Eurlex)", "Knowledge Graphs"]
    )

    gdpr_data = json.load(gdpr_file)
    article_title_map = {f"Article {a['article_number']}": a['article_title'] for a in gdpr_data}

    policy_text = policy_file.read().decode("utf-8")

    with st.spinner("Analyzing..."):
        if model_choice == "LegalBERT (Eurlex)":
            checker = GDPRComplianceChecker()
            gdpr_map, gdpr_embeddings = checker.load_gdpr_articles(gdpr_data)
            result = checker.calculate_compliance_score(policy_text, gdpr_map, gdpr_embeddings)

        elif model_choice in ["Logistic Regression", "MultinomialNB"]:
            if model_choice == "Logistic Regression":
                model = joblib.load("logistic_regression_model.joblib")
                vectorizer = joblib.load("logistic_regression_vectorizer.joblib")
            else:
                model = joblib.load("multinomialNB_model.joblib")
                vectorizer = joblib.load("multinomialNB_vectorizer.joblib")

            chunks = chunk_policy_text(policy_text)
            chunks = [c.strip() for c in chunks if len(c.strip()) > 40]
            X_tfidf = vectorizer.transform(chunks)
            y_pred = model.predict(X_tfidf)
            y_proba = model.predict_proba(X_tfidf)

            article_scores = defaultdict(lambda: {
                "article_title": "",
                "compliance_percentage": 0.0,
                "similarity_score": 0.0,
                "matched_text_snippet": ""
            })
            total_score = 0
            counted_chunks = 0

            for i, (label, prob_vector) in enumerate(zip(y_pred, y_proba)):
                max_prob = max(prob_vector)
                if max_prob >= 0.35:
                    score_pct = min(100.0, max(0.0, (max_prob - 0.35) / (1 - 0.35) * 100))
                    if score_pct > article_scores[label]["compliance_percentage"]:
                        article_scores[label]["compliance_percentage"] = score_pct
                        article_scores[label]["similarity_score"] = round(max_prob, 4)
                        article_scores[label]["matched_text_snippet"] = chunks[i][:300] + "..."
                    article_scores[label]["article_title"] = article_title_map.get(label, label)
                    total_score += score_pct
                    counted_chunks += 1

            overall = round(total_score / counted_chunks, 2) if counted_chunks else 0
            result = {
                "overall_compliance_percentage": overall,
                "relevant_articles_analyzed": len(article_scores),
                "total_policy_chunks": len(chunks),
                "article_scores": dict(article_scores)
            }

        elif model_choice == "Knowledge Graphs":
            st.warning("Knowledge Graphs model is not implemented yet.")
            result = {}

        else:
            result = {}

    if result:
        st.subheader(f"✅ Overall Compliance Score: {result['overall_compliance_percentage']}%")
        st.markdown("---")
        st.subheader("📋 Detailed Article Breakdown")
        for art_num, data in sorted(result['article_scores'].items(), key=lambda x: -x[1]['compliance_percentage']):
            with st.expander(f"Article {art_num} - {data['article_title']} ({data['compliance_percentage']}%)"):
                st.write(f"**Similarity Score**: {data['similarity_score']}")
                st.write(f"**Matched Text**:\n\n{data['matched_text_snippet']}")
else:
    st.info("Please upload both a GDPR JSON file and a company policy text file to begin.")
