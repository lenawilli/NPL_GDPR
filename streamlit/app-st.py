import streamlit as st
import json
import numpy as np
import joblib
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, XSD
import os
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

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

def prepare_article_text(article: Dict[str, Any]) -> str:
    body = " ".join(
        " ".join(sec.values()) if isinstance(sec, dict) else str(sec)
        for sec in article.get("sections", [])
    )
    return f"Art. {article['article_number']} – {article['article_title']} {body}"

def get_embedding(text: str) -> List[float]:
    # If input is a list of strings, clean each string
    if isinstance(text, list):
        cleaned_text = [t.replace("\n", " ") for t in text]
    else:  # single string
        cleaned_text = text.replace("\n", " ")
    resp = client.embeddings.create(model=EMBED_MODEL, input=cleaned_text)
    if isinstance(cleaned_text, list):
        return [item.embedding for item in resp.data]
    else:
        return resp.data[0].embedding

def rdflib_to_networkx(rdflib_graph):
    nx_graph = nx.MultiDiGraph()
    for s, p, o in rdflib_graph:
        nx_graph.add_edge(str(s), str(o), label=str(p))
    return nx_graph

def draw_pyvis_graph(nx_graph):
    net = Network(height="600px", width="100%", directed=True, notebook=False)
    net.from_nx(nx_graph)
    net.repulsion(node_distance=200, central_gravity=0.33, spring_length=100, spring_strength=0.10, damping=0.95)
    return net
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
        ["Logistic Regression", "MultinomialNB", "LegalBERT (Eurlex)", "SentenceTransformer", "LLM Model", "Knowledge Graphs"]
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

        elif model_choice == "SentenceTransformer":
            model = joblib.load("sentence_transformer_model.joblib")
            gdpr_texts = []
            gdpr_map = {}
            for article in gdpr_data:
                number, title = article["article_number"], article["article_title"]
                body = " ".join([f"{k} {v}" for sec in article["sections"] for k, v in sec.items()])
                full_text = f"Article {number}: {title}. {body}"
                gdpr_map[number] = {
                    "title": title,
                    "text": full_text
                }
                gdpr_texts.append(full_text)

            gdpr_embeddings = model.encode(gdpr_texts, convert_to_numpy=True)

            chunks = chunk_policy_text(policy_text)
            chunk_embeddings = model.encode(chunks, convert_to_numpy=True)

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
            result = {
                "overall_compliance_percentage": overall,
                "relevant_articles_analyzed": counted_articles,
                "total_policy_chunks": len(chunks),
                "article_scores": article_scores
            }

        elif model_choice == "LLM Model":
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)
            EMBED_MODEL = "text-embedding-3-small"
            gdpr_embeddings = {}
            gdpr_map = {}
            for art in gdpr_data:
                number, title = art["article_number"], art["article_title"]
                art_text = prepare_article_text(art)
                gdpr_embeddings[art["article_number"]] = {
                    "embedding": get_embedding(art_text),
                    "title": art["article_title"]
                }
                gdpr_map[number] = {"title": title, "text": art_text}
            chunks = chunk_policy_text(policy_text)
            chunk_embeddings = get_embedding(chunks)
            gdpr_embedding_vectors = [v["embedding"] for v in gdpr_embeddings.values()]
            sim_matrix = cosine_similarity(gdpr_embedding_vectors, chunk_embeddings)

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
            result = {
                "overall_compliance_percentage": overall,
                "relevant_articles_analyzed": counted_articles,
                "total_policy_chunks": len(chunks),
                "article_scores": article_scores
            }
        elif model_choice == "Knowledge Graphs":
            EMBED_MODEL = "all-MiniLM-L6-v2"
            model = SentenceTransformer(EMBED_MODEL)
            TOP_N = 1
            BASE_URI = "http://example.org/gdpr#"
            gdpr_embeddings = {}
            gdpr_map = {}
            for art in gdpr_data:
                number, title = art["article_number"], art["article_title"]
                art_text = prepare_article_text(art)
                gdpr_embeddings[art["article_number"]] = {
                    "embedding": model.encode(art_text),
                    "title": art["article_title"],
                    "uri": URIRef(f"{BASE_URI}Article{art['article_number']}")
                }
                gdpr_map[number] = {"title": title, "text": art_text}
            g = Graph()
            EX = Namespace(BASE_URI)
            g.bind("ex", EX)

            # Add article nodes
            for num, info in gdpr_embeddings.items():
                g.add((info["uri"], RDF.type, EX.Article))
                g.add((info["uri"], RDFS.label, Literal(f"Article {num}: {info['title']}")))
            # Extract GDPR article vectors
            article_nums = list(gdpr_embeddings.keys())
            article_vectors = np.array([gdpr_embeddings[num]["embedding"] for num in article_nums])

            # Score tracking
            total_score = 0
            counted_sections = 0
            chunks = chunk_policy_text(policy_text)
            report = []
            presence_threshold = 0.35

            # Process each policy chunk
            for idx, text in enumerate(chunks, start=1):
                if not text.strip():
                    continue

                # RDF section node
                sec_uri = URIRef(f"{BASE_URI}PolicySection{idx}")
                g.add((sec_uri, RDF.type, EX.PolicySection))
                g.add((sec_uri, RDFS.label, Literal(f"Section {idx}")))

                # Embed section
                sec_emb = model.encode(text)

                # Similarities to all articles
                sims = []
                for i, art_num in enumerate(article_nums):
                    art_emb = article_vectors[i]
                    sim = cosine_similarity([sec_emb], [art_emb])[0][0]
                    sims.append({
                        "article": art_num,
                        "title": gdpr_embeddings[art_num]["title"],
                        "similarity": round(sim, 4),
                        "uri": gdpr_embeddings[art_num]["uri"],
                        "text": gdpr_map[art_num]["text"]
                    })

                # Sort and pick best match
                sims.sort(key=lambda x: x["similarity"], reverse=True)
                top_match = sims[0]

                # Threshold filtering
                if top_match["similarity"] < presence_threshold:
                    continue

                # Compliance score
                score_pct = min(100, max(0, (top_match["similarity"] - presence_threshold) / (1 - presence_threshold) * 100))

                # Add RDF triples
                g.add((sec_uri, EX.relatesTo, top_match["uri"]))
                g.add((sec_uri, EX.similarityScore, Literal(top_match["similarity"], datatype=XSD.float)))

                
                g.serialize(destination="gdpr_policy_graph.ttl", format="turtle")

                total_score += score_pct
                counted_sections += 1

            # Final summary
            overall = round(total_score / counted_sections, 2) if counted_sections else 0
            result = {
                "overall_compliance_percentage": overall,
                "relevant_sections_analyzed": counted_sections,
                "total_policy_sections": len(chunks),
                "ttl": True
            }

        else:
            result = {}

    if result:
        st.subheader(f"✅ Overall Compliance Score: {result['overall_compliance_percentage']}%")
        st.markdown("---")
        st.subheader("📋 Detailed Article Breakdown")
        ttl_file_path = "gdpr_policy_graph.ttl"
        if result.get('article_scores'):
            for art_num, data in sorted(result['article_scores'].items(), key=lambda x: -x[1]['compliance_percentage']):
                with st.expander(f"Article {art_num} - {data['article_title']} ({data['compliance_percentage']}%)"):
                    st.write(f"**Similarity Score**: {data['similarity_score']}")
                    st.write(f"**Matched Text**:\n\n{data['matched_text_snippet']}")
        elif result.get("ttl") and os.path.exists(ttl_file_path):
            st.markdown("---")
            st.subheader("🧠 Interactive RDF Graph Visualization")

            g = Graph()
            g.parse(ttl_file_path, format="ttl")

            nx_graph = rdflib_to_networkx(g)
            net = draw_pyvis_graph(nx_graph)

            # Save the interactive graph temporarily
            net.save_graph("rdf_graph.html")
            HtmlFile = open("rdf_graph.html", "r", encoding="utf-8").read()

            # Display interactive graph inside Streamlit
            components.html(HtmlFile, height=650, scrolling=True)
        
        else:
            st.info("No article scores or RDF graph to display.")

else:
    st.info("Please upload both a GDPR JSON file and a company policy text file to begin.")
