import os
import json
import streamlit as st
from collections import defaultdict

# -------------------------------
# Helper functions
# -------------------------------
def load_parsed_policies(folder_path):
    files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".json") and f.startswith("parsed-")
    ])
    return files

def format_policy_name(filename):
    name = filename.replace("parsed-", "").replace("_policy.json", "").replace("_", " ")
    return name.title()

def calculate_avg_score(matches):
    scores = [m["similarity_score"] for sec in matches.values() for m in sec]
    return round(sum(scores) / len(scores), 4) if scores else 0.0

def deduplicate_passages(matches):
    seen = set()
    cleaned = {}
    for sec, passages in matches.items():
        unique_passages = []
        for p in passages:
            norm = p["matched_policy_passage"].strip().lower()
            prefix = norm[:60]
            if prefix in seen:
                continue
            seen.add(prefix)
            unique_passages.append(p)
        if unique_passages:
            cleaned[sec] = unique_passages
    return cleaned

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Parsed Policy Viewer", layout="wide")
st.title("🗂️ All Parsed GDPR Matches Across Companies")

parsed_folder = "website-scraping/new_parsed_policies"
parsed_files = load_parsed_policies(parsed_folder)

if not parsed_files:
    st.warning("No parsed policy files found.")
    st.stop()

# Load all company data
company_data = []
for file in parsed_files:
    with open(os.path.join(parsed_folder, file), "r", encoding="utf-8") as f:
        parsed_json = json.load(f)

    policy_name = format_policy_name(file)
    article_blocks = []

    for article in parsed_json:
        cleaned = deduplicate_passages(article["matches"])
        avg = calculate_avg_score(cleaned)
        article_blocks.append((article, cleaned, avg))

    # Average of article-level averages
    if article_blocks:
        overall_avg = round(sum(a[2] for a in article_blocks) / len(article_blocks), 2)
    else:
        overall_avg = 0.0

    company_data.append((policy_name, overall_avg, article_blocks))

# -------------------------------
# Tab Layout
# -------------------------------
tabs = st.tabs([f"{name} ({avg*100:.1f}%)" for name, avg, _ in company_data])

for i, (name, avg, articles) in enumerate(company_data):
    with tabs[i]:
        st.header(f"{name} — Average Similarity: {avg*100:.1f}%")
        st.markdown("---")

        for article, matches, article_avg in articles:
            st.subheader(f"🧾 Article {article['article_number']}: {article['article_title']}")
            st.caption(f"Average Similarity for this Article: **{article_avg}**")

            for sec_num, passages in matches.items():
                for passage in passages:
                    st.markdown(f"**Section {sec_num}** — Score: `{passage['similarity_score']}`")
                    st.write(passage["matched_policy_passage"])
                    if "gdpr_clause" in passage:
                        st.caption(f"🔗 Matched GDPR Clause: _{passage['gdpr_clause']}_")
                    st.markdown("---")
