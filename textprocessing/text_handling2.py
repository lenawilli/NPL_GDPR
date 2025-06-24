import json
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import torch
import nltk

nltk.download("punkt")

# Load model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Load GDPR baseline JSON
with open("gdpr_articles_baseline.json", "r", encoding="utf-8") as f:
    gdpr_data = json.load(f)

# Flatten GDPR clauses
gdpr_clauses = []
gdpr_map = []

for article in gdpr_data:
    for section in article["sections"]:
        for sec_num, clause in section.items():
            gdpr_clauses.append(clause)
            gdpr_map.append({
                "article_number": article["article_number"],
                "article_title": article["article_title"],
                "section_number": sec_num,
                "clause_text": clause
            })

# Combine questions with likely answer sentences: preserve meaning & complete context
def merge_question_with_list(sentences, window_size=3):
    merged = []
    i = 0
    while i < len(sentences):
        s = sentences[i]
        if s.endswith("?"):
            combined = s
            j = 1
            while j <= window_size and (i + j) < len(sentences):
                next_s = sentences[i + j].strip()

                # Accept list-like or explanatory answers
                if next_s[0].isupper() and (
                    not next_s.endswith("?") or  # declarative
                    any(punct in next_s for punct in [":", ";"]) or  # list indicator
                    len(next_s.split()) < 20  # short items
                ):
                    combined += " " + next_s
                    j += 1
                else:
                    break

            merged.append(combined)
            i += j
        else:
            merged.append(s)
            i += 1
    return merged

# Re-joins bullet points or numbered lists that got broken during tokenization:
def merge_list_items(sentences):
    merged = []
    buffer = ""
    for s in sentences:
        stripped = s.strip()
        if re.match(r"^[-•\d]+\s", stripped) or stripped.startswith("•"):
            buffer += " " + stripped
        else:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append(stripped)
    if buffer:
        merged.append(buffer.strip())
    return merged

# Merge all questions with guaranteed following context
def merge_questions_with_context(sentences, forced_context=4):
    merged = []
    skip = 0
    for i in range(len(sentences)):
        if skip > 0:
            skip -= 1
            continue
        s = sentences[i]
        if s.endswith("?"):
            combined = s
            for j in range(1, forced_context + 1):
                if (i + j) < len(sentences):
                    combined += " " + sentences[i + j]
            merged.append(combined)
            skip = forced_context
        else:
            merged.append(s)
    return merged

import re

# Fix kumbering spills (8., e.g.8.), Run-on capitalization (UKWe → UK. We) Extra whitespace or punctuation

def clean_sentence_artifacts(sent):
    # Remove common numbering artifacts like '8.', '9.' that leak into sentence ends
    sent = re.sub(r"\.\s*\d+\.", ".", sent)  # e.g., "...period. 8." → "...period."
    
    # Remove mid-sentence enumeration that got wrongly merged: "e.g.8." → "e.g."
    sent = re.sub(r"(?<=\w)\.(\d+)\.", ".", sent)

    # Fix spacing issues from bad merges: "UKWe" → "UK. We"
    sent = re.sub(r"([a-z])([A-Z])", r"\1. \2", sent)

    # Normalize multiple spaces
    sent = re.sub(r"\s{2,}", " ", sent)

    # Strip trailing junk punctuation
    sent = sent.strip(" .;")

    return sent.strip()

# Mapping function, core function. processes one privacy policy and outputs GDPR article matches
def map_policy_to_gdpr_with_windowing(raw_text, similarity_threshold=0.60, window_size=4):
    from hashlib import sha1

    # Tokenize and clean sentences

    #testing the order of functions:
   # sentences = [s.strip() for s in sent_tokenize(raw_text) if len(s.strip()) > 20]
  #  sentences = merge_question_with_list(sentences, window_size=3)
  #  sentences = merge_questions_with_context(sentences, forced_context=3)
  #  sentences = [clean_sentence_artifacts(s) for s in sentences]
  #  sentences = merge_question_with_list(sentences, window_size=3)

    sentences = sent_tokenize(raw_text)
    sentences = merge_list_items(sentences)                      
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  
    sentences = merge_question_with_list(sentences, window_size=3)     
    sentences = merge_questions_with_context(sentences, forced_context=2)  
    sentences = [clean_sentence_artifacts(s) for s in sentences]       

    # Create sentence windows ie overlapping chunks of multiple sentences (e.g., 4 in a row) for more context
    # this unit is compared to GDPR clauses using embeddings.
    blocks = []
   # for i in range(len(sentences)):
     #   window = " ".join(sentences[i:i+window_size])
     #   blocks.append((i, window))
    
    for i in range(len(sentences) - window_size + 1):
        window = " ".join(sentences[i:i+window_size])
        blocks.append((i, window))


    # Embed
    #Convert text to numerical vectors
    #Compute cosine similarity between policy blocks and GDPR clauses


    # Represent each block of your policy and each GDPR clause as a vector in a high-dimensional space, where similar meanings are closer together
    policy_embeddings = model.encode([b[1] for b in blocks], convert_to_tensor=True)
    gdpr_embeddings = model.encode(gdpr_clauses, convert_to_tensor=True)

    # Similarity Computes cosine similarity between each policy_embedding and each gdpr_embedding
    #Returns a 2D matrix where:
    #Rows = each policy block
    #Columns = each GDPR clause
    #Value at [i][j] = similarity score between policy block i and GDPR clause j
    
    cos_scores = util.cos_sim(policy_embeddings, gdpr_embeddings)

    seen_blocks = set()
    output = defaultdict(list)

    for idx, (sent_idx, block_text) in enumerate(blocks):
        block_id = sha1(block_text.encode()).hexdigest()
        if block_id in seen_blocks:
            continue

        best_idx = torch.argmax(cos_scores[idx]).item()
        best_score = cos_scores[idx][best_idx].item()

        if best_score >= similarity_threshold:
            gdpr_info = gdpr_map[best_idx]
            output[gdpr_info["article_number"]].append({
                "gdpr_article_title": gdpr_info["article_title"],
                "gdpr_section_number": gdpr_info["section_number"],
                "gdpr_clause": gdpr_info["clause_text"],
                "matched_policy_passage": block_text,
                "similarity_score": round(best_score, 4)
            })
            seen_blocks.add(block_id)

    # Clean structure
    final_output = []
    for article_number, matches in output.items():
        grouped = defaultdict(list)
        seen_sent_texts = set()

        for match in matches:
            text_hash = sha1(match["matched_policy_passage"].lower().encode()).hexdigest()
            if text_hash in seen_sent_texts:
                continue

            grouped[match["gdpr_section_number"]].append({
                "matched_policy_passage": match["matched_policy_passage"],
                "similarity_score": match["similarity_score"]
            })
            seen_sent_texts.add(text_hash)

        final_output.append({
            "article_number": article_number,
            "article_title": matches[0]["gdpr_article_title"],
            "matches": grouped
        })

    return final_output

"""
# --- single file test---

# Load raw privacy policy text
with open("website-scraping/scraped-policies-2/bitbucket_org_policy.txt", "r", encoding="utf-8") as f:
    raw_policy_text = f.read()

# Run mapping
structured_json = map_policy_to_gdpr_with_windowing(raw_policy_text)

# Save result
with open("parsed-ikea-gdpr.json", "w", encoding="utf-8") as f:
    json.dump(structured_json, f, indent=2, ensure_ascii=False)

print(" Output saved to parsed-bitbucket-gdpr.json (no duplicates).")

"""

import os
import json

# Directory paths
input_folder = "website-scraping/scraped-policies-2"
output_folder = "website-scraping/new_parsed_policies"
os.makedirs(output_folder, exist_ok=True)

# Loop over all .txt files
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(input_folder, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            raw_policy_text = f.read()

        print(f" Processing {filename}...")

        structured_json = map_policy_to_gdpr_with_windowing(raw_policy_text)

        # Output filename
        base_name = os.path.splitext(filename)[0]
        out_name = f"parsed-{base_name}.json"
        out_path = os.path.join(output_folder, out_name)

        with open(out_path, "w", encoding="utf-8") as out_file:
            json.dump(structured_json, out_file, indent=2, ensure_ascii=False)

        print(f" Saved to {out_path}")
