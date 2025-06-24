# 1. Import Libraries
import os
import re
import json
import nltk
import spacy
from nltk.tokenize import sent_tokenize
from pprint import pprint

# Download NLTK assets
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# 2. Load Policy Text
def load_policy_text(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()

# 3. Domain-Based Section Anchors
SECTION_DOMAINS = {
    "Article 1 - Data Collection": ["we collect", "information we collect", "data you provide", "receive from third", "automatically collect"],
    "Article 2 - Data Usage": ["we use", "purpose", "processing"],
    "Article 3 - Data Sharing": ["we share", "we disclose", "third-party"],
    "Article 4 - Data Retention": ["how long we keep", "retention", "storage duration"],
    "Article 5 - User Rights": ["your rights", "access and control", "opt out", "request"],
    "Article 6 - Security": ["security", "protect", "safeguards"],
    "Article 7 - Cookies": ["cookies", "tracking", "browser"],
    "Article 8 - Legal Basis": ["legal basis", "lawful", "gdpr", "eea"],
    "Article 9 - International Transfer": ["transfer", "international", "outside eu"],
    "Article 10 - Contact Information": ["contact", "reach us", "email", "inquiries"]
}

# 4. Title Canonicalizer
def canonicalize_title(label):
    return label

# 5. Domain-based Bucketing

def group_into_sections(text):
    sentences = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 20]
    sections = {label: [] for label in SECTION_DOMAINS}
    sections["Article 11 - General"] = []

    for s in sentences:
        matched = False
        s_lower = s.lower()
        for label, cues in SECTION_DOMAINS.items():
            if any(cue in s_lower for cue in cues):
                sections[label].append(s)
                matched = True
                break
        if not matched:
            sections["Article 11 - General"].append(s)
    return sections

# 6. Parser

def gdpr_aligned_parser_single_paragraph(text):
    sectioned = group_into_sections(text)
    output = []
    for label, sents in sectioned.items():
        if not sents:ß
            continue
        full_block = " ".join(sents)
        output.append({
            "title": canonicalize_title(label),
            "paragraphs": [full_block]
        })
    return output

# 7. Parse All .txt Files in Folder
def parse_all_policies(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            path = os.path.join(input_folder, filename)
            text = load_policy_text(path)
            parsed = gdpr_aligned_parser_single_paragraph(text)

            out_name = filename.replace(".txt", ".json")
            out_path = os.path.join(output_folder, out_name)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)

            print(f" Parsed and saved: {out_name}")

# 8. Run Batch Processing
parse_all_policies("website-scraping/scraped-policies-2", "website-scraping/parsed-json")

