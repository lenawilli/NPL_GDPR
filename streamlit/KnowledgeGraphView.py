import streamlit as st
from rdflib import Graph, RDFS, Namespace, URIRef, RDF, Literal
from pyvis.network import Network
import streamlit.components.v1 as components

# --- Load RDF ---
g = Graph()
g.parse("../KnowledgeGraph/gdpr_policy_graph.ttl", format="ttl")

BASE_URI = "http://example.org/gdpr#"
EX = Namespace(BASE_URI)

# --- UI Layout ---
st.set_page_config(layout="wide")
st.title("🕸️ GDPR Knowledge Graph Visualizer")

# --- Get Filter Options ---
articles = sorted({str(o).split(" ")[1] for s, p, o in g.triples((None, RDFS.label, None)) if "Article" in str(o)})
sections = sorted({
    str(label)
    for sec in g.subjects(RDF.type, EX.PolicySection)
    for label in g.objects(sec, RDFS.label)
})

col1, col2 = st.columns([1, 3])
selected_article = col1.selectbox("🔍 Filter by Article Number", ["All"] + articles)
selected_section = col2.selectbox("📄 Filter by Policy Section", ["All"] + sections)

# --- Determine nodes to show based on filters ---
def get_related_nodes_for_section(section_label):
    # Find the PolicySection node
    matching_sections = [
        s for s, p, o in g.triples((None, RDFS.label, Literal(section_label)))
        if (s, RDF.type, EX.PolicySection) in g
    ]
    if not matching_sections:
        return set()
    sec_node = matching_sections[0]

    clause_nodes = set(o for _, _, o in g.triples((sec_node, EX.relatesToClause, None)))
    article_nodes = set(o for c in clause_nodes for _, _, o in g.triples((c, EX.partOf, None)))
    return {sec_node} | clause_nodes | article_nodes

def get_related_nodes_for_article(article_number):
    # Find Article node(s)
    article_nodes = set(
        s for s, p, o in g.triples((None, RDFS.label, None))
        if (s, RDF.type, EX.Article) in g and f"Article {article_number}" in str(o)
    )
    if not article_nodes:
        return set()
    article_node = list(article_nodes)[0]

    # Clauses of the article
    clause_nodes = set(o for _, _, o in g.triples((None, EX.partOf, article_node)))

    # Find all policy sections relating to these clauses
    policy_sections = set(
        s for s, p, o in g.triples((None, EX.relatesToClause, None)) if o in clause_nodes
    )
    return {article_node} | clause_nodes | policy_sections

def expand_with_neighbors(graph, nodes):
    expanded = set(nodes)
    for node in nodes:
        # Outgoing neighbors
        for _, _, o in graph.triples((node, None, None)):
            expanded.add(o)
        # Incoming neighbors
        for s, _, _ in graph.triples((None, None, node)):
            expanded.add(s)
    return expanded

if selected_section != "All" and selected_article != "All":
    # Filter by both: intersection of sets
    nodes_for_section = get_related_nodes_for_section(selected_section)
    nodes_for_article = get_related_nodes_for_article(selected_article)
    base_nodes = nodes_for_section & nodes_for_article
elif selected_section != "All":
    base_nodes = get_related_nodes_for_section(selected_section)
elif selected_article != "All":
    base_nodes = get_related_nodes_for_article(selected_article)
else:
    # Show everything
    base_nodes = set()
    for s, p, o in g:
        base_nodes.add(s)
        base_nodes.add(o)

nodes_to_show = expand_with_neighbors(g, base_nodes)
# --- Initialize network ---
net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
net.force_atlas_2based(gravity=-30)

added_nodes = set()

def get_type(uri):
    for s, p, o in g.triples((uri, RDFS.label, None)):
        label = str(o)
        if label.startswith("Article"):
            return "article"
        elif label.startswith("Art."):
            return "clause"
        elif label.startswith("Section"):
            return "section"
    return "unknown"

def get_label(g, node):
    for _, _, label in g.triples((node, RDFS.label, None)):
        return str(label)
    return None

def color_by_type(node_type):
    return {
        "article": "#77B5FE",   # blue
        "clause": "#81C784",    # green
        "section": "#FFB74D",   # orange
    }.get(node_type, "#D3D3D3")

# --- Add nodes ---
for node in nodes_to_show:
    label = get_label(g, node) or (str(node).split("#")[-1] if isinstance(node, URIRef) else str(node))
    n_type = get_type(node)
    tooltip = label
    for val in g.objects(node, EX.similarityScore):
        tooltip += f"\nSimilarity: {val}"
    for val in g.objects(node, RDFS.comment):
        tooltip += f"\n{val}"
    net.add_node(str(node), label=label, title=tooltip, color=color_by_type(n_type))
    added_nodes.add(str(node))

# --- Add edges only between shown nodes ---
for s, p, o in g:
    if s in nodes_to_show and o in nodes_to_show:
        pred_label = p.split("#")[-1] if isinstance(p, URIRef) else str(p)
        net.add_edge(str(s), str(o), label=pred_label)

# --- Render ---
net.save_graph("graph.html")
with open("graph.html", "r", encoding="utf-8") as f:
    html = f.read()

components.html(html, height=780, scrolling=True)