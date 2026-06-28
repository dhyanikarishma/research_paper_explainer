"""Render concept-map data as an interactive HTML network graph (pyvis).

Takes the {"nodes": [...], "edges": [...]} produced by
features.generate_concept_map and returns a self-contained HTML string that
Streamlit can embed with st.components.v1.html. Users can drag nodes around,
zoom, and hover - a genuinely interactive concept map.

Security: node/edge labels originate from the LLM/paper (untrusted) and are
injected into an HTML page, so every label is HTML-escaped (escape_label)
to prevent stored-XSS via crafted text.
"""

from pyvis.network import Network

from src.security import escape_label


def build_concept_map_html(graph: dict, height: str = "600px") -> str:
    """Return interactive HTML for the given concept-map graph."""
    net = Network(
        height=height,
        width="100%",
        bgcolor="#0E1117",
        font_color="#FAFAFA",
        directed=True,
    )
    net.barnes_hut(gravity=-8000, spring_length=160, spring_strength=0.02)

    for node in graph.get("nodes", []):
        net.add_node(
            escape_label(node["id"]),
            label=escape_label(node.get("label", node["id"])),
            color="#6C5CE7",
            shape="dot",
            size=22,
        )

    for edge in graph.get("edges", []):
        label = escape_label(edge.get("label", ""))
        net.add_edge(
            escape_label(edge["source"]),
            escape_label(edge["target"]),
            title=label,
            label=label,
            color="#888",
        )

    return net.generate_html(notebook=False)
