"""Render concept-map data as an interactive HTML network graph (pyvis).

Takes the {"nodes": [...], "edges": [...]} produced by
features.generate_concept_map and returns a self-contained HTML string that
Streamlit can embed with st.components.v1.html. Users can drag nodes around,
zoom, and hover - a genuinely interactive concept map.
"""

from pyvis.network import Network


def build_concept_map_html(graph: dict, height: str = "600px") -> str:
    """Return interactive HTML for the given concept-map graph."""
    net = Network(
        height=height,
        width="100%",
        bgcolor="#0E1117",
        font_color="#FAFAFA",
        directed=True,
    )
    # A gentle physics layout so the map self-organizes nicely.
    net.barnes_hut(gravity=-8000, spring_length=160, spring_strength=0.02)

    for node in graph.get("nodes", []):
        net.add_node(
            node["id"],
            label=node.get("label", node["id"]),
            color="#6C5CE7",
            shape="dot",
            size=22,
        )

    for edge in graph.get("edges", []):
        net.add_edge(
            edge["source"],
            edge["target"],
            title=edge.get("label", ""),
            label=edge.get("label", ""),
            color="#888",
        )

    # generate_html() returns the full standalone page as a string.
    return net.generate_html(notebook=False)
