# main.py
import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[0] / "src"))
from graph.state import compiled_graph, QAState
from utils.inventory import list_available_documents


st.set_page_config(page_title="Financial Document Analyzer", page_icon="💵", layout="wide")
st.title("Financial Document Analyzer")


docs = list_available_documents()
doc_options = ["All documents"] + [f"{doc_id}  ({cnt} chunks)" for doc_id, cnt in docs]


# Controls
c1, c2 = st.columns([4, 1])
with c1:
    question = st.text_input("Question", placeholder="Ask a question about the financial documents...")
with c2:
    topk = st.slider("Top-K", min_value=3, max_value=30, value=10, step=1)

doc_choice = st.selectbox("Filter by document (optional)", options=doc_options, index=0,
                          help="Choose a specific document or search all documents.")

run = st.button("Get Answer", type="primary")

if run:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        # Filter by Doc name
        selected_doc = None
        if doc_choice != "All documents":
            # "DOCID  (N chunks)" → DOCID extract
            selected_doc = doc_choice.split("  (", 1)[0]

        with st.spinner("Retrieving and generating..."):
            inputs = QAState(
                question=question.strip(),
                topk=topk,
                source_doc=selected_doc,
            )
            result = compiled_graph.invoke(inputs)

        # Answer
        answer_dict = result.get("answer", {}) or {}
        answer_text = answer_dict.get("answer", "No answer.")
        st.subheader("Answer")
        st.write(answer_text)

        # Hits & Citations
        hits = result.get("hits", []) or []
        cites = answer_dict.get("citations", []) or []

        st.markdown("---")
        st.caption(f"Top-K returned: **{len(hits)}**")
        if cites:
            st.caption(f"Citations: {cites}")

        for i, h in enumerate(hits, 1):
            header = f"[{i}] {h.get('doc_id')}  p{h.get('page_start')}-{h.get('page_end')}  — {h.get('type')}"
            with st.expander(header, expanded=(i in cites)):
                st.write((h.get("text") or "").strip() or "_(empty)_")

st.markdown("---")