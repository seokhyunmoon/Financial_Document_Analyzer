import streamlit as st
import sys
from pathlib import Path

# src import
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from utils.logger import get_logger
from services.ingest import ingest_files
from utils.inventory import list_available_documents

logger = get_logger(__name__)
st.set_page_config(page_title="Ingest", page_icon="📥", layout="wide")
st.title("Upload")

st.markdown("Upload Financial Document PDF(s) to ask Questions")

# Controls
c1, c2 = st.columns([3, 1])
with c1:
    files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
with c2:
    reset_col = st.checkbox("Reset Database", value=False,
                            help="Delete all Datas in Vector Database.")

run = st.button("Run Ingest", type="primary", disabled=(not files))

if run and files:
    with st.status("Ingest running...", expanded=True) as status:
        st.write(f"- Files: {len(files)}")
        st.write(f"- Reset collection: {reset_col}")
        try:
            results = ingest_files(files, reset=reset_col)
            st.success("Ingest finished.")
            status.update(state="complete")
        except Exception as e:
            st.error(f"Ingest failed: {e}")
            st.stop()

    # Summary
    st.subheader("Result")
    for r in results:
        with st.expander(f"{r['doc_id']} — chunks: {r['n_chunks']} | vectors: {r['n_vectors']} | time: {r['elapsed_sec']}s", expanded=True):
            st.write(f"Elements: {r['n_elements']}")
            st.write({k: str(v) for k, v in r["paths"].items()})

    st.divider()
    st.subheader("Indexed documents (from embeddings on disk)")
    docs = list_available_documents()
    if docs:
        for did, cnt in docs:
            st.write(f"- **{did}** — {cnt} chunks")
    else:
        st.caption("No embedded documents found yet.")
else:
    st.caption("Select PDF and **Run Ingest**")