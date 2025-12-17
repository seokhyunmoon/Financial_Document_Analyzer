# main.py
import streamlit as st
import sys
import time
from pathlib import Path
import re

# src import
sys.path.append(str(Path(__file__).resolve().parents[0] / "src"))
from graph.state import compiled_graph, QAState
from utils.inventory import list_available_documents
from services.ingest import ingest_files
from utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Financial Document Analyzer", page_icon="💵", layout="wide")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False


# --- UI Components ---
st.title("Financial Document Analyzer")

# Document selection for filtering
available_docs = [doc[0] for doc in list_available_documents()]
doc_options = ["All Documents"] + available_docs

if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = "All Documents"

selected_doc = st.selectbox(
    "Filter by document:",
    options=doc_options,
    key="selected_doc"
)

source_doc_filter = selected_doc if selected_doc != "All Documents" else None

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "details" in message: # Display citations/sources if available
            with st.expander("Sources"):
                st.json(message["details"])

# Chat input
if prompt := st.chat_input("Send a message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            inputs = QAState(question=prompt.strip(), topk=10, source_doc=source_doc_filter)
            logger.info(
                "[INFO] Invoking QA graph",
                extra={"question_len": len(prompt.strip()), "source_filter": source_doc_filter or "ALL"},
            )
            try:
                result = compiled_graph.invoke(inputs)
                logger.info("[OK] Graph invocation completed")
            except Exception as exc:
                logger.error(f"[ERROR] Graph invocation failed: {exc}")
                st.error("Sorry, something went wrong while generating the answer.")
                st.stop()
            
            answer_dict = result.get("answer", {}) or {}
            answer_text = answer_dict.get("answer", "No answer found.")
            
            message_placeholder = st.empty()
            full_response = ""
            for chunk in answer_text.split():
                full_response += chunk + " "
                time.sleep(0.05) # 50ms delay
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response) # Display final response without cursor

            # Show sources
            # hits = result.get("hits", []) or []
            
            # details = {"hits": hits}
            # if hits:
            #     with st.expander("Show Sources"):
            #         st.json(details)
            
            sources = answer_dict.get("citations", []) or []
            logger.info(
                "[OK] Response ready",
                extra={"answer_len": len(answer_text), "citations": len(sources)},
            )
            # (선택) 상단에 인덱스만 간단히 표기
            idxs = [c.get("i") for c in sources if isinstance(c, dict) and "i" in c]
            if idxs:
                st.caption(f"Cited Sources: {idxs}")

            # 토글엔 '사용된 소스'만 JSON으로
            if sources:
                with st.expander("Sources (used)"):
                    st.json(sources)        
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer_text,
        "details": sources
    })
    st.rerun()

st.divider()

# --- Ingest UI ---
st.header("Upload Financial Documents")
files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("Upload", type="primary", disabled=(not files), use_container_width=True):
        with st.status("Ingest running...", expanded=True) as status:
            try:
                logger.info(f"[INFO] Starting upload for {len(files)} file(s)")
                ingest_files(files, reset=False)
                logger.info("[OK] Upload successful")
                st.success("Uploaded")
                st.rerun() # Rerun to update the list of indexed documents
            except Exception as e:
                logger.error(f"[ERROR] Upload failed: {e}")
                st.error(f"Upload failed: {e}")

with col2:
    if st.session_state.get("confirm_reset", False):
        st.warning("Are you sure you want to reset the database? All indexed data will be lost.")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes", type="primary", use_container_width=True):
                with st.status("Resetting database...", expanded=True):
                    try:
                        logger.info("[INFO] Reset requested via UI")
                        ingest_files([], reset=True)
                        logger.info("[OK] Reset completed")
                        st.success("Database reset successfully.")
                        st.session_state.confirm_reset = False
                        st.rerun()
                    except Exception as e:
                        logger.error(f"[ERROR] Database reset failed: {e}")
                        st.error(f"Database reset failed: {e}")
        with c2:
            if st.button("Cancel", type="secondary", use_container_width=True):
                st.session_state.confirm_reset = False
                st.rerun()
    else:
        if st.button("Reset Database", type="secondary", use_container_width=True):
            st.session_state.confirm_reset = True
            st.rerun()

st.divider()
st.subheader("Documents")
docs_ingest = list_available_documents()
for did, cnt in docs_ingest:
    st.write(f"- **{did}** — {cnt} chunks")
