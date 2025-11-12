import streamlit as st
import sys
from pathlib import Path

# Adjust path so we can import from src
sys.path.append(str(Path(__file__).resolve().parents[0] / "src"))

from graph.app import compiled_graph, QAState

st.set_page_config(page_title="Financial Document Analyzer", 
                   page_icon="💵",
                   layout="wide",
                   initial_sidebar_state=None,
                   menu_items={})

st.title("Financial Document Analyzer")

# User input for question
question = st.text_input("Ask a question about the financial documents:")

if st.button("Get Answer"):
    if question:
        with st.spinner("Searching for answer..."):
            # Run the LangGraph
            inputs = QAState(question=question)
            result = compiled_graph.invoke(inputs)

            answer = result.get("answer", {}).get("answer", "No answer found.")
            
            st.subheader("Answer:")
            st.write(answer)
            
    else:
        st.warning("Please enter a question.")

st.markdown("---")
st.write("Demo")
