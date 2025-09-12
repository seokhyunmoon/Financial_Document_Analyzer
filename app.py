# app.py
import gradio as gr
from langgraph.graph import StateGraph, START, END

# State
from graph.state import RAGState

# Nodes
from graph.node import retrieve_node, generate_node

# Indexing Utils
from ingest.loader import load_docs
from ingest.splitter import recursive_chunking
from ingest.index import add_chunks, reset_index, index_status

g = StateGraph(RAGState) 
g.add_node("retrieve", retrieve_node)
g.add_node("generate", generate_node)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "generate")
g.add_edge("generate", END)
APP = g.compile()


with gr.Blocks(title="Mini LangGraph RAG (Gemini Free)") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1) Documents")
            files = gr.File(file_count="multiple", file_types=[".txt",".md",".log"])
            status = gr.Markdown(index_status())

            def do_index(fs):
                if not fs:
                    return "업로드된 파일이 없습니다."
                paths = [f.name for f in fs]
                raw_docs = load_docs(paths)
                chunks = recursive_chunking(raw_docs, chunk_size=1000, chunk_overlap=150, reset_chunk_id_per_doc=True)
                n = add_chunks(chunks)
                return f"인덱싱 완료: {n} chunks"

            gr.Button("Index").click(do_index, inputs=[files], outputs=[status])
            gr.Button("Reset Index").click(lambda: (reset_index() or index_status()), outputs=[status])

        with gr.Column(scale=2):
            gr.Markdown("### 2) Ask")
            chat = gr.Chatbot(height=420, type="messages")
            q = gr.Textbox(placeholder="문서에 대해 질문하세요")
            src = gr.Markdown("**Sources will appear here.**")
            
            def _to_messages(history):
                """[(user, assistant), ...] 형식이 들어오면 messages 포맷으로 변환"""
                if history and isinstance(history[0], tuple):
                    msgs = []
                    for u, a in history:
                        if u: msgs.append({"role": "user", "content": u})
                        if a: msgs.append({"role": "assistant", "content": a})
                    return msgs
                return history or []


            def ask(message, history):
                # 1) 히스토리를 messages 포맷으로 통일
                history = _to_messages(history)

                # 2) 사용자 메시지 추가
                history.append({"role": "user", "content": message})

                # 3) LangGraph 실행
                out = APP.invoke({
                    "question": message,
                    "answer": "",
                    "chat_history": history,  # 지금은 노드에서 안 써도 OK
                    "docs": [],
                    "sources": []
                })

                # 4) 모델 답변을 assistant 메시지로 추가
                history.append({"role": "assistant", "content": out["answer"]})

                # 5) Sources 렌더링
                src_md = "### Sources\n" + (
                    "\n".join(
                        f"- [{s['id']}] {s.get('source','?')}" + (f" p.{s.get('page')}" if s.get('page') else "")
                        for s in out["sources"]
                    ) if out["sources"] else "_No sources_"
                )

                # Chatbot, 입력창 비우기, 소스 패널
                return history, "", src_md


            q.submit(ask, [q, chat], [chat, q, src])

demo.launch()