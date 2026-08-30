from RAG import *
import os
import time
import tempfile
from streamlit_chat import message
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

@timer
def display_messages():
    """Display the chat history."""
    st.subheader("Chat History")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

@timer
def process_input():
    """Process the user input and generate an assistant response."""
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("Thinking..."):
            try:
                st.session_state["assistant"].load_index()
                retrieved_docs = st.session_state["assistant"].search(
                    user_text,
                    top_k=st.session_state["retrieval_k"],
                )
            except ValueError as e:
                retrieved_docs = str(e)


        pre_prompt = """You are a RAG system(in Persian Language), some question will be asked by the user, you need to retrieved_docs based on only the retreived docs and only the question asked and not any irrelevant retrieved docs."""
        retrieved_docs_str = "\n\n".join(text for text, _ in retrieved_docs)
        prompt = pre_prompt + f'\n\nUser question:\n{user_text}' + '\n' + 'Retrieved docs:\n' + retrieved_docs_str
        llm = LLM()
        agent_text = llm.llm_call(prompt=prompt)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append(('retrieved docs: \n' + retrieved_docs_str, True))
        st.session_state["messages"].append((agent_text, False))

@timer
def read_and_save_file():
    """Handle file upload and ingestion."""
    st.session_state["assistant"].clear_data()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}..."):
            t0 = time.time()
            st.session_state["assistant"].process_pdf_to_faiss(file_path)
            t1 = time.time()

        st.session_state["messages"].append(
            (f"Ingested {file.name} in {t1 - t0:.2f} seconds", False)
        )
        os.remove(file_path)

@timer
def page():
    """Main app page layout."""
    # below code made the Message section shaking...
    # if len(st.session_state) == 0:
    #     st.session_state["messages"] = []
    #     st.session_state["assistant"] = PDFtoFAISS(chunk_size=500)
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "assistant" not in st.session_state:
        st.session_state["assistant"] = PDFtoFAISS(chunk_size=500)


    st.header("📄 PDF Q&A with RAG 📄")

    st.subheader("Upload a Document")
    st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    # Retrieval settings
    st.subheader("Settings")
    st.session_state["retrieval_k"] = st.slider(
        "Number of Retrieved Results (k)", min_value=1, max_value=10, value=5
    )
# Display messages and text input
    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

    # Clear chat
    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.session_state["assistant"].clear_data()


if __name__ == "__main__":
    page()
