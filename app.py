"""Streamlit app for a Knowledge Base Chatbot using LangChain."""

import os
import shutil
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from chatbot import KnowledgeBaseChatbot


def main():
    """User interface for the Knowledge Base Chatbot using Streamlit."""
    st.set_page_config(page_title="Knowledge Base Chatbot", page_icon=":robot_face:")
    st.title("Knowledge Base Chatbot")
    st.sidebar.title("📚 Document Management")

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = KnowledgeBaseChatbot()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())

    with st.sidebar:
        st.header("Upload documents")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
        )

        if st.button("Process Documents"):
            if uploaded_files:
                temp_dir = "temp_docs"
                os.makedirs(temp_dir, exist_ok=True)

                for uploaded_file in uploaded_files:
                    with open(f"{temp_dir}/{uploaded_file.name}", "wb") as f:
                        f.write(uploaded_file.getbuffer())

                with st.spinner("Processing documents..."):
                    documents = st.session_state.chatbot.load_documents(temp_dir)
                    chunk_count = st.session_state.chatbot.process_documents(documents)

                st.success(
                    f"Processed {len(documents)} documents into {chunk_count} chunks."
                )

                shutil.rmtree(temp_dir)

    st.header("💬 Chat with your documents")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = st.session_state.chatbot.ask_question(prompt)
                    st.markdown(answer)

                    if sources:
                        with st.expander("Source Documents"):
                            for index, source in enumerate(sources, 1):
                                st.write(f"**Source {index}:**")
                                st.write(
                                    source.page_content[:200] + "..."
                                )  # Display first 200 chars
                                st.write("---")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    load_dotenv()
    main()
