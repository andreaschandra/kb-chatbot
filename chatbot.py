import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

load_dotenv()


class KnowledgeBaseChatbot:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        self.llm = ChatAnthropic(model="claude-3-5-haiku-latest")
        self.vector_store = None
        self.conversation_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )
        # self.memory = MemorySaver()

    def load_documents(self, directory_path):
        """Load documents from a directory"""

        loaders = [
            DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader),
        ]

        documents = []
        for loader in loaders:
            documents.extend(loader.load())

        return documents

    def process_documents(self, documents):
        """Split documents into chunks and create embeddings"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        self.vector_store = Chroma.from_documents(
            documents=chunks, embedding=self.embeddings, persist_directory=".chroma_db"
        )

        return len(chunks)

    def setup_conversation_chain(self):
        """Setup the conversational retrieval chain"""
        if self.vector_store is None:
            raise ValueError(
                "Vector store is not initialized. Load and process documents first."
            )

        retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            output_key="answer",
        )

    def ask_question(self, question):
        if self.conversation_chain is None:
            raise ValueError(
                "Conversation chain is not set up. Call setup_conversation_chain() first."
            )

        response = self.conversation_chain.invoke(
            {"question": question, "chat_history": ""}
        )

        return response["answer"], response["source_documents"]
