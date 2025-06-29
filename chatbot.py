from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from typing_extensions import List, TypedDict


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class KnowledgeBaseChatbot:
    def __init__(self):
        self.llm = init_chat_model(
            "claude-3-5-haiku-latest", model_provider="anthropic"
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.vector_store = Chroma(
            collection_name="docs",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db",
        )
        self.graph = self.build_graph()
        self.prompt = hub.pull("rlm/rag-prompt")

    def load_documents(self, directory_path):
        """Load documents from a directory"""
        # This method should be implemented to load documents from the specified directory
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
        # This method should be implemented to process the documents and create embeddings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        result = self.vector_store.add_documents(chunks)
        print(f"Added {len(result)} chunks to the vector store.")

    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State):
        """Ask a question and get an answer from the conversation chain"""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def build_graph(self):
        # Compile application and test
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        return graph

    def ask_question(self, question):
        response = self.graph.invoke({"question": question})
        print(f"Response from graph: {response}")
        print(f"Answer: {response['answer']}")

        return response["answer"], response["context"]
