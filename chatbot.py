"""Main flow and functionality for knowledge base chatbot."""

from typing import Dict, List, Tuple
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


class KnowledgeBaseChatbot:
    """Chatbot class for setting up embeddings, vector store, llm, graph, and tools."""

    def __init__(self):
        print("init_chat_model...")
        self.llm = init_chat_model(
            "claude-3-5-haiku-latest", model_provider="anthropic"
        )

        print("init HF Embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        print("init ChromaDB...")
        self.vector_store = Chroma(
            collection_name="docs",
            embedding_function=embeddings,
            persist_directory="./.chroma_db",
        )

        print("Init ToolNode...")
        self.retrieve = self._create_retriever()
        self.tools = ToolNode([self.retrieve])

        self.memory = MemorySaver()

        print("Build graph...")
        self.build_graph()

        get_client()
        self.langfuse_handler = CallbackHandler()

    def load_documents(self, directory_path: str) -> List:
        """Load documents from a directory.

        Args:
            directory_path (str): directory location to load documents.

        Returns:
            _type_: _description_
        """

        # This method should be implemented to load documents from the specified directory
        loaders = [
            DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader),
        ]

        documents = []
        for loader in loaders:
            documents.extend(loader.load())

        return documents

    def process_documents(self, documents: List) -> int:
        """Split documents into chunks and transform into embeddings and store in vector store.

        Args:
            documents (List): list of documents to process.

        Returns:
            int: Number of chunks added to the vector store.
        """

        # This method should be implemented to process the documents and create embeddings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        result = self.vector_store.add_documents(chunks)
        print(f"Added {len(result)} chunks to the vector store.")

        return len(result)

    def _create_retriever(self):
        """Private method to create a retriever tool."""

        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information from the document knowledge base.
            Use this tool whenever you need to answer questions
            about documents, papers, or any content-related queries.
            """

            retrieved_docs = self.vector_store.similarity_search(query, k=3)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            retrieved_docs = [doc.__dict__ for doc in retrieved_docs]
            return serialized, retrieved_docs

        return retrieve

    def query_or_respond(self, state: MessagesState) -> Dict:
        """entry point for the chatbot to either
        query the knowledge base or respond to a user message.

        Args:
            state (MessagesState): Previous state of the chatbot, including messages.

        Returns:
            _type_: _description_
        """

        print(f"State messages in query_or_respond: {state['messages']}\n")

        messages = state["messages"]
        if not any(msg.type == "system" for msg in messages):
            system_msg = SystemMessage(
                content="You are a helpful assistant with access to a document knowledge base. "
                "When users ask questions about documents, papers, research, "
                "or any content that might be in the knowledge base, "
                "you MUST use the retrieve tool to search for relevant information first. "
                "Always use the retrieve tool for questions that could be answered from documents."
            )
            messages = [system_msg] + messages

        llm_with_tools = self.llm.bind_tools([self.retrieve])
        response = llm_with_tools.invoke(state["messages"])
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}

    def generate(self, state: MessagesState) -> Dict:
        """Return a response when the user asks a query
        that requires information from the knowledge base.

        Args:
            state (MessagesState): Previous state of the chatbot, including messages.

        Returns:
            Dict: A dictionary containing the generated response message.
        """
        # Get generated ToolMessages
        print("Trigger generate method...")
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = self.llm.invoke(prompt)
        return {"messages": [response]}

    def build_graph(self):
        """Build the state graph, node and edges for the chatbot."""

        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("query_or_respond", self.query_or_respond)
        graph_builder.add_node("tools", self.tools)
        graph_builder.add_node("generate", self.generate)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        graph = graph_builder.compile(checkpointer=self.memory)
        graph.get_graph().print_ascii()

        self.graph = graph

    def ask_question(self, question: str, thread_id: str = None) -> Tuple[str, List]:
        """Return a response to a user includes retrieved documents from the knowledge base.

        Args:
            question (_type_): _description_
            thread_id (str, optional): thread id in memory. Defaults to None.

        Returns:
            tuple: return reponse and list of retrieved documents.
        """

        if thread_id is None:
            thread_id = str(uuid4())

        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [self.langfuse_handler],
        }
        response = self.graph.invoke(
            {"messages": [{"role": "user", "content": question}]}, config
        )

        latest_response = response["messages"][-1]
        print(f"Response from graph: {response}\n")
        print(f"Latest response: {latest_response}\n")

        latest_tool_msg = next(
            (
                msg
                for msg in reversed(response["messages"])
                if msg.type == "tool" and hasattr(msg, "artifact") and msg.artifact
            ),
            None,
        )

        retrieved_docs = []
        if latest_tool_msg:
            for artifact in latest_tool_msg.artifact:
                if isinstance(artifact, dict):
                    doc = Document(
                        id=artifact["id"],
                        page_content=artifact["page_content"],
                        metadata=artifact["metadata"],
                        page_content_type=artifact["page_content"],
                    )
                else:
                    doc = artifact
                retrieved_docs.append(doc)

        # retrieved_docs = []
        # for msg in response["messages"]:
        #     if msg.type == "tool" and hasattr(msg, "artifact") and msg.artifact:
        #         for artifact in msg.artifact:
        #             if isinstance(artifact, dict):
        #                 # Convert dict to Document if necessary
        #                 doc = Document(
        #                     id=artifact["id"],
        #                     page_content=artifact["page_content"],
        #                     metadata=artifact["metadata"],
        #                     page_content_type=artifact["page_content"],
        #                 )
        #             else:
        #                 doc = artifact

        #             retrieved_docs.append(doc)

        print(f"Total retrieved documents: {len(retrieved_docs)}\n")
        print(f"Retrieved documents: {retrieved_docs}\n")
        print("=" * 50)

        return latest_response.content, retrieved_docs
