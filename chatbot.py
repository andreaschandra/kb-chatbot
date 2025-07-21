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
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from schema import PointSchema


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

    def _clean_message_sequence(self, messages: List) -> List:
        """Clean up message sequence to ensure tool_use blocks have corresponding tool_result blocks.

        Args:
            messages: List of messages to clean

        Returns:
            List of cleaned messages
        """
        cleaned_messages = []
        i = 0

        while i < len(messages):
            current_msg = messages[i]

            # If current message is AI with tool calls
            if (
                current_msg.type == "ai"
                and hasattr(current_msg, "tool_calls")
                and current_msg.tool_calls
            ):

                # Check if next message is a tool result
                if i + 1 < len(messages) and messages[i + 1].type == "tool":
                    # Valid tool_use -> tool_result sequence
                    cleaned_messages.append(current_msg)
                    cleaned_messages.append(messages[i + 1])
                    i += 2
                else:
                    # Incomplete tool sequence - skip the tool_use message
                    print(f"Adding missing tool results for message {i}")
                    cleaned_messages.append(current_msg)
                    for tool_call in current_msg.tool_calls:
                        tool_result = ToolMessage(
                            content=tool_call["args"],
                            tool_call_id=tool_call["id"],
                            name=tool_call["name"],
                        )
                        cleaned_messages.append(tool_result)
                    i += 1
            else:
                # Regular message
                cleaned_messages.append(current_msg)
                i += 1

        return cleaned_messages

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
        # Clean up incomplete tool sequences
        messages = self._clean_message_sequence(messages)

        for msg in messages:
            print("Message type: ", msg.type)
            print(f"Msg: {msg}\n")

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
        response = llm_with_tools.invoke(messages)
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
            "Use tool calls to extract key points and summary from the retrieved documents."
            "do not use any tool if it is not needed."
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
        llm_with_tools = self.llm.bind_tools([PointSchema])
        response = llm_with_tools.invoke(prompt)

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

        if latest_response.tool_calls:
            for tool_call in latest_response.tool_calls:
                print(f"tool_call: {tool_call}")
                if tool_call["name"] == "PointSchema":
                    keypoints = tool_call["args"]["keypoints"]
                    summary = tool_call["args"]["summary"]

                    response = keypoints + "\n\n" + summary
        else:
            response = latest_response.content

        print(f"Total retrieved documents: {len(retrieved_docs)}\n")
        print(f"Retrieved documents: {retrieved_docs}\n")
        print("=" * 50)

        return response, retrieved_docs
