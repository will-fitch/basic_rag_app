from dotenv import load_dotenv
import os
import getpass
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass
class RAGResponse:
    """Response from RAG model with content and source citations."""
    content: str
    sources: List[Dict[str, Any]]
    query: str

class RAGModel:
    """
    A Retrieval-Augmented Generation (RAG) model that combines vector search with LLM responses.
    
    This class provides functionality to:
    - Store and search document chunks in a vector database
    - Generate responses using retrieved context and an LLM
    - Add new content with automatic chunking
    - Provide source citations for responses
    """

    def __init__(self) -> None:
        """
        Initialize the RAG model with vector database, chat model, and prompt template.
        
        Sets up:
        - OpenAI embeddings for vector similarity search
        - Chroma vector database with persistent storage
        - GPT-4o-mini chat model for response generation
        - System prompt template for RAG responses
        """
        # initialize vector database
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_db = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",
        )

        self.vector_db.reset_collection()

        # initialize chat bot model
        self.chat_model = init_chat_model("gpt-4o-mini", model_provider="openai")

        # initialize query prompt template
        self.prompt_template = ChatPromptTemplate([
            ("system", "You are a helpful assistant. Answer questions using only the following information:\n{documents}"),
            MessagesPlaceholder("messages")
        ])

    def query(self, query: str) -> RAGResponse:
        """
        Query the RAG model with a question and return a response with source citations.
        
        Args:
            query (str): The question or query to process
            
        Returns:
            RAGResponse: Contains the LLM response and source citations
            
        Process:
            1. Performs similarity search to find relevant document chunks
            2. Formats retrieved documents as context
            3. Generates response using the LLM with retrieved context
            4. Returns response with source information
        """
        relevant_documents = self.vector_db.similarity_search_with_score(query, k=5)
        
        if(len(relevant_documents) == 0):
            return RAGResponse(
                content="No relevant information found. Please try again with different query.",
                sources=[],
                query=query
            )

        prompt = self.prompt_template.invoke({
            "documents": "\n".join([document[0].page_content for document in relevant_documents]),
            "messages": [HumanMessage(content=query)]
        })

        response = self.chat_model.invoke(prompt)
        
        # Extract source information from documents
        sources = self._extract_sources(relevant_documents)
        
        return RAGResponse(
            content=str(response.content),
            sources=sources,
            query=query
        )

    def _extract_sources(self, documents: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
        """
        Extract source information from retrieved documents.
        
        Args:
            documents: List of documents from similarity search
            
        Returns:
            List of source dictionaries with metadata
        """
        sources = []
        for i, (doc, _) in enumerate(documents):
            source = {
                "id": i + 1,  # Sequential numbering
                "content": doc.page_content,
                "metadata": doc.metadata,
                "filename": doc.metadata.get("filename", "unknown"),
            }
            sources.append(source)
        return sources

    def _add_chunks(self, document_strings: List[str], filename: str = "unknown") -> None:
        """
        Add pre-split document chunks to the vector database with filename metadata.
        Args:
            document_strings (list[str]): List of text chunks to add to the vector database
            filename (str): Name of the source file for metadata
        """
        documents = []
        for content in document_strings:
            metadata = {"filename": filename}
            documents.append(Document(page_content=content, metadata=metadata))
        self.vector_db.add_documents(documents)

    def add_content(self, file_content: str, filename: str = "unknown file", chunk_size: int = 500, chunk_overlap: int = 10) -> int:
        """
        Split file content into chunks and add those chunks to the vector database with source tracking.
        
        Args:
            file_content (str): The text content to split and add
            filename (str): Name of the source file
            chunk_size (int): Maximum size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
            
        Returns:
            int: Number of chunks added to the vector database
        """
        try:
            # Split the text into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_text(file_content)
            
            # Add chunks with source metadata
            self._add_chunks(chunks, filename)
            
            return len(chunks)  # Return number of chunks added
            
        except Exception as e:
            raise Exception(f"Error processing file content: {str(e)}")

    def delete_by_filename(self, filename: str) -> None:
        """
        Delete all documents in the vector database with the given filename as metadata.
        Args:
            filename (str): The filename to match in document metadata.
        """
        self.vector_db.delete(where={"filename": filename})
