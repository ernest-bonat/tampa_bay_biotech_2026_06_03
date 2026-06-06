
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    chunk_size: int = 800
    chunk_overlap: int = 200
    retrieval_k: int = 3
    semantic_weight: float = 0.6
    keyword_weight: float = 0.4
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2

def setup_logging() -> logging.Logger:
    """
    Configure logging for the application.
    Returns:
        Configured logger instance
    """
    # Suppress third-party warnings
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    logging.getLogger("chromadb.db.duckdb").setLevel(logging.ERROR)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Configure main logger
    logger = logging.getLogger("HybridRAG")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


class RAGException(Exception):
    """Base exception for RAG system."""
    pass

class DocumentLoadError(RAGException):
    """Raised when document loading fails."""
    pass

class ChainBuildError(RAGException):
    """Raised when chain building fails."""
    pass

class QueryError(RAGException):
    """Raised when query execution fails."""
    pass

def get_medical_prompt_template() -> PromptTemplate:
    """
    Get prompt template for medical records retrieval.
    Returns:
        Configured prompt template
    """
    template = """You are a precise medical records retrieval assistant.

    ANSWER PROTOCOL:
    1. Provide a direct answer in one clear sentence
    2. Support with specific evidence (dates, dosages, values) from the records
    3. If any detail is uncertain or absent, state: "Not found in the records"

    CONSTRAINTS:
    - Use ONLY information explicitly stated in the context
    - Never extrapolate or use external medical knowledge
    - For ambiguous queries, ask for clarification rather than guessing
    - Always specify patient name when multiple patients exist

    Patient Records Context:
    {context}

    Question: {question}

    Structured Answer:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def load_pdf(pdf_path: str, logger: logging.Logger) -> List[Document]:
    """
    Load PDF document and extract pages.
    Args:
        pdf_path: Path to PDF file
        logger: Logger instance
    Returns:
        List of Document objects (one per page)
    Raises:
        DocumentLoadError: If PDF loading fails
    """
    try:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not path.suffix.lower() == '.pdf':
            raise ValueError(f"File must be PDF, got: {path.suffix}")
        logger.info(f"Loading PDF: {path.name}")
        loader = PyPDFLoader(str(path))
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} pages")
        return documents
    except Exception as e:
        logger.error(f"Failed to load PDF: {str(e)}")
        raise DocumentLoadError(f"PDF loading failed: {str(e)}") from e


def chunk_documents(
    documents: List[Document],
    config: RAGConfig,
    logger: logging.Logger
) -> List[Document]:
    """
    Split documents into smaller chunks.
    Args:
        documents: List of documents to chunk
        config: RAG configuration
        logger: Logger instance
    Returns:
        List of chunked documents
    Raises:
        DocumentLoadError: If chunking fails
    """
    try:
        logger.info(f"Chunking {len(documents)} documents...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        # Log statistics
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunks else 0
        logger.info(
            f"Created {len(chunks)} chunks | "
            f"Avg: {avg_length:.0f} chars | "
            f"Min: {min(chunk_lengths) if chunks else 0} | "
            f"Max: {max(chunk_lengths) if chunks else 0}"
        )
        return chunks
    except Exception as e:
        logger.error(f"Chunking failed: {str(e)}")
        raise DocumentLoadError(f"Document chunking failed: {str(e)}") from e

def create_semantic_retriever(
    chunks: List[Document],
    config: RAGConfig,
    logger: logging.Logger
) -> Any:
    """
    Create semantic (vector-based) retriever.
    Args:
        chunks: List of document chunks
        config: RAG configuration
        logger: Logger instance
    Returns:
        Configured semantic retriever
    Raises:
        ChainBuildError: If retriever creation fails
    """
    try:
        logger.info("Creating semantic retriever (OpenAI embeddings)...")
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.retrieval_k}
        )
        logger.info("✓ Semantic retriever created")
        return retriever
    except Exception as e:
        logger.error(f"Semantic retriever creation failed: {str(e)}")
        raise ChainBuildError(f"Failed to create semantic retriever: {str(e)}") from e
    
def create_keyword_retriever(
    chunks: List[Document],
    config: RAGConfig,
    logger: logging.Logger
) -> BM25Retriever:
    """
    Create keyword (BM25-based) retriever.
    Args:
        chunks: List of document chunks
        config: RAG configuration
        logger: Logger instance
    Returns:
        Configured keyword retriever
    Raises:
        ChainBuildError: If retriever creation fails
    """
    try:
        logger.info("Creating keyword retriever (BM25)...")
        keyword_retriever = BM25Retriever.from_documents(chunks)
        keyword_retriever.k = config.retrieval_k
        logger.info("✓ Keyword retriever created")
        return keyword_retriever
    except Exception as e:
        logger.error(f"Keyword retriever creation failed: {str(e)}")
        raise ChainBuildError(f"Failed to create keyword retriever: {str(e)}") from e
    
def create_hybrid_retriever(
    semantic_retriever: Any,
    keyword_retriever: BM25Retriever,
    config: RAGConfig,
    logger: logging.Logger
) -> EnsembleRetriever:
    """
    Create hybrid retriever combining semantic and keyword search.
    Args:
        semantic_retriever: Vector-based retriever
        keyword_retriever: BM25-based retriever
        config: RAG configuration
        logger: Logger instance
    Returns:
        Configured ensemble retriever
    Raises:
        ChainBuildError: If hybrid retriever creation fails
    """
    try:
        logger.info(
            f"Creating hybrid retriever "
            f"(semantic: {config.semantic_weight}, keyword: {config.keyword_weight})..."
        )
        retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, keyword_retriever],
            weights=[config.semantic_weight, config.keyword_weight]
        )
        logger.info("✓ Hybrid retriever created")
        return retriever
    except Exception as e:
        logger.error(f"Hybrid retriever creation failed: {str(e)}")
        raise ChainBuildError(f"Failed to create hybrid retriever: {str(e)}") from e


def build_hybrid_rag(
    pdf_path: str,
    config: RAGConfig,
    logger: logging.Logger
) -> RetrievalQA:
    """
    Build hybrid RAG system with semantic and keyword search.
    Args:
        pdf_path: Path to PDF file
        config: RAG configuration
        logger: Logger instance
    Returns:
        Configured RetrievalQA chain
    Raises:
        ChainBuildError: If chain building fails
    """
    try:
        logger.info("=" * 70)
        logger.info("Building Hybrid RAG System")
        logger.info("=" * 70)
        # Step 1: Load PDF
        documents = load_pdf(pdf_path, logger)
        # Step 2: Chunk documents
        chunks = chunk_documents(documents, config, logger)
        # Step 3: Create semantic retriever
        semantic_retriever = create_semantic_retriever(chunks, config, logger)
        # Step 4: Create keyword retriever
        keyword_retriever = create_keyword_retriever(chunks, config, logger)
        # Step 5: Create hybrid retriever
        hybrid_retriever = create_hybrid_retriever(
            semantic_retriever,
            keyword_retriever,
            config,
            logger
        )
        # Step 6: Create LLM
        logger.info(f"Initializing LLM: {config.llm_model}...")
        llm = ChatOpenAI(
            model=config.llm_model,
            api_key=os.getenv('OPENAI_API_KEY'),
            temperature=config.llm_temperature
        )
        # Step 7: Get prompt template
        prompt_template = get_medical_prompt_template()
        # Step 8: Build QA chain
        logger.info("Building RetrievalQA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=hybrid_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        logger.info("✓ RAG system built successfully")
        logger.info("=" * 70)
        return qa_chain
    except Exception as e:
        logger.error(f"Failed to build RAG chain: {str(e)}")
        raise ChainBuildError(f"RAG chain building failed: {str(e)}") from e


def execute_query(
    qa_chain: RetrievalQA,
    question: str,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Execute query against RAG system.
    Args:
        qa_chain: Configured RetrievalQA chain
        question: User question
        logger: Logger instance
    Returns:
        Dictionary containing query, answer, and source documents
    Raises:
        QueryError: If query execution fails
    """
    try:
        logger.info(f"Query: {question}")
        response = qa_chain.invoke({"query": question})
        logger.info("✓ Query executed successfully")
        logger.debug(f"Answer: {response['result'][:100]}...")
        return response
    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        raise QueryError(f"Failed to execute query: {str(e)}") from e

def format_sources(source_documents: List[Document]) -> Tuple[Dict[str, List[int]], int]:
    """
    Format and deduplicate source documents.
    Args:
        source_documents: List of source documents
    Returns:
        Tuple of (unique sources dict, total unique files count)
    """
    unique_sources = {}
    for doc in source_documents:
        source_path = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        if source_path not in unique_sources:
            unique_sources[source_path] = []
        unique_sources[source_path].append(page)
    return unique_sources, len(unique_sources)


def print_sources(unique_sources: Dict[str, List[int]], logger: logging.Logger) -> None:
    """
    Print formatted source information.
    Args:
        unique_sources: Dictionary mapping source paths to page numbers
        logger: Logger instance
    """
    logger.info(f"Relevant Sources ({len(unique_sources)} unique file(s) found):")
    for source, pages in unique_sources.items():
        # Deduplicate and sort pages
        pages_str = ", ".join(map(str, sorted(set(pages))))
        source_name = Path(source).name if source != "Unknown" else "Unknown"
        logger.info(f"  - {source_name} (Pages: {pages_str})")

def main() -> None:
    """Main execution function."""
    # Load environment variables
    load_dotenv()
    # Setup logging
    logger = setup_logging()
    try:
        logger.info("\n Classic RAG Assistant (Semantic + Keyword Search)")
        logger.info("=" * 70)
        # Configuration
        config = RAGConfig(
            chunk_size=800,
            chunk_overlap=200,
            retrieval_k=3,
            semantic_weight=0.6,
            keyword_weight=0.4,
            llm_model="gpt-4o-mini",
            llm_temperature=0.2
        )
        # Get PDF path
        pdf_path = os.getenv('PDF_PATH_FILE')
        if not pdf_path:
            raise ValueError("PDF_PATH_FILE environment variable not set")
        # Build RAG system
        qa_chain = build_hybrid_rag(pdf_path, config, logger)
        # Test questions
        questions = [
            "What medications is Hassan Kim currently prescribed?",
            "What allergies does Hassan have?",
            "What is Hassan's blood pressure?"
        ]
        # Execute queries
        for i, question in enumerate(questions, 1):
            logger.info(f"\n{'=' * 70}")
            logger.info(f"QUERY {i}/{len(questions)}")
            logger.info("=" * 70)
            response = execute_query(qa_chain, question, logger)
            # Print results
            logger.info(f"Question: {response['query']}")
            logger.info(f"Answer: {response['result']}")
            logger.info(f"Sources: {len(response['source_documents'])} documents retrieved")
            # Format and print sources
            unique_sources, count = format_sources(response['source_documents'])
            print_sources(unique_sources, logger)
        logger.info("\n" + "=" * 70)
        logger.info("✓ All queries completed successfully")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()