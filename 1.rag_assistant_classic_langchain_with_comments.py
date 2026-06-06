
import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever 
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# suppress chromadb warnings
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.db.duckdb").setLevel(logging.ERROR)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

prompt_template = PromptTemplate(
    template="""You are a precise medical records retrieval assistant.
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
    Structured Answer:""",
    input_variables=["context", "question"],
)

def build_hybrid_rag(pdf_path: str):
    # 1. LOAD PDF - SYNTHETIC DATA FOR DEMO/TESTING ONLY!
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    # print(documents)

    # -----------------------------------------------------------------------
    # WHY SPLIT DOCUMENTS?
    # 1. LLMs have **token limits** for context windows
    # 2. Entire pages are too large to embed efficiently
    # 3. Hard to find **specific** relevant information
    # 4. Semantic search works better on focused content

    # chunk_size = how much text (characters) goes into each chunk.
    # “How big is each puzzle piece?”
    # Larger chunks = better context, fewer pieces
    # Smaller chunks = more precise retrieval, more pieces

    # why overlap?    
    # prevents context loss at boundaries
    # if a sentence is split between chunks, overlap ensures both chunks get full context
    # example: "prescribed metformin 500mg | twice daily" ← overlap prevents breaking this

    # chunk_overlap = how many characters of context spill into the next chunk.
    # “How much glue keeps the puzzle pieces connected?”
    # Overlap prevents the model from losing meaning across boundaries.
    
    # For most RAG systems
    # chunk_size = 500–1000
    # chunk_overlap = 100–300
    
    # result: ~25-30 document objects (chunks)
    # each chunk: ~800 chars
    # each chunk overlaps 200 chars with neighbors
    
    # 800 chars ≈ 150-200 tokens → Perfect for embedding models
    # 200 char overlap = 25% overlap → Good balance (prevents context loss)

    # 2. SPLIT INTO CHUNKS
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    doc_chunks = splitter.split_documents(documents)
    
    # # PRINT THE CHUNKS
    # print(f"\nTotal chunks created: {len(doc_chunks)}\n")
    # for i, doc in enumerate(doc_chunks): 
    #     print(f"\n--- CHUNK #{i+1} ---")
    #     print(f"Page: {doc.metadata.get('page')}")
    #     print(f"Length: {len(doc.page_content)} characters")
    #     print(f"\n{doc.page_content}\n")
    #     print("="*60)
    # exit()

    # ---------------------------------------------------------------
    
    # WHAT ARE EMBEDDINGS?
    # Embeddings are numerical representations (vectors) of text that capture semantic meaning.
    # Example
    # Text: "Hassan Kim has diabetes"
    # ↓
    # Embedding: [0.23, -0.45, 0.89, 0.12, ..., 0.67]  # 1536 numbers for OpenAI
    
    # THE EMBEDDINGS MODEL IS CREATED:
    # 1. takes text as input
    # 2. sends it to openai's api
    # 3. returns a vector (array of 1,536 numbers for text-embedding-3-small)
    # 4. these numbers encode the semantic meaning
    
    # SEMANTIC RETRIEVAL = finding the most semantically similar chunks using embeddings.
    # Measuring similarity using cosine similarity, dot product, etc
    # It retrieves information based on meaning, not matching words.

    # COSINE SIMILARITY = text is converted into embeddings (vectors), each sentence becomes a point in 
    # high-dimensional space. measures angle between vectors:
    # Do these two vectors point in nearly the same direction?
    # If yes → high similarity
    # If no → low similarity
    
    # 3. CREATE EMBEDDING MODEL
    embeddings = OpenAIEmbeddings()
    
    # 3.1 USING FAISS - FACEBOOK AI SIMILARITY SEARCH
    # It is a high-performance vector search library used to quickly find the most similar embeddings
    # (vectors) in very large datasets
    # vectorstore = FAISS.from_documents(docs, embeddings)
    
    # 4. USING CHROMA - EMBED ALL CHUNKS AND STORE
    vectorstore = Chroma.from_documents(doc_chunks, embeddings)
    # internally:
    # for chunk in doc_chunks:
    #     vector = embeddings.embed_query(chunk.text)
    #     store(chunk.text, vector)
    
    # WHEN SHOULD YOU USE FAISS VS CHROMA?
    # For your healthcare RAG use-cases:
    # Use FAISS when:
    # You want maximum speed
    # Dataset size is < ~5 million chunks
    # You do NOT need long-term persistence
    # You are deploying locally or on GPU servers
    # Use Chroma when:
    # You want persistent storage on disk
    # You want METADATA FILTERING (IDS, PATIENT NUMBERS, FILE NAMES, DATES, ETC.))
    # You are deploying in production with updates
    # You want better observability
    
    # SIMPLE ANALOGY:
    # FAISS = a very fast search engine
    # Chroma = a full database + search engine built for RAG

    # 5. CREATE SEMANTIC RETRIEVER
    semantic_retriever = vectorstore.as_retriever(
        search_type="similarity", # use cosine similarity
        search_kwargs={"k": 3}   #return top 3 most similar
    )

    # A keyword retriever returns documents based on literal keyword overlap between the user’s query and the documents.
    # “Find me documents that contain the same words I typed.”
    
    # 6. CREATE KEYWORD RETRIEVER (BM25)
    keyword_retriever = BM25Retriever.from_documents(doc_chunks)
    keyword_retriever.k = 3

    # A hybrid retriever (semantic + keyword search) will improve accuracy, especially with structured records like PDFs where names, 
    # medications, or lab values might not embed well semantically.
    # In LangChain, we can combine:
    # FAISS (vector search) → semantic similarity search
    # BM25 (keyword search) → exact word matching
    # and then merge their results.

    # 7. COMBINE THEM INTO A HYBRID RETRIEVER
    retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, keyword_retriever],
    weights=[0.6, 0.4] # semantic prioritized, but keywords matter
    )

    # temperature controls randomness in the model's output:
    # Low values → more focused, factual, reproducible
    # High values → more creative, varied, less predictable
    # for fact-based rag assistant recommended:0.0 – 0.3, ensures precise, stable answers from retrieved context
    
    # 8. CREATE THE QA CHAIN
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'), temperature=0.0),
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )
    return qa_chain

if __name__ == "__main__":
    print("\n--- Classic RAG Assistant ---")
    pdf_path = os.getenv('PDF_PATH_FILE')
    qa_chain = build_hybrid_rag(pdf_path)

    # question = "1. What medications is Hassan Kim currently prescribed?" 
    # question = "2. What allergies does Hassan have?" 
    question = "3. What is Hassan's blood pressure?" 
    response = qa_chain.invoke({"query": question}) #must be a dictionary object

    # WHAT DOES INVOKE METHOD? - magic line that runs your entire RAG assistant
    # 1. Embed question                  
    #    → [0.234, -0.456, ...]      
    #   
    # 2. Search vectorstore (Chroma)       
    #   → Find top 3 similar chunks
    #
    # 3. Extract chunk text 
    # PROTECTED HEALTH INFORMATION (PHI) USED FOR TESTING ONLY!
    # Use HuggingFaceEmbeddings and LLM llama3.3:70b or MEDITRON_70B locally)
    #    → "Medications: Metformin..."       
    #    → "Hassan Kim, Age 45..."           
    #    → "Recent prescription..."
    #
    # 4. Format prompt                      
    #   → Insert chunks into template 
    #
    # 5. Send to LLM (GPT-4o-mini)         
    #   → Generate answer        
    #                                     
    # 6. Return response    

    print("Question:", response["query"])
    print("Answer:", response["result"])
    print(f"Sources: {len(response['source_documents'])} documents")
    
    # show document sources
    source_documents = response["source_documents"]
    unique_sources = {}
    for doc in source_documents:
        source_path = doc.metadata.get("source", "Unknown")
        if source_path not in unique_sources:
            unique_sources[source_path] = []
        unique_sources[source_path].append(doc.metadata.get("page", "N/A"))
    print(f"Relevant Sources ({len(unique_sources)} unique file(s) found):")
    for source, pages in unique_sources.items():
        pages_str = ", ".join(map(str, sorted(set(pages))))
        print(f"- {source} (Pages: {pages_str})")


