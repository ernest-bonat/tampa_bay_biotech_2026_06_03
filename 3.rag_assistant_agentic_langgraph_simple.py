import os
import warnings
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# suppress warnings
warnings.filterwarnings("ignore")
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

@tool
def patient_records_search(query: str):
    """Look up patient information from medical PDF records."""
    # We'll initialize the retriever globally or pass it in. 
    # For this script, we'll assume it's accessible.
    docs = hybrid_retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

# 6. define graph state & logic
class AgentState(TypedDict):
    # this keeps track of the conversation history
    messages: Annotated[Sequence[BaseMessage], add_messages]

def medical_agent(state: AgentState):
    """The node that decides what to do next."""
    system_prompt = SystemMessage(content=(
        "You are a medical assistant. Answer the user's question using only the "
        "patient records provided. Do NOT repeat long chunks of text. "
        "If the answer is not in the records, say: 'Not found in the records.'"
    ))
    # concatenate system prompt with current message history
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    """determines if the agent should call a tool or finish."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def get_hybrid_retriever(pdf_path: str):
    # 1.
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # 2. 
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # 3.
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings)
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4.
    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = 3

    # 5.
    return EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=[0.6, 0.4]
    )

# 5. define tools
tools = [patient_records_search]
tool_node = ToolNode(tools)

# 6. define the llm with tool-calling capabilities
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2).bind_tools(tools)

# 7. build the graph 
workflow = StateGraph(AgentState)
# add nodes
workflow.add_node("agent", medical_agent)
workflow.add_node("tools", tool_node)
# set entry point
workflow.set_entry_point("agent")
# add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
)
# add edge from tools back to agent to process the tool output
workflow.add_edge("tools", "agent")
# compile the graph
app = workflow.compile()

if __name__ == "__main__":
    print("\n--- LangGraph Agentic RAG Assistant ---")
    pdf_path = os.getenv('PDF_PATH_FILE')
    
    if not pdf_path:
        print("Error: PDF_PATH_FILE not found in environment variables.")
    else:
        # initialize the global retriever used by the tool
        hybrid_retriever = get_hybrid_retriever(pdf_path)

        question = "1. What medications is Hassan Kim currently prescribed?"        
        # question = "2. What allergies does Hassan have?" 
        # question = "3. What is Hassan's blood pressure?" 
        
        # stream the output
        inputs = {"messages": [HumanMessage(content=question)]}
        result = app.invoke(inputs)

        # the last message in the state is the final answer
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['messages'][-1].content}")
        
    source_documents = hybrid_retriever.get_relevant_documents(question)    
    print(f"\nRelevant Sources ({len(source_documents)} documents found):")
    for i, doc in enumerate(source_documents, 1):
        source_file = doc.metadata.get("source", "Unknown")
        page_num = doc.metadata.get("page", "Unknown")
        # extract filename from full path
        if "\\" in source_file:
            filename = source_file.split("\\")[-1]
        elif "/" in source_file:
            filename = source_file.split("/")[-1]
        else:
            filename = source_file
        print(f"{i}. File: {filename}")
        print(f"   Page: {page_num}")
        print(f"   Content preview: {doc.page_content[:150]}...")