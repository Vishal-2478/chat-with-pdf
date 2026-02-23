import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Modern LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ===============================
# Load environment
# ===============================
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(page_title="Chat with PDFs", page_icon="📄")
st.title("📄 Chat with Multiple PDFs - RAG")

# ===============================
# Initialize Session State FIRST (before anything else)
# ===============================
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "pdf_ready" not in st.session_state:
    st.session_state["pdf_ready"] = False


# ===============================
# Helper Functions
# ===============================
def extract_text_from_pdfs(pdf_files):
    text = []
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text.append(content)
    return "\n".join(text)


def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
    )
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


def get_chat_history():
    """Safely fetch chat history from session state."""
    msgs = st.session_state.get("messages", [])
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs[-6:])


def build_rag_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant. Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I don't know based on the provided documents."

Chat History:
{history}

Context:
{context}

Question: {question}

Answer:
"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "history": lambda _: get_chat_history(),  # safely called at invoke time
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# ===============================
# Sidebar - Upload PDFs
# ===============================
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button("Process PDFs"):
        if not uploaded_files:
            st.error("Please upload at least one PDF.")
        else:
            with st.spinner("Processing PDFs..."):
                text = extract_text_from_pdfs(uploaded_files)

                if not text.strip():
                    st.error("No readable text found in the uploaded PDFs.")
                else:
                    vectorstore = create_vector_store(text)
                    st.session_state["vectorstore"] = vectorstore
                    st.session_state["pdf_ready"] = True
                    st.session_state["messages"] = []  # reset chat
                    st.success("✅ PDFs processed! You can now chat.")

# ===============================
# Block Chat If No PDFs Processed
# ===============================
if not st.session_state["pdf_ready"]:
    st.info("👈 Upload and process your PDFs from the sidebar to start chatting.")
    st.stop()

# ===============================
# Display Chat History
# ===============================
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ===============================
# Chat Input
# ===============================
query = st.chat_input("Ask something about your PDFs")

if query:
    # Save and display user message
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    try:
        retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 4})
        rag_chain = build_rag_chain(retriever)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = rag_chain.invoke(query)
            st.markdown(answer)

        # Save assistant reply
        st.session_state["messages"].append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
