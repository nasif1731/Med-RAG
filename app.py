import os
import pandas as pd
import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import CTransformers
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURATION ---
CSV_PATH = "mtsamples.csv"
DB_PATH = "chroma_db_local"

# CPU Configuration
# context_length: How much text the AI can read at once. 2048 is safe for most laptops.
config = {
    'max_new_tokens': 256,
    'temperature': 0.1,
    'context_length': 2048,
    'gpu_layers': 0  # CRITICAL: 0 means run entirely on CPU
}

print("--- STARTING LOCAL CPU RAG ---")

# 1. SETUP EMBEDDINGS
print("Loading Embedding Model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. LOAD OR CREATE DATABASE
# This checks if you've already built the database to save time.
if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
    print("Found existing vector database. Loading...")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
else:
    print("No database found. Building from CSV (This happens only once)...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Could not find {CSV_PATH}. Please make sure it is in the folder.")
    
    # Load Data
    df = pd.read_csv(CSV_PATH).dropna(subset=['transcription']).head(500) # Limit to 500 for speed on laptop
    
    documents = []
    for i, row in df.iterrows():
        content = f"Medical Specialty: {row['medical_specialty']}\nSample: {row['sample_name']}\nContent: {row['transcription']}"
        documents.append(Document(page_content=content, metadata={"source": str(row['sample_name'])}))
    
    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Create DB
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=DB_PATH)
    print("Database built and saved locally.")

retriever = vectorstore.as_retriever(search_kwargs={'k': 2}) # Retrieve top 2 matches to save RAM

# 3. LOAD LLM (The Mistral Model)
print("Loading Mistral-7B Model (Downloads ~4GB on first run)...")
llm = CTransformers(
    model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    config=config
)

# 4. CREATE CHAIN
system_prompt = (
    "You are a helpful medical assistant. Use the provided context to answer the question. "
    "If the answer is not in the context, say 'I do not know based on the documents'.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# 5. DEFINE GRADIO INTERFACE
def ask_doctor(query):
    if not query: return "Please ask a question."
    print(f"Processing query: {query}")
    try:
        response = rag_chain.invoke({"input": query})
        return response["answer"]
    except Exception as e:
        return f"Error occurred: {str(e)}"

# Custom UI Layout
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🏥 Local Medical AI Assistant")
    gr.Markdown("Running offline on your laptop CPU.")
    
    with gr.Row():
        txt_input = gr.Textbox(label="Your Question", placeholder="e.g. Symptoms of allergic rhinitis?")
        txt_output = gr.Textbox(label="AI Answer", lines=5)
        
    btn_submit = gr.Button("Submit Question")
    
    btn_submit.click(ask_doctor, inputs=txt_input, outputs=txt_output)

if __name__ == "__main__":
    app.launch()