# 🩺 Med-RAG: AI-Powered Clinical Assistant

**Evidence-based answers for clinical questions.**  
*A Retrieval-Augmented Generation (RAG) pipeline that allows users to chat with a dataset of medical transcriptions using a quantized local LLM.*

---

## 🧐 What is this?

This project implements a **RAG (Retrieval-Augmented Generation)** system designed to answer medical queries based *only* on a provided knowledge base of clinical transcriptions.

Unlike standard ChatGPT, which answers from general training data, **Med-RAG** retrieves specific patient samples and surgical reports to ground its answers in reality. It uses **Mistral-7B-Instruct (Quantized)** to run efficiently on consumer hardware (like a T4 GPU).

---

## ⚙️ System Architecture

The pipeline is built using the **LangChain** ecosystem:

1. **Ingestion:** Loads 600+ rows from `mtsamples.csv` (Medical Specialties, Sample Names, Transcriptions).
2. **Chunking:** Splits long medical reports into manageable 512-character chunks using `RecursiveCharacterTextSplitter`.
3. **Embedding:** Converts text into vector space using Hugging Face's `all-MiniLM-L6-v2`.
4. **Storage:** Indexes vectors in **ChromaDB** for millisecond-latency retrieval.
5. **Generation:**
   * **Retrieval:** Fetches the top 3 most relevant clinical documents for a user query.
   * **Inference:** Feeds the context + query into **Mistral-7B (4-bit GGUF)** to generate a concise, evidence-based answer.

---

## ⚡ Quick Start

**1. Install Dependencies**

```bash
pip install langchain langchain-community langchain-huggingface chromadb pandas ctransformers sentence-transformers
```

**2. Download the Model**  
The script automatically fetches the GGUF model:

* **Model:** `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`
* **File:** `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (approx 4.37GB)

**3. Run the Pipeline**

```bash
python med_rag.py
```

*This will process the data, build the vector DB, and generate answers for the test suite.*

---

## 📊 Test Results

The system was evaluated on **32 Queries** across 3 categories. The results are saved to `rag_results.csv`.

| Category | Query Type | Goal |
| --- | --- | --- |
| **General Medicine** | "Symptoms of allergic rhinitis?" | Retrieve accurate symptom lists from patient history. |
| **Surgery** | "Procedure for cataract surgery?" | Summarize operative reports and techniques. |
| **Negative Controls** | "How to bake a cake?" | **Refusal.** The model should state "I do not know" as this is outside the medical context. |

---

## 🧠 Under the Hood

**Code Snippet: The Retrieval Chain**

```python
# We use a standard Stuff Documents Chain to feed context to Mistral
system_prompt = (
    "You are a medical assistant. Answer based ONLY on the context provided. "
    "If the answer is missing, say 'I do not know'.\n\nContext:\n{context}"
)

rag_chain = create_retrieval_chain(
    retriever, 
    create_stuff_documents_chain(llm, prompt)
)
```

**Hardware Usage:**

* **GPU:** Tesla T4 (required for reasonably fast inference with 40 GPU layers offloaded).
* **RAM:** ~8GB System RAM + ~6GB VRAM.

---

## 📜 Credits

* **Dataset:** Medical Transcriptions (Kaggle)
* **LLM:** Mistral AI via TheBloke (Quantized)
* **Framework:** LangChain v0.2

---

**🚑 Trust, but Verify.**  
*This tool is for educational purposes. Always consult a real medical professional.*
