## 📚 Advance RAG

![alt text](images/rag.gif)

---
<strong>**Retrieval-Augmented Generation (RAG)**</strong> enhances Large Language Models (LLMs) by combining them with an external knowledge retrieval system.

When you send a query:

1. The retrieval system fetches the most relevant data from a knowledge base.

2. The LLM uses both the query and the retrieved data to generate a more accurate, context-aware response.

This makes RAG essential for building reliable GenAI applications. As queries and data grow in complexity, advanced RAG techniques, like agentic RAG, Graph RAG, Corrective RAG, Reranking RAG etc further improve accuracy, adaptability & relevance.

---
## 🔑 Core Components of RAG

   - <strong>**Knowledge Base**</strong>: External data source (e.g. documents, databases) that the system relies on.

   - <strong>**Retrieval System**</strong>:

      - <strong>**Vector Database**</strong> for storing and searching embeddings.

      - <strong>**Embedding Model**</strong> to convert queries and documents into vector representations.

   - <strong>**Language Model (LLM)**</strong>: Generates responses using both retrieved data and the query.

---
## 📚 Super Handy Resources 

| Resource | Link | Description |
|----------|------|-------------|
| **Embedding Models • Vector Stores • Vector Embeddings (Guide)** | [PDF](https://github.com/genieincodebottle/generative-ai/blob/main/docs/vector-embeddings-guide.pdf) | Guide covering embedding models, vector stores, and embeddings explained with examples |
| **RAG Decision Flow** | [PDF](./docs/advance-rag-decision-flow-chart.pdf) | Quick reference flowchart to select the right **RAG technique** for your use case |
| **Advanced Snapshot-Based PDF Parsing** | [GitHub](https://github.com/genieincodebottle/parsemypdf) | End-to-end parsing using **Docling, Markitdown, Gemini, Llama 4, Claude, GPT-4 & more** |
| **Uber's Usecase -> Enhanced Agentic-RAG: What If Chatbots Could Deliver Near-Human Precision?** | [Blog](https://lnkd.in/eGz5a9xm) | Genie is Uber’s internal on-call copilot in Slack, delivering real-time, cited answers from internal docs and boosting on-call engineers’ and SMEs’ productivity by handling common queries efficiently. |

---
## 🧪 Exploring Advanced RAG

<strong>A. Try Graph RAG</strong>

👉 [Graph RAG](graph-rag/)

---
<strong>B. Try Agentic RAG</strong>

👉 [Agentic RAG](agentic-rag/)

---
<strong>C. Try Multimodal RAG</strong>

👉 [Multimodal RAG](multimodal-rag/)

---
<strong>D. Try Advanced RAG Techniques in Google Colab</strong>

👉 [Notebooks](notebooks/)

   - [Basic RAG](notebooks/basic-rag.ipynb)
   - [Corrective RAG](notebooks/corrective-rag.ipynb)
   - [Re-ranking RAG](notebooks/re_ranking_rag.ipynb)
   - [Hybrid Search RAG](notebooks/hybrid-search-rag.ipynb)
   - [Hypothetical Doucment Embedding RAG](notebooks/hypothetical-document-embedding-rag.ipynb)
   - [Multi-index RAG](notebooks/multi-index-rag.ipynb)
   - [Query Expansion RAG](notebooks/query-expansion-rag.ipynb)
   - [Adaptive RAG](notebooks/adaptive-rag.ipynb)
   - [Self Adaptive RAG](notebooks/self-adaptive-rag.ipynb)

---
<strong>E. Try Advanced RAG Techniques in Streamlit UI</strong>

<strong>🛠️ Setup Instructions</strong>

<strong>✅ Prerequisites</strong>
   - Python 3.10 or higher
   - pip (Python package installer)

<strong>📦 Installation & Running App</strong>
   1. Clone the repository:

      ```bash
      git clone https://github.com/genieincodebottle/generative-ai.git
      cd genai-usecases\advance-rag\rag_techniques
      ```
   2. Open the Project in VS Code or any code editor.
   3. Create a virtual environment by running the following command in the terminal:
   
      ```bash
      pip install uv #if uv not installed
      uv venv
      .venv\Scripts\activate # On Linux -> source venv/bin/activate
      ```
   4. Create a `requirements.txt` file and add the following libraries:
      
      ```bash
        streamlit>=1.47.1 
        langchain>=0.3.27 
        langchain-google-genai>=2.1.8 
        langchain-chroma>=0.2.5 
        langchain-community>=0.3.27
        nest-asyncio>=1.6.0
        pypdf>=5.9.0
        python-dotenv>=1.0.1
        flashrank>=0.2.10
        rank_bm25>=0.2.2
      ```
   5. Install dependencies:
      
      ```bash
      uv pip install -r requirements.txt
      ```
   6. Configure Environment
      * Rename .env.example → .env
      * Update with your keys:

         ```bash
         GOOGLE_API_KEY=your_key_here # Using the free-tier API Key
         ```
      * Get **GOOGLE_API_KEY** here -> https://aistudio.google.com/app/apikey

   9. Run RAG implementations in ```genai-usecases\advance-rag\rag_techniques```
   
      * [basic_rag](./rag_techniques/basic_rag.py) 

        `streamlit run basic_rag.py`
    
      * [adaptive_rag](./rag_techniques/adaptive_rag.py)
      
        `streamlit run adaptive_rag.py`

      * [corrective_rag](./rag_techniques/corrective_rag.py)
      
        `streamlit run corrective_rag.py`

      * [re_ranking_rag](./rag_techniques/re_ranking_rag.py)
      
        `streamlit run re_ranking_rag.py`

      * [hybrid_search_rag](./rag_techniques/hybrid_search_rag.py)
      
        `streamlit run hybrid_search_rag.py`

---
<strong>F. Local LLM Powered Multi-Function App (RAG Included)</strong>

[Local LLM RAG App Powered by Llama](../llama-4-multi-function-app/)

---
<strong>G. RAG with MCP Server (Fullstack Agentic RAG on AWS Cloud)</strong>

[RAG Application with AWS & MCP Server Integration](https://github.com/genieincodebottle/rag-app-on-aws)
