## 🧠 LLM Providers

Explore how to use different LLM APIs and local LLMs.

### A. Run on Google Colab

Use the following notebooks to run LLM provider APIs on Google Colab:

👉 [Colab Notebooks](notebooks/)

### B. Run Local LLMs

To run local LLMs using Ollama or HuggingFace, follow the steps in:

👉 [Local LLM Guide](./local_llms/)

### C. Run Individual LLM APIs

Use the guide below to set up and run each LLM provider API individually.

### 🛠️ Setup Instructions

#### ✅ Prerequisites
   - Python 3.10 or higher
   - pip (Python package installer)

#### 📦 Installation & Running App
   1. Clone the repository:

      ```bash
      git clone https://github.com/genieincodebottle/generative-ai.git
      cd genai-usecases\llm-providers
      
      ```bash
        streamlit>=1.47.1
        langchain-anthropic>=0.3.18
        langchain-google-genai>=2.1.8
        langchain-groq>=0.3.6
        python-dotenv>=1.0.1
      ```
   5. Install dependencies:
      
      ```bash
      uv pip install -r requirements.txt
      ```
   6. Configure Environment
      * Rename .env.example → .env
      * Update with your keys:

        ```bash
        ANTHROPIC_API_KEY=your_anthropic_api_key_here
        GOOGLE_API_KEY=your_google_api_key_here
        OPENAI_API_KEY=your_openai_api_key_here
        GROQ_API_KEY=your_groq_api_key_here
        ```
      * Get your keys here:
        * 🔑 [GROQ_API_KEY](https://console.groq.com/keys)

        * 🔑 [ANTHROPIC_API_KEY](https://console.anthropic.com/settings/keys)

        * 🔑 [OPENAI_API_KEY](https://platform.openai.com/api-keys)

        * 🔑 [GOOGLE_API_KEY](https://aistudio.google.com/apikey)


   9. Run the different RAG implementations located in ```genai-usecases\advance-rag```
   
      * [Claude](./claude.py) 

        `streamlit run claude.py`
    
      * [gemini](./gemini.py)
      
        `streamlit run gemini.py`

      * [openai](./openai.py)
      
        `streamlit run openai.py`

      * [groq](./groq_api.py)
      
        `streamlit run groq.py`

