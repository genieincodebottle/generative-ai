## 🧠  Run & Experiment with LLMs  

Explore and run different LLM APIs (OpenAI, Google, Anthropic, Groq) as well as local LLMs (Ollama, HuggingFace).

---

## Options to Run

### A. Run each LLM API on Google Colab

Easily try out LLM providers using given Colab notebooks:

👉 [Colab Notebooks](notebooks/)

---

### B. Run Local LLMs

Set up and run models locally with **Ollama** and **HuggingFace**:

👉 [Ollama](./local_llms/ollama/)

👉 [HuggingFace](./local_llms/huggingface/)

---

### C. Run each LLM API with UI

You can run each LLM provider independently using dedicated Streamlit UI scripts available in the `python_scripts` folder.

### 🛠️ Setup Instructions

#### ✅ Prerequisites
   - Python 3.10 or higher
   - pip (Python package installer)

#### 📦 Installation & Running App
   1. Clone the repository:

      ```bash
      git clone https://github.com/genieincodebottle/generative-ai.git
      cd genai-usecases\llm-providers\python_scripts
      ```   
   2. Open the Project in VS Code or any code editor. 
   3. Create a virtual environment by running the following command in the terminal:
   
      ```bash
      pip install uv #if uv not installed
      uv venv
      .venv\Scripts\activate # On Linux -> source venv/bin/activate
      ```
   4. Create a requirements.txt file and add the following libraries:
      
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


   7. Run the different LLM API located in ```genai-usecases\llm-providers\python_scripts```
   
      * [gemini (free-tier)](./python_scripts/gemini.py)
      
        `streamlit run gemini.py`

      * [groq (free-tier)](./python_scripts/groq_api.py)
      
        `streamlit run groq.py`

      * [Claude (paid)](./python_scripts/claude.py) 

        `streamlit run claude.py`
    
      * [openai (paid)](./python_scripts/openai.py)
      
        `streamlit run openai.py`



