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
      cd genai-usecases/llm-providers/python_scripts
      ```   
   2. Open the Project in VS Code or any code editor. 
   3. Create a virtual environment by running the following command in the terminal:
   
      ```bash
      pip install uv #if uv not installed
      uv venv
      .venv\Scripts\activate # On Linux -> source venv/bin/activate
      ```
   4. The `requirements.txt` file contains all necessary dependencies.

   5. Install dependencies:
      
      ```bash
      uv pip install -r requirements.txt
      ```
   6. Configure Environment
      * Rename .env.example → .env
      * **You only need ONE provider key.** Start with a free one:

        ```bash
        # --- Free-tier (no credit card) - start here ---
        GROQ_API_KEY=your_groq_key_here        # Free - https://console.groq.com/keys
        GOOGLE_API_KEY=your_google_key_here     # Free - https://aistudio.google.com/apikey

        # --- Paid (optional) ---
        ANTHROPIC_API_KEY=your_anthropic_key    # Paid - https://console.anthropic.com/settings/keys
        OPENAI_API_KEY=your_openai_key          # Paid - https://platform.openai.com/api-keys
        ```
      > **Recommended for beginners:** Get a free Groq or Google Gemini key. You can add paid providers later.


   7. Run the different LLM API located in `genai-usecases/llm-providers/python_scripts`
   
      * [gemini (free-tier)](./python_scripts/gemini.py)
      
        `streamlit run gemini.py`

      * [groq (free-tier)](./python_scripts/groq_api.py)
      
        `streamlit run groq.py`

      * [Claude (paid)](./python_scripts/claude.py) 

        `streamlit run claude.py`
    
      * [openai (paid)](./python_scripts/openai.py)
      
        `streamlit run openai.py`



