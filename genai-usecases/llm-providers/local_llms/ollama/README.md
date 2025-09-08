## Ollama: Simple Guide to Download, Pull Models, and Run

Ollama is a lightweight tool that lets you run large language models (LLMs) locally on your computer. This guide will help you install Ollama, download models & run them easily.

### 🔹 Installation

#### 🪟 Windows
1. Download the Windows installer from [ollama.ai](https://ollama.ai)
2. Run the installer and follow the prompts
3. Ollama will be installed and should start automatically

#### 🐧Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### 🍏 macOS
1. Download the installer from [ollama.ai](https://ollama.ai)
2. Open the downloaded file and follow the installation prompts
3. Ollama will be installed and started automatically

---

### 🔹 Pulling Models

Once installed, you can pull (download) models using the following command:

```bash
ollama pull <modelname>
```

Popular models include:
* `deepseek-r1:7b`
* `deepseek-r1:8b`
* `deepseek-r1:70b`
* `llama3.1:8b`
* `llama3.2`
* `phi4`

Full list of models can be found here -> https://ollama.com/search

For specific model versions, use:
```bash
ollama pull llama3.1:8b
ollama pull deepseek-r1:8b
```
---

### 🔹 Running Models

  #### ▶️ Command Line Interface
  Run a model in the terminal:
  ```bash
  ollama run llallama3.1:8b
  ```

  This will start an interactive chat session. Type your prompt and press Enter.

  #### ▶️ API Usage
  Ollama also provides a local API that you can use:

  ```bash
  # Start the Ollama service if not already running
  ollama serve
  ```

  Then you can query models using curl:
  ```bash
  curl -X POST http://localhost:11434/api/generate -d '{
    "model": "llama3.1:8b",
    "prompt": "Explain quantum computing in simple terms"
  }'
  ```

  #### ▶️ Custom Parameters
  You can customize model parameters:

  ```bash
  ollama run llama3.1:8b --temperature 0.7 --top_p 0.9
  ```
---

### 🔹 Managing Models

  List downloaded models:
  ```bash
  ollama list
  ```

  Remove a model:
  ```bash
  ollama rm <modelname>
  ```
---

### 🔹 Example: Run Ollama App

   1. Clone the repository:

      ```bash
      git clone https://github.com/genieincodebottle/generative-ai.git
      cd genai-usecases\llm-providers\local_llms\ollama
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
        langchain-ollama
      ```
   5. Install dependencies:
      
      ```bash
      uv pip install -r requirements.txt
      ```
   6. Run the ollama based local LLM app
   
       ```bash
      python ollama_example.py
      ```
---
### 🔹 Learn More

* Official documentation: [Ollama Documentation](https://github.com/ollama/ollama/blob/main/README.md)
* Check GitHub for the latest updates: [Ollama GitHub](https://github.com/ollama/ollama)
