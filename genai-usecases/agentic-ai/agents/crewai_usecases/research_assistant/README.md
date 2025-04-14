

<div align="center">
   <h1>Research Assistant</h1>
</div>

### 🧩 Flow Diagram

   <img src="https://github.com/genieincodebottle/generative-ai/blob/main/images/sequence_diagram.png" alt="Flow Diagram"/>

### 🌟 Overview

An AI-powered agentic research assistant that automates topic research, data analysis, and report generation using CrewAI's agentic framework.

### ✨ Features

- AI-driven topic research
- Supports multiple LLM providers

### ⚙️ Setup Instructions

- #### Prerequisites
   - Python 3.9 or higher
   - pip (Python package installer)

- #### Installation
   1. Clone the repository:
      ```bash
      git clone https://github.com/genieincodebottle/generative-ai.git
      cd generative-ai/genai-usecases/agentic-ai/agents/crewai_usecases/research_assistant
      ```
   2. Create a virtual environment:
      ```bash
      python -m venv venv
      venv\Scripts\activate # On Linux -> source venv/bin/activate
      ```
   3. Install dependencies:
      ```bash
      pip install -r requirements.txt
      ```
   4. Rename `.env.example` to `.env` and update with appropriate values.
      - For **GROQ_API_KEY** follow this -> https://console.groq.com/keys
      - For **OPENAI_API_KEY** follow this -> https://platform.openai.com/api-keys
      - For **GOOGLE_API_KEY** follow this -> https://ai.google.dev/gemini-api/docs/api-key
      - For **ANTHROPIC_API_KEY** follow this -> https://console.anthropic.com/settings/keys
      - For **SERPER_API_KEY** follow this -> https://serper.dev/api-key
<hr>

### 💻 Running the Application
To start the application, run:
```bash
streamlit run app.py
```

<img src="https://github.com/genieincodebottle/generative-ai/blob/main/images/research_assistant_ui.png" alt="UI"/>
<hr>

### 📖 Usage

1. Select LLM provider and model
2. Enter a research topic
3. Select the research type, depth level and specific requirements
4. Start the research
5. View insights and download the final report