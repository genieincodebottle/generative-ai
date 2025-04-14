

<div align="center">
   <h1>Data Analysis Assistant</h1>
</div>

### 🧩 Flow Diagram

   <img src="https://github.com/genieincodebottle/generative-ai/blob/main/images/data_analysis_sequence_diagram.png" alt="Sequence Diagram"/>

### 🌟 Overview

A data analysis assistant that combines AI powered analysis with interactive visualizations and analysis generation using CrewAI agentic framework.

<hr>

### ✨ Features

- Automated data preprocessing
- Comprehensive statistical analysis
- Interactive data visualizations
- Supports multiple LLM providers

<hr>

### ⚙️ Setup Instructions

- #### Prerequisites
   - Python 3.9 or higher
   - pip (Python package installer)

- #### Installation
   1. Clone the repository:
      ```bash
      git clone https://github.com/genieincodebottle/generative-ai.git
      cd generative-ai/genai-usecases/agentic-ai/agents/crewai_usecases/data_analysis_assistant
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
   4. Rename `.env.example` to `.env` and update with appropriate values as per LLM Provider.
      - For **GROQ_API_KEY** follow this -> https://console.groq.com/keys
      - For **OPENAI_API_KEY** follow this -> https://platform.openai.com/api-keys
      - For **GOOGLE_API_KEY** follow this -> https://ai.google.dev/gemini-api/docs/api-key
      - For **ANTHROPIC_API_KEY** follow this -> https://console.anthropic.com/settings/keys
<hr>

### 💻 Running the Application
To start the application, run:
```bash
streamlit run app.py
```
<br>
<img src="https://github.com/genieincodebottle/generative-ai/blob/main/images/data_analysis_assistant_ui.png" alt="UI"/>

<hr>

### 📖 Usage

1. Select LLM provider and model
2. Upload your dataset (CSV or Excel)
3. Start the analysis
4. Explore results through interactive visualizations
5. Download analysis reports
