
<div align="center">
    <a target="_blank" href="https://www.youtube.com/@genieincodebottle"><img src="https://img.shields.io/badge/YouTube-@genieincodebottle-blue"></a>&nbsp;
    <a target="_blank" href="https://www.linkedin.com/in/rajesh-srivastava"><img src="https://img.shields.io/badge/style--5eba00.svg?label=LinkedIn&logo=linkedin&style=social"></a>&nbsp;
    <a target="_blank" href="https://www.instagram.com/genieincodebottle/"><img src="https://img.shields.io/badge/@genieincodebottle-C13584?style=flat-square&labelColor=C13584&logo=instagram&logoColor=white&link=https://www.instagram.com/eduardopiresbr/"></a>&nbsp;
    <a target="_blank" href="https://github.com/genieincodebottle/generative-ai/blob/main/GenAI_Roadmap.md"><img src="https://img.shields.io/badge/style--5eba00.svg?label=GenAI Roadmap&logo=github&style=social"></a>
</div>

### Llama-4 Scout: All-in-One App for Chat, OCR, RAG & Agentic AI with CrewAI Integration

### 🔗 Dependencies

```bash
streamlit>=1.43.2 
groq>=0.22.0
python-dotenv>=1.1.0
langchain-groq>=0.3.2
langchain-community>=0.0.27
langchain>=0.0.27
python-dotenv>=1.0.0
pypdf>=4.0.0
faiss-cpu>=1.7.4
pillow>=10.2.0
streamlit-chat>=0.1.1
sentence-transformers>=2.2.2
crewai>=0.28.5
crewai-tools>=0.40.1
google-genai>=1.5.0
```

### ⚙️ Setup Instructions

- #### Prerequisites
   - Python 3.9 or higher
   - pip (Python package installer)

- #### Installation
   1. Clone the repository:
      ```bash
      git clone https://github.com/genieincodebottle/generative-ai.git
      cd genai-usecases/llms/llama-4
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
   4. Provide Groq API key in `.env` file
      ```bash
      GROQ_API_KEY=your_key_here
      GOOGLE_API_KEY=your_key_here
      ```

      For **GROQ_API_KEY** follow this -> https://console.groq.com/keys

      For **GOOGLE_API_KEY** follow this -> https://aistudio.google.com/app/apikey

6. Run App
   
   `streamlit run app.py`