### Gemini-2.5 Flash Image Generator -> Code name: nano banana :)

<img src="./image.png"/>

### 🔗 Dependencies

```bash
google-genai>=1.31.0
python-dotenv>=1.0.1
streamlit>=1.48.1
```

### ⚙️ Setup Instructions

- #### Prerequisites
   - Python 3.10 or higher
   - pip (Python package installer)

- #### Installation
   1. Clone the repository:
      ```bash
      git clone https://github.com/genieincodebottle/generative-ai.git
      cd genai-usecases\gemini-nano-banana
      ```
   2. Create a virtual environment:
      ```bash
      pip install uv #if uv not installed
      uv venv
      .venv\Scripts\activate # On Linux -> source venv/bin/activate
      ```
   3. Install dependencies:
      ```bash
      uv pip install -r requirements.txt
      ```
   4. Set up environment variables
      * Rename .env.example to .env
      * Update the file with your API keys:
      
      ```bash
      GOOGLE_API_KEY=your_key_here # Gemini-2.5 Flash Image Preview is available as paid api at present
      ```
      * 🔑 Get your API keys:

      For **GOOGLE_API_KEY** follow this -> https://aistudio.google.com/app/apikey

6. Run App
   
   `streamlit run app.py`