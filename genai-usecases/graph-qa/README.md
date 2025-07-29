### Graph-QA using free tier Gemini LLM

🎥 YouTube Video:  Walkthrough on setup and running the app

[![Watch the video](https://img.youtube.com/vi/PJTxPW5He7w/0.jpg)](https://www.youtube.com/watch?v=PJTxPW5He7w)

### Neo4J Installation

#### Download and Install Neo4j Desktop:
- Navigate to the Neo4j download page at [neo4j.com/download/](https://neo4j.com/download/)
- Click the "Download" button.
- Fill out the registration form with your details and click "Download Desktop".
- On the next page, copy the Neo4j Desktop Activation Key to your clipboard.
- Once the download is complete, run the installer.
- Follow the on-screen instructions, keeping the default settings.
- After installation, the Neo4j Desktop application will launch.
- Agree to the license agreement.
- Choose a path to store application data or keep the default.
- In the software registration window, paste the activation key you copied earlier and click "Activate".

#### Configure the Neo4j Database:
- Once Neo4j Desktop is open, an example project named "Movie DBMS" will be available and active.
- Click on the "Movie DBMS" instance.
- Go to the "Plugins" tab.
- Select "APOC" and click "Install and Restart".
- Go to the "Details" tab and click on "Reset DBMS password".
- Enter a new password for your database.

### Setup Instructions

#### Prerequisites
   - Python 3.10 or higher
   - pip (Python package installer)

#### Installation & Running App
   1. Clone the repository:
      ```bash
      git clone https://github.com/genieincodebottle/generative-ai.git
      cd genai-usecases/graph-qa
      ```
   2. Open your project in a code editor like VS Code.
   3. Create a virtual environment by running the following command in the terminal:
      ```bash
      python -m venv .venv
      .venv\Scripts\activate # On Linux -> source venv/bin/activate
      ```
   4. Create a requirements.txt file and add the following libraries:
      ```
      langchain>=0.3.27
      langchain-community>=0.2.1
      langchain-google-genai>=2.1.8
      neo4j>=5.28.1
      streamlit>=1.47.1
      streamlit-chat>=0.1.1
      python-dotenv==1.1.1
      ```
   5. Install dependencies:
      ```bash
      pip install -r requirements.txt
      ```
   6. Rename .env.example to .env and update the following API keys in the .env file
      ```bash
      GOOGLE_API_KEY=your_key_here # Using the free-tier API Key
      NEO4J_URI="bolt://localhost:7687"
      NEO4J_USERNAME="neo4j"
      NEO4J_PASSWORD="your_new_password"
      ```
      For **GOOGLE_API_KEY** follow this -> https://aistudio.google.com/app/apikey
   7. **utility file (utils.py):** This file contains the backend logic for connecting to the database and processing queries. The code in the video includes functions to load environment variables, connect to the Neo4j graph, and use the LangChain library to convert natural language questions into Cypher queries.
   8. **main application file (app.py):** This file contains the Streamlit code for the user interface. It creates the "Graph Search Tool" title, an input box for the user's question, and a button to submit. When the user submits a question, it calls the main function from utils.py to get the answer from the graph database and displays the result.
   9. Run App
   
      `streamlit run app.py`