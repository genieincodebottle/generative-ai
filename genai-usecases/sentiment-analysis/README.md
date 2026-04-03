# Customer Call Sentiment & Aggressiveness Analyzer

A beginner-friendly GenAI application that analyzes customer call transcripts using Cloud LLMs to extract **sentiment** (Positive / Negative / Neutral) and **aggressiveness scores** (1-10) with real-time processing and SQLite database storage.

Supports two LLM providers:
- **Groq** (Primary - Free tier, fast inference) - recommended for beginners
- **Google Gemini** (Secondary - Free tier available)

<img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" /> <img src="https://img.shields.io/badge/Streamlit-1.49+-red?logo=streamlit" /> <img src="https://img.shields.io/badge/LangChain-0.3+-green" />

## Features

- 🤖 **Dual LLM Providers** - Groq (free, fast) and Google Gemini with easy switching
- 🛡️ **Structured Output** - Pydantic models ensure consistent sentiment and aggressiveness scoring
- 🔐 **Security** - Parameterized SQL queries prevent injection attacks
- 🗄️ **Zero-Setup Database** - SQLite auto-creates everything, no installation needed
- 📊 **Statistics Dashboard** - Sentiment distribution, aggression scores, and more

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.9 or higher ([download](https://www.python.org/downloads/)) |
| **API Key** | At least one: Groq (free) or Google Gemini (free) |
| **uv** (recommended) | Fast Python package manager (`pip install uv`) |

### Get Your Free API Key (pick one or both)

| Provider | Free Tier | Get Key |
|---|---|---|
| **Groq** (recommended) | Generous free tier | [console.groq.com/keys](https://console.groq.com/keys) |
| **Google Gemini** | Free tier available | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

## Quick Start (Step by Step)

### Step 1: Clone and Navigate

```bash
git clone https://github.com/genieincodebottle/generative-ai.git
cd generative-ai/genai-usecases/sentiment-analysis
```

### Step 2: Create Virtual Environment

```bash
pip install uv          # if uv is not installed
uv venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
uv pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

```bash
# Copy the example file
cp .env.example .env    # macOS/Linux
copy .env.example .env  # Windows
```

Open `.env` in any text editor and add your API key:

```env
# If using Groq (recommended):
GROQ_API_KEY=gsk_your_actual_key_here

# If using Gemini:
GOOGLE_API_KEY=your_actual_key_here

# Choose your provider (default is groq):
LLM_PROVIDER=groq
```

> **Tip**: You only need ONE API key to get started. Groq is recommended because it's free and fast.

### Step 5: Run the Application

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

### Step 6: Use the App

1. **Setup Database tab** → Click **"Initialize Database"** to create the SQLite database with 10 sample customer calls
2. **Process Calls tab** → Click **"Fetch and Process All Calls"** to analyze all calls
3. **View Results tab** → See sentiment analysis results and statistics

That's it! You're running an end-to-end GenAI sentiment analysis pipeline.

## Available Models

### Groq Models (Primary)
| Model | Parameters | Best For |
|---|---|---|
| `llama-3.1-8b-instant` | 8B | Good balance of speed and quality (default) |
| `llama-3.3-70b-versatile` | 70B | Best quality, slower |
| `meta-llama/llama-4-scout-17b-16e-instruct` | 17B MoE | Latest Llama 4, great quality |

### Gemini Models (Secondary)
| Model | Best For |
|---|---|
| `gemini-2.0-flash` | Fast and efficient (default) |
| `gemini-2.5-flash` | Better reasoning |
| `gemini-2.5-pro` | Best quality |

## Configuration

All settings can be configured through the **sidebar** in the app:

- **Provider**: Switch between Groq and Gemini
- **Model**: Choose from available models for the selected provider
- **Temperature**: 0 = deterministic (recommended), 1 = creative
- **Database Path**: SQLite file location (default: `sentiment_analysis.db`)

You can also configure via the `.env` file (see `.env.example` for all options).

## Output Format

The analysis returns for each customer call:
- **Sentiment**: `Positive`, `Negative`, or `Neutral`
- **Aggressiveness**: Integer from 1 (calm) to 10 (extremely aggressive)

## Troubleshooting

| Problem | Solution |
|---|---|
| **"GROQ_API_KEY not found"** | Add your key to `.env` file. Get one free at [console.groq.com/keys](https://console.groq.com/keys) |
| **"GOOGLE_API_KEY not found"** | Add your key to `.env` file. Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| **"No LLM provider installed"** | Run `uv pip install -r requirements.txt` again |
| **"Please initialize the database first"** | Go to **Setup Database** tab and click **Initialize Database** |
| **Processing errors** | Check that call text is 10-10,000 characters. Try a different model. |
| **After database reset, no data** | Normal. Go to Setup Database tab → Initialize Database to reload sample data |
| **Rate limit errors (Groq)** | Wait a minute and retry. Free tier has rate limits. |
| **Import errors** | Make sure your virtual environment is activated and dependencies are installed |

## Technology Stack

| Component | Technology |
|---|---|
| **Frontend** | Streamlit |
| **LLM Framework** | LangChain 0.3+ |
| **LLM Providers** | Groq (ChatGroq), Google Gemini (ChatGoogleGenerativeAI) |
| **Output Parsing** | Pydantic + LangChain JsonOutputParser |
| **Database** | SQLite (built-in, zero setup) |
| **Config** | python-dotenv |
