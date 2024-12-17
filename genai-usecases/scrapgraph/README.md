
# Research Tool using ScrapeGraphAI

ScrapeGraphAI is a *web scraping* python library that uses LLM and direct graph logic to create scraping pipelines for websites and local documents (XML, HTML, JSON, etc.).

Just say which information you want to extract and the library will do it for you!

<p align="center">
  <img src="./img/globe.png" style="width: 50%;">
</p>

#  Installation Steps

## WSL at Windows PC if not Using Linux system
Install WSL if it is not already installed to run the code at Linux env. I found some issue running ScrapeGraphAI library at windows system.

https://learn.microsoft.com/en-us/windows/wsl/install



## Install Ollama in the WSL/Linux system

Install Ollama at WSL using following curl command.

https://ollama.com/download/linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```


## Install libraries using requirements.txt in your cloned project (At your IDE using WSL prompt or directly at WSL prompt)

```bash
pip install -r requirements.txt
```

## Create Virtual env

```bash
virtualenv .venv
```

## Activate Virtual env (You need to activate virtual env everytime you start streamlit app)

```bash
source .venv/bin/activate
```

## Start Streamlit App

```bash
streamlit run app.py
```
