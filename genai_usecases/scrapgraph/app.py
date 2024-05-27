import streamlit as st

from scrapegraphai.graphs import SmartScraperGraph, SearchGraph, SpeechGraph
from scrapegraphai.utils import prettify_exec_info

import utils
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='app.log', encoding='utf-8', level=logging.INFO)


# Page configuration
st.set_page_config(page_title="Research Tool", page_icon="🌐")
st.header("`Research Tool`")
st.info("`I am an AI Agent equipped to provide insightful answers by delving into, comprehending, \
        and condensing information from various web sources.`")

# Hide Streamlit menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Sidebar setup
st.sidebar.image("img/globe.png")

# Model selection
model_options = ['ollama', 'OpenAI', 'gemini-pro']
model_selection = st.sidebar.selectbox('Select Model', options=model_options)
model_type_selection = None
if model_selection == "ollama":
    model_type_options = ['llama3', 'mistral', 'phi3']
    model_type_selection = st.sidebar.selectbox('Select Model Type', options=model_type_options)
elif model_selection == "OpenAI":
    model_type_selection = "gpt-3.5-turbo"
elif model_selection == "gemini-pro":
    model_type_selection = "gemini-pro"

# Scraping options
scrap_options = ['SearchGraph', 'SmartScraperGraph', 'SpeechGraph']
scrap_selection = st.sidebar.selectbox('Select Scrap Option', options=scrap_options)

# User input handling
if scrap_selection in ["SmartScraperGraph", "SpeechGraph"]:
    source_text_input = st.sidebar.text_input("`Source:`", key='source')
    #source_text_input = st.sidebar.file_uploader("Upload file", accept_multiple_files=True)
else:
    source_text_input = None

# Step 2: Initialize Streamlit
#file_uploader, the data are copied to the Streamlit backend via the browser, 
#and contained in a BytesIO buffer in Python memory (i.e. RAM, not disk).
#pdf_files = st.sidebar.file_uploader("Upload file", accept_multiple_files=True)


# Container setup
reply_container = st.container()
container = st.container()

submit_button = None
user_input = None
with container:
    if scrap_selection in ["SmartScraperGraph", "SpeechGraph"] and source_text_input:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("`Ask a question:`", key='input')
            submit_button = st.form_submit_button(label='Send ⬆️')
    elif scrap_selection == "SearchGraph":
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("`Ask a question:`", key='input')
            submit_button = st.form_submit_button(label='Send ⬆️')
    else:
        st.text("Source Input required")

    # Response generation
    if submit_button and user_input:
        result = utils.main(model_selection, model_type_selection, scrap_selection, user_input, source_text_input)    
        st.info(result)
