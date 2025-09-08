import streamlit as st

import translation_agent as ta
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='app.log', encoding='utf-8', level=logging.INFO)


# Page configuration
st.set_page_config(page_title="Machine Translation Tool", page_icon="🌐")
st.header("`Machine Translation`")
st.info("`Machine Translation using LLM.`")
st.sidebar.image("img/globe.png")
# Hide Streamlit menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Model selection
model_options = ['gemini-pro', 'OpenAI']
model_selection = st.sidebar.selectbox('Select Model', options=model_options)

source_language = ['English', 'Spanish', 'French', 'Hindi']
source_language_selection = st.sidebar.selectbox('Select Source', options=source_language)

target_language = ['Spanish', 'English', 'French', 'Hindi']
target_language_selection = st.sidebar.selectbox('Select Target', options=target_language)

country = ['Mexico', 'India', 'USA', 'England', 'France']
country_selection = st.sidebar.selectbox('Select Country', options=country)


# Container setup
reply_container = st.container()
container = st.container()

submit_button = None
user_input = None
with container:
    if model_selection and source_language_selection and country_selection:
        with st.form(key='chat_form', clear_on_submit=True):
            source_text = st.text_input("`Translate`", key='input')
            submit_button = st.form_submit_button(label='Send ⬆️')
    else:
        st.text("Input required")

    # Response generation
    if submit_button and source_text:
        translation = ta.translate(
                        source_lang=source_language_selection,
                        target_lang=target_language_selection,
                        source_text=source_text,
                        country=country,
                        model=model_selection
                    )
        st.info(f"Source: : {source_text}")
        st.info(f"Translation: {translation}")
