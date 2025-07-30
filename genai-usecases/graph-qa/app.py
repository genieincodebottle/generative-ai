import streamlit as st

import utils
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='app.log', encoding='utf-8', level=logging.INFO)

# Page configuration
st.set_page_config(page_title="Graph Search Tool", page_icon="🌐")
st.header("`Graph Search Tool`")
st.info("`Graph Search tool designed to explore, understand, and distill insights from Graph Databases with precision and ease.`")
st.sidebar.image("img/globe.png")
# Hide Streamlit menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Container setup
reply_container = st.container()
container = st.container()

submit_button = None
user_input = None
with container:
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("`Ask a question:`", key='input')
        submit_button = st.form_submit_button(label='Send ⬆️')
    
    # Response generation
    if submit_button and user_input:
        result = utils.main(user_input)    
        st.info(result)
