import streamlit as st
from streamlit_chat import message

# All utility functions
import utils

from PIL import Image

def initialize_session_state():
    """
    Session State is a way to share variables between reruns, for each user session.
    """

    st.session_state.setdefault('history', [])
    st.session_state.setdefault('generated', ["Hello! I am here to provide answers to questions fetched from Database."])
    st.session_state.setdefault('past', ["Hello Buddy!"])

def display_chat(conversation_chain, chain):
    """
    Streamlit relatde code where we are passing conversation_chain instance created earlier
    It creates two containers
    container: To group our chat input form
    reply_container: To group the generated chat response

    Args:
    - conversation_chain: Instance of LangChain ConversationalRetrievalChain
    """
    #In Streamlit, a container is an invisible element that can hold multiple 
    #elements together. The st.container function allows you to group multiple 
    #elements together. For example, you can use a container to insert multiple 
    #elements into your app out of order.
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask me questions from uploaded PDF", key='input')
            submit_button = st.form_submit_button(label='Send ⬆️')
        
        #Check if user submit question with user input and generate response of the question
        if submit_button and user_input:
            generate_response(user_input, conversation_chain, chain)
    
    #Display generated response to streamlit web UI
    display_generated_responses(reply_container)


def generate_response(user_input, conversation_chain, chain):
    """
    Generate LLM response based on the user question by retrieving data from Database
    Also, stores information to streamlit session states 'past' and 'generated' so that it can
    have memory of previous generation for converstational type of chats (Like chatGPT)

    Args
    - user_input(str): User input as a text
    - conversation_chain: Instance of ConversationalRetrievalChain 
    """

    with st.spinner('Spinning a snazzy reply...'):
        output = conversation_chat(user_input, conversation_chain, chain, st.session_state['history'])

    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

def conversation_chat(user_input, conversation_chain, chain, history):
    """
    Returns LLM response after invoking model through conversation_chain

    Args:
    - user_input(str): User input
    - conversation_chain: Instance of ConversationalRetrievalChain
    - history: Previous response history
    returns:
    - result["answer"]: Response generated from LLM
    """
    response = conversation_chain.invoke(user_input)
    final_response = chain.invoke(f"Based on the following information generate human redable response: {response['query']},  {response['result']}")

    history.append((user_input, final_response))
    return final_response


def display_generated_responses(reply_container):
    """
    Display generated LLM response to Streamlit Web UI

    Args:
    - reply_container: Streamlit container created at previous step
    """
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="adventurer")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

def main():
    """
    First function to call when we start streamlit app
    """
    # Step 1: Initialize session state
    initialize_session_state()
    
    st.title("Genie")

    image = Image.open('chatbot.jpg')
    st.image(image, width=150)
    
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>

            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    # Step 2: Initialize Streamlit
    conversation_chain, chain = utils.create_conversational_chain()

    #Step 3 - Display Chat to Web UI
    display_chat(conversation_chain, chain)

if __name__ == "__main__":
    main()
