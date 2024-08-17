from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os
from dotenv import load_dotenv

load_dotenv()

def main(model=None, u_question=None):
    prompt_template = """
        Answer the question as detailed as possible.
            Question: \n{question}\n

        Answer:
        """
    
    if model == "open-ai":
        api_key = os.getenv("OPENAI_API_KEY")
        
        llm = ChatOpenAI(api_key=api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
        llm_chain = prompt | llm
        response = llm_chain.invoke({'question':u_question})
        return response
    elif model == "gemini-pro":

        api_key = os.getenv("GOOGLE_API_KEY")

        llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        temperature=0.3,
        google_api_key=api_key,  
        safety_settings={ 
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },)
        
        prompt = PromptTemplate(template=prompt_template, input_variables=[ "question"])
        llm_chain = prompt | llm
        response = llm_chain.invoke({'question':u_question})
        
        return response.content

if __name__ == "__main__":
    
    question = """How can AI and machine learning be integrated into 
    microservices architecture to improve scalability and reliability in telecom systems?
    """

    response = main(model="gemini-pro", u_question=question)

    print (response)