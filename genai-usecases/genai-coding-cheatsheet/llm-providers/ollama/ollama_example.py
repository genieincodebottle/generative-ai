""""
1. Follow READEME file to Download, Install, Pull Models and Run
For the this example pull below model

ollama pull llama3.2:1b

2. Install required library
pip install -qU langchain-ollama

3. Run this python module
"""

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.5,
)

messages = [
    (
        "system",
        "You are a helpful assistant that explains problems step-by-step.",
    ),
    ("human", "Solve this problem step by step: {problem}"),
]

ai_msg = llm.invoke("Provide me python code of sudoku")
print(ai_msg.content)