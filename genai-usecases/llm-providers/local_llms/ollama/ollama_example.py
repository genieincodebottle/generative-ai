""""
1. Follow READEME file to Download, Install, Pull Models and Run
For the this example pull below model

ollama pull llama3.2:1b

2. Install required library
pip install -qU langchain-ollama

3. Run this python module
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Loads a local Llama 3.2 model through Ollama, with a temperature (0.5) that 
# controls how creative the answers are.
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.5,
)

# Creates a chat prompt:
# "system" message sets the AI’s role (a helpful step-by-step assistant).
# "human" message is the user’s question, where {problem} is a placeholder.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains problems step-by-step."),
    ("human", "Solve this problem step by step: {problem}"),
])

# Connects the prompt and the LLM into a pipeline (so input flows through the prompt into the LLM).
chain = prompt | llm

# Sends the user’s request by filling {problem} with "Provide python code of sudoku".
ai_msg = chain.invoke({"problem": "Provide python code of sudoku"})

# Displays the AI’s final answer.
print(ai_msg.content)