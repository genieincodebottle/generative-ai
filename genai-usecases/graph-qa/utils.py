import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

def main(query):

    graph = Neo4jGraph()
 
    print(graph.schema)

    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)
    response = chain.invoke({"query": query})
    print(response)
    return response['result']
