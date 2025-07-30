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

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

    """
    parameter: allow_dangerous_requests: bool = True
        Forced user opt-in to acknowledge that the chain can make dangerous requests.
    Security note: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions. Failure to do so may result 
        in data corruption or loss, since the calling code may attempt commands 
        that would result in deletion, mutation of data if appropriately prompted or 
        reading sensitive data if such data is present in the database. The best way 
        to guard against such negative outcomes is to (as appropriate) limit the permissions 
        granted to the credentials used with this tool.
        See https://python.langchain.com/docs/security for more information.
    """
    
    chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True)
    response = chain.invoke({"query": query})
    return response['result']
