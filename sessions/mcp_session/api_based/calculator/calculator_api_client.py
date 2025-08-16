import os
from dotenv import load_dotenv
load_dotenv()

import requests
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Tool for calculating expressions
class CalculateExpressionTool(BaseTool):
    name: str = "calculate_expression"
    description: str = """
    Calculate mathematical expressions using a REST API.
    Input should be a valid mathematical expression as a string.
    Examples: "2+2", "sqrt(16)", "sin(pi/2)", "log(e)"
    """
    
    def _run(self, expression: str) -> str:
        """Execute calculation via REST API"""
        try:
            response = requests.post(
                "http://localhost:8000/calculate",
                json={"expression": expression},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            if data["success"]:
                return f"{data['expression']} = {data['result']}"
            else:
                return f"Error: {data['error']}"
                
        except requests.exceptions.RequestException as e:
            return f"API Error: {str(e)}"

# Tool for getting available functions
class GetFunctionsTool(BaseTool):
    name: str = "get_available_functions"
    description: str = """
    Get information about available mathematical functions and operations.
    No input required. Returns documentation of what mathematical operations are supported.
    """
    
    def _run(self, query: str = "") -> str:
        """Get available functions via REST API"""
        try:
            response = requests.get(
                "http://localhost:8000/functions",
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            result = "Available Mathematical Operations:\n\n"
            result += f"Basic Operations: {', '.join(data['operations'])}\n\n"
            
            result += "Functions:\n"
            for func, desc in data['functions'].items():
                result += f"  {func}: {desc}\n"
            
            result += "\nConstants:\n"
            for const, value in data['constants'].items():
                result += f"  {const} = {value}\n"
            
            result += "\nExamples:\n"
            for example in data['examples']:
                result += f"  {example}\n"
            
            return result
            
        except requests.exceptions.RequestException as e:
            return f"API Error: {str(e)}"
    
# Initialize tools
def create_calculator_agent():
    """Create a LangChain agent with calculator tools"""
    
    # Initialize tools
    tools = [
        CalculateExpressionTool(),
        GetFunctionsTool()
    ]
    
    # Initialize Google Gemini LLM (replace with your Google API key)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.5
    )
    
    # Create agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent

# Example usage functions
def run_calculator_examples():
    """Run some example calculations"""
    
    print("Creating calculator agent...")
    agent = create_calculator_agent()
    
    examples = [
       "Calculate 15% of 250",
        "What mathematical functions are available?",
        "Show me the calculation history",
        "Solve the quadratic equation: find the discriminant of 2x^2 + 5x + 2"
    ]
    
    for example in examples:
        print(f"\n{'='*50}")
        print(f"Query: {example}")
        print(f"{'='*50}")
        
        try:
            result = agent.invoke(example)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("LangChain Calculator Tools with Google Gemini")
    print("Make sure the FastAPI server is running on localhost:8000")
    print("Set GOOGLE_API_KEY environment variable or update the code with your API key")
    
    # Run with agent (requires Google API key)
    run_calculator_examples()
    