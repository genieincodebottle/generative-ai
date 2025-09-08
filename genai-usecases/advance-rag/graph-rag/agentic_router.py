"""
Agentic Router using LangGraph for intelligent retrieval strategy selection.
This module implements an AI based decision system that analyzes queries
and selects the optimal retrieval strategy using LLM reasoning.
"""

from typing import Dict, Any, Literal
from dataclasses import dataclass
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph


class RetrievalStrategy(Enum):
    TRAVERSAL = "traversal"
    STANDARD = "standard"


@dataclass
class RouterState:
    """State for the agentic router workflow"""
    query: str
    query_analysis: str = ""
    selected_strategy: RetrievalStrategy = None
    confidence: float = 0.0
    reasoning: str = ""


class AgenticRetrieverRouter:
    """
    LangGraph-based intelligent router that uses LLM analysis to select
    the optimal retrieval strategy for each query.
    """
    
    def __init__(self, llm, traversal_retriever: BaseRetriever, standard_retriever: BaseRetriever):
        self.llm = llm
        self.traversal_retriever = traversal_retriever
        self.standard_retriever = standard_retriever
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> CompiledStateGraph:
        """Build the LangGraph workflow for routing decisions"""
        
        # Create the workflow graph
        workflow = StateGraph(RouterState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("select_strategy", self._select_strategy)
        
        # Add edges
        workflow.add_edge(START, "analyze_query")
        workflow.add_edge("analyze_query", "select_strategy")
        workflow.add_edge("select_strategy", END)
        
        return workflow.compile()
    
    def _analyze_query(self, state: RouterState) -> RouterState:
        """Analyze the query to understand its characteristics and requirements"""
        
        analysis_prompt = ChatPromptTemplate.from_template("""
        Analyze the following query to understand what type of information retrieval would be most effective.
        
        Query: {query}
        
        Consider these aspects:
        1. Does the query ask about relationships, connections, or comparisons between entities?
        2. Does it require exploring related or similar information?
        3. Is it a direct factual question that can be answered with specific information?
        4. Does it involve complex reasoning that might benefit from multiple connected documents?
        5. Are there keywords that suggest the need for context from related documents?
        
        Provide a detailed analysis of the query characteristics in 2-3 sentences.
        Focus on whether the query would benefit from exploring document relationships or direct similarity search.
        """)
        
        try:
            formatted_prompt = analysis_prompt.invoke({"query": state.query})
            response = self.llm.invoke(formatted_prompt)
            
            analysis = response.content if hasattr(response, 'content') else str(response)
            
            # Update state
            state.query_analysis = analysis
            print(f"Query Analysis: {analysis}")
            
        except Exception as e:
            print(f"Error in query analysis: {e}")
            state.query_analysis = f"Error analyzing query: {str(e)}"
        
        return state
    
    def _select_strategy(self, state: RouterState) -> RouterState:
        """Select the optimal retrieval strategy based on query analysis"""
        
        selection_prompt = ChatPromptTemplate.from_template("""
        Based on the query and analysis below, select the best retrieval strategy.
        
        Original Query: {query}
        Query Analysis: {analysis}
        
        Available Strategies:
        
        1. TRAVERSAL RETRIEVER - Best for:
           - Questions about relationships, connections, or similarities between entities
           - Queries that benefit from exploring related documents through graph connections
           - Complex questions requiring context from multiple related sources
           - Comparative analysis or finding patterns across connected information
           
        2. STANDARD RETRIEVER - Best for:
           - Direct factual questions with specific answers
           - Simple information lookup that doesn't require exploring relationships
           - Questions where the most relevant documents can be found through direct similarity
           - Straightforward queries about specific entities or concepts
        
        Instructions:
        1. Choose either "TRAVERSAL" or "STANDARD" based on which strategy would work best
        2. Provide your confidence level as a number between 0.0 and 1.0
        3. Give a brief explanation (1-2 sentences) of why you chose this strategy
        
        Response format:
        STRATEGY: [TRAVERSAL or STANDARD]
        CONFIDENCE: [0.0-1.0]
        REASONING: [Your explanation]
        """)
        
        try:
            formatted_prompt = selection_prompt.invoke({
                "query": state.query,
                "analysis": state.query_analysis
            })
            response = self.llm.invoke(formatted_prompt)
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            print(f"Strategy Selection Response: {response_text}")
            
            # Parse the response
            strategy, confidence, reasoning = self._parse_selection_response(response_text)
            
            state.selected_strategy = strategy
            state.confidence = confidence
            state.reasoning = reasoning
            
            print(f"Selected Strategy: {strategy.value}, Confidence: {confidence}")
            
        except Exception as e:
            print(f"Error in strategy selection: {e}")
            # Fallback to traversal with low confidence
            state.selected_strategy = RetrievalStrategy.TRAVERSAL
            state.confidence = 0.5
            state.reasoning = f"Defaulted to traversal due to error: {str(e)}"
        
        return state
    
    def _parse_selection_response(self, response_text: str) -> tuple[RetrievalStrategy, float, str]:
        """Parse the LLM response to extract strategy, confidence, and reasoning"""
        
        lines = response_text.strip().split('\n')
        strategy = RetrievalStrategy.TRAVERSAL  # Default
        confidence = 0.5  # Default
        reasoning = "Default reasoning"
        
        for line in lines:
            line = line.strip()
            if line.startswith('STRATEGY:'):
                strategy_str = line.split(':', 1)[1].strip().upper()
                if strategy_str == 'STANDARD':
                    strategy = RetrievalStrategy.STANDARD
                elif strategy_str == 'TRAVERSAL':
                    strategy = RetrievalStrategy.TRAVERSAL
            
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence_str = line.split(':', 1)[1].strip()
                    confidence = float(confidence_str)
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                except ValueError:
                    confidence = 0.5
            
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
        
        return strategy, confidence, reasoning
    
    def route(self, query: str) -> tuple[BaseRetriever, Dict[str, Any]]:
        """
        Route the query to the appropriate retriever using agentic decision making.
        
        Returns:
            tuple: (selected_retriever, routing_info)
        """
        
        # Initialize state
        initial_state = RouterState(query=query)
        
        # Run the workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            # Handle case where workflow returns dict instead of RouterState object
            if isinstance(final_state, dict):
                # Convert dict to RouterState-like object access
                selected_strategy = final_state.get('selected_strategy')
                confidence = final_state.get('confidence', 0.5)
                reasoning = final_state.get('reasoning', 'No reasoning provided')
                query_analysis = final_state.get('query_analysis', 'No analysis provided')
            else:
                # Normal RouterState object
                selected_strategy = final_state.selected_strategy
                confidence = final_state.confidence
                reasoning = final_state.reasoning
                query_analysis = final_state.query_analysis
            
            # Select the appropriate retriever
            if selected_strategy == RetrievalStrategy.TRAVERSAL:
                selected_retriever = self.traversal_retriever
            else:
                selected_retriever = self.standard_retriever
            
            # Prepare routing information
            routing_info = {
                "strategy": selected_strategy.value if hasattr(selected_strategy, 'value') else str(selected_strategy),
                "confidence": confidence,
                "reasoning": reasoning,
                "analysis": query_analysis
            }
            
            return selected_retriever, routing_info
            
        except Exception as e:
            print(f"Error in routing workflow: {e}")
            # Fallback to traversal retriever
            routing_info = {
                "strategy": "traversal",
                "confidence": 0.5,
                "reasoning": f"Fallback due to error: {str(e)}",
                "analysis": "Error in analysis"
            }
            return self.traversal_retriever, routing_info
    
    def get_routing_explanation(self, query: str) -> Dict[str, Any]:
        """
        Get detailed routing explanation without actually performing retrieval.
        Useful for debugging and understanding routing decisions.
        """
        _, routing_info = self.route(query)
        return routing_info


def create_agentic_router(llm, traversal_retriever: BaseRetriever, standard_retriever: BaseRetriever) -> AgenticRetrieverRouter:
    """
    Factory function to create an agentic router.
    
    Args:
        llm: Language model for decision making
        traversal_retriever: Graph-based retriever for relationship queries
        standard_retriever: Vector similarity retriever for direct queries
    
    Returns:
        AgenticRetrieverRouter: Configured agentic router
    """
    return AgenticRetrieverRouter(llm, traversal_retriever, standard_retriever)