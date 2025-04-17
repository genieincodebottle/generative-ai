"""
RAG Evaluation Module: Provides tools to evaluate RAG system responses
using either Gemini's lightweight evaluation or RAGAS metrics.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any

import plotly.graph_objects as go
from datasets import Dataset
from dotenv import load_dotenv
from google import genai
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiLightweightEvaluator:
    """Lightweight RAG Evaluator using Google's Gemini model"""
    
    def __init__(self, google_api_key=None):
        """Initialize the evaluator with the Gemini model"""
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.google_api_key:
            raise ValueError("Google API key is required for Gemini evaluation")
            
        # Import here to avoid dependency issues if not using this evaluator
        client = genai.Client(api_key=self.google_api_key)
        self.client = client
        
    def evaluate_response(self, query: str, answer: str, contexts: List[str], 
                         ground_truth: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate RAG response using Gemini model
        
        Args:
            query: The user's query
            answer: The generated answer
            contexts: Retrieved context passages
            ground_truth: Optional ground truth answer
            
        Returns:
            Dict with evaluation metrics
        """
        results = {}
        
        # Evaluate answer relevancy
        results["answer_relevancy"] = self._evaluate_answer_relevancy(query, answer)
        
        # Evaluate faithfulness to the context
        results["faithfulness"] = self._evaluate_faithfulness(query, answer, contexts)
        
        # If ground truth is provided, evaluate precision
        if ground_truth:
            results["context_precision"] = self._evaluate_context_precision(answer, ground_truth)
        
        return results
    
    def _evaluate_answer_relevancy(self, query: str, answer: str) -> float:
        """Evaluate how relevant the answer is to the query"""
        prompt = f"""On a scale of 0 to 1 (where 1 is best), rate how directly this answer addresses the query.
        Only respond with a number between 0 and 1, with up to 2 decimal places.
        
        Query: {query}
        Answer: {answer}
        
        Rating (0-1):"""
        
        try:
            response = self.client.models.generate_content(
                model="gemini-1.5-pro",
                contents=[prompt]
            )
            rating_text = response.text.strip()
            
            # Extract the numerical rating
            if rating_text:
                try:
                    # Find the first float in the response
                    import re
                    matches = re.findall(r"0\.\d+|\d+\.?\d*", rating_text)
                    if matches:
                        rating = float(matches[0])
                        return min(max(rating, 0.0), 1.0)  # Ensure value is between 0 and 1
                except:
                    pass
        except Exception as e:
            logger.error(f"Error evaluating answer relevancy with Gemini: {str(e)}")
            raise e
                
    def _evaluate_faithfulness(self, query: str, answer: str, contexts: List[str]) -> float:
        """Evaluate how faithful the answer is to the provided contexts"""
        # Join contexts with separators for clarity
        context_text = "\n\n---\n\n".join(contexts)
        
        prompt = f"""On a scale of 0 to 1 (where 1 is best), evaluate how factually accurate and faithful this answer is based ONLY on the provided context.
        Does the answer contain claims not supported by the context? 
        Does it contradict the context?
        Only respond with a number between 0 and 1, with up to 2 decimal places.
        
        Query: {query}
        Context: {context_text}
        Answer: {answer}
        
        Faithfulness rating (0-1):"""
        
        try:
            response = self.client.models.generate_content(
                model="gemini-1.5-pro",
                contents=[prompt]
            )
            rating_text = response.text.strip()
            
            # Extract the numerical rating
            if rating_text:
                try:
                    # Find the first float in the response
                    import re
                    matches = re.findall(r"0\.\d+|\d+\.?\d*", rating_text)
                    if matches:
                        rating = float(matches[0])
                        return min(max(rating, 0.0), 1.0)  # Ensure value is between 0 and 1
                except:
                    pass
        except Exception as e:
            logger.error(f"Error evaluating faithfulness with Gemini: {str(e)}")
            raise e
    
    def _evaluate_context_precision(self, answer: str, ground_truth: str) -> float:
        """Evaluate how close the answer is to the ground truth"""
        prompt = f"""On a scale of 0 to 1 (where 1 is best), rate how well the given answer matches the ground truth.
        Consider factual accuracy, completeness, and correctness.
        Only respond with a number between 0 and 1, with up to 2 decimal places.
        
        Answer: {answer}
        Ground Truth: {ground_truth}
        
        Rating (0-1):"""
        
        try:
            response = self.client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[
                    prompt
                ]
            )
            rating_text = response.text.strip()
            
            # Extract the numerical rating
            if rating_text:
                try:
                    # Find the first float in the response
                    import re
                    matches = re.findall(r"0\.\d+|\d+\.?\d*", rating_text)
                    if matches:
                        rating = float(matches[0])
                        return min(max(rating, 0.0), 1.0)  # Ensure value is between 0 and 1
                except:
                    pass
        except Exception as e:
            logger.error(f"Error evaluating context precision with Gemini: {str(e)}")
            raise e

def evaluate_with_gemini(query, answer, contexts, ground_truth=None):
    """
    Convenience function to evaluate using Gemini
    
    Args:
        query: The user's question
        answer: The generated answer
        contexts: The context passages
        ground_truth: Optional ground truth answer
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        evaluator = GeminiLightweightEvaluator()
        return evaluator.evaluate_response(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )
    except Exception as e:
        logger.error(f"Gemini evaluation failed: {str(e)}")
        results = {"answer_relevancy": 0.5, "faithfulness": 0.5}
        if ground_truth:
            results["context_precision"] = 0.5
        return results

# For RAGAs based Rag Evaluation
# Define basic metrics
BASE_METRICS = [answer_relevancy, faithfulness]
GROUND_TRUTH_METRICS = [answer_relevancy, faithfulness, context_precision]

def format_context(context: Union[str, List[str], List[List[str]]]) -> List[str]:
    """Format context into a flat list of strings as expected by SingleTurnSample."""
    if not context:
        return [""]

    if isinstance(context, str):
        return [context]

    if isinstance(context[0], str):
        return context

    # Flatten list of lists
    return [item for sublist in context for item in sublist]

def convert_to_float(value: Any) -> float:
    """Convert value to float with validation."""
    try:
        if isinstance(value, list):
            value = value[0] if len(value) > 0 else 0.0
        return float(value)
    except (TypeError, ValueError):
        logger.warning(f"Failed to convert value to float: {value}")
        return 0.0

def process_results(results: Dict[str, Any], include_precision: bool) -> Dict[str, float]:
    """Process evaluation results to proper format."""
    try:
        processed_results = {
            "answer_relevancy": convert_to_float(results["answer_relevancy"]),
            "faithfulness": convert_to_float(results["faithfulness"])
        }
        
        if include_precision and "context_precision" in results:
            processed_results["context_precision"] = convert_to_float(results["context_precision"])
            
        return processed_results
        
    except Exception as e:
        logger.error(f"Results processing failed: {str(e)}")
        return create_empty_results(include_precision)

def create_empty_results(include_precision: bool) -> Dict[str, float]:
    """Create empty results dictionary."""
    results = {
        "answer_relevancy": 0.0,
        "faithfulness": 0.0
    }
    if include_precision:
        results["context_precision"] = 0.0
    return results

def setup_environment(model_name: str, reuse_same_model: bool):
    """Set up environment and return evaluator if needed."""
    evaluator = None
    
    try:
        if reuse_same_model:
            # Use Groq as evaluator
            os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
            evaluator = ChatGroq(
                model_name=model_name,
                temperature=0.5
            )
            os.environ["OPENAI_API_KEY"] = "DUMMY"
        else:
            # Use OpenAI as evaluator
            load_dotenv(override=True)
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    except Exception as e:
        logger.error(f"Environment setup failed: {str(e)}")
        
    return evaluator

def evaluate_single_query(
    model_provider: str,
    model_name: str,
    question: str,
    answer: str, 
    contexts: List[str],
    resuse_same_model_for_eval: bool,
    ground_truth: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate a single query response.
    
    Args:
        model_provider: The LLM provider (e.g., "openai", "claude")
        model_name: Specific model name to use
        question: Query question
        answer: Generated answer
        contexts: Retrieved context passages
        resuse_same_model_for_eval: Whether to reuse same model for evaluation
        ground_truth: Optional ground truth answer
        
    Returns:
        Dictionary containing evaluation scores
    """
    try:
        # Setup environment and get evaluator if needed
        evaluator = setup_environment(model_name, resuse_same_model_for_eval)
        
        # Format context
        formatted_contexts = format_context(contexts)
        
        # Create a dataset with the single query
        eval_dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [formatted_contexts],
            **({"ground_truth": [ground_truth]} if ground_truth else {})
        })

        # Select metrics based on ground truth availability
        metrics = GROUND_TRUTH_METRICS if ground_truth else BASE_METRICS
        
        # Run evaluation
        if resuse_same_model_for_eval and evaluator:
            results = evaluate(eval_dataset, metrics=metrics, llm=evaluator)
        else:
            results = evaluate(eval_dataset, metrics=metrics)

        # Process and return results
        return process_results(results, bool(ground_truth))
        
    except Exception as e:
        logger.error(f"Query evaluation failed: {str(e)}")
        return create_empty_results(bool(ground_truth))

def create_evaluation_chart(eval_results: Dict[str, float]) -> go.Figure:
    """
    Create visualization for evaluation results
    
    Args:
        eval_results: Dictionary of evaluation metrics
        
    Returns:
        Plotly figure object
    """
    # Create friendly metric names
    metric_names = {
        "answer_relevancy": "Answer Relevancy",
        "faithfulness": "Faithfulness",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall"
    }
    
    # Colors for each metric
    colors = {
        "answer_relevancy": "#2563eb",  # Blue
        "faithfulness": "#16a34a",      # Green
        "context_precision": "#d97706", # Amber
        "context_recall": "#9333ea"     # Purple
    }
    
    # Prepare data for chart
    x_values = [metric_names.get(k, k) for k in eval_results.keys()]
    y_values = list(eval_results.values())
    bar_colors = [colors.get(k, "#6b7280") for k in eval_results.keys()]
    
    # Create the bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=x_values,
            y=y_values,
            text=[f"{v:.2f}" for v in y_values],
            textposition="auto",
            marker_color=bar_colors
        )
    ])
    
    # Update layout
    fig.update_layout(
        title="RAG Evaluation Metrics",
        xaxis_title="Metrics",
        yaxis_title="Score (0-1)",
        yaxis_range=[0, 1],
        template="plotly_white"
    )
    
    return fig

