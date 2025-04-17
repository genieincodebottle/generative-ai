from dataclasses import dataclass
from functools import lru_cache
import logging
import os
from typing import List, Dict, Any, Optional, Sequence, Union
import numpy as np
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision
)
from typing import Union, List as TypeList
from datasets import Dataset
from dotenv import load_dotenv

import plotly.graph_objects as go


from langchain_groq import ChatGroq

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
        from google import genai
        genai.configure(api_key=self.google_api_key)
        self.genai = genai
        
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
            model = self.genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
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
            return 0.5  # Default if parsing fails
        except Exception as e:
            logger.error(f"Error evaluating answer relevancy with Gemini: {str(e)}")
            return 0.5
    
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
            model = self.genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
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
            return 0.5  # Default if parsing fails
        except Exception as e:
            logger.error(f"Error evaluating faithfulness with Gemini: {str(e)}")
            return 0.5
    
    def _evaluate_context_precision(self, answer: str, ground_truth: str) -> float:
        """Evaluate how close the answer is to the ground truth"""
        prompt = f"""On a scale of 0 to 1 (where 1 is best), rate how well the given answer matches the ground truth.
        Consider factual accuracy, completeness, and correctness.
        Only respond with a number between 0 and 1, with up to 2 decimal places.
        
        Answer: {answer}
        Ground Truth: {ground_truth}
        
        Rating (0-1):"""
        
        try:
            model = self.genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
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
            return 0.5  # Default if parsing fails
        except Exception as e:
            logger.error(f"Error evaluating context precision with Gemini: {str(e)}")
            return 0.5
        
@dataclass
class EvaluationConfig:
    """Configuration settings for RAG evaluation."""
    reuse_same_model: bool = False
    cache_size: int = 128
    batch_size: int = 10

@dataclass
class EvaluationResult:
    """Structured container for evaluation results."""
    answer_relevancy: float
    faithfulness: float
    context_precision: Optional[float] = None
    
    @classmethod
    def create_empty(cls, include_precision: bool = False) -> 'EvaluationResult':
        """Create an empty result with zero scores."""
        return cls(
            answer_relevancy=0.0,
            faithfulness=0.0,
            context_precision=0.0 if include_precision else None
        )

class RAGEvaluator:
    """Enhanced evaluator class for RAG system using RAGAS metrics."""
    
    def __init__(self, config: EvaluationConfig):
        """Initialize the RAG evaluator with configuration."""
        self.config = config
        self._setup_environment()
        self._initialize_metrics()
        self._initialize_evaluator()
        
    def _setup_environment(self) -> None:
        """Set up environment variables and API keys."""
        try:
            if self.config.reuse_same_model:
                # Initialize evaluator LLM using manager with the same config as RAG agent
                os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
                evaluator_llm = ChatGroq(
                        model_name=self.config.model_name,
                        temperature=0.5
                )
                self.evaluator = evaluator_llm
                os.environ["OPENAI_API_KEY"] = "DUMMY"
            else:
                load_dotenv(override=True)
                os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        except Exception as e:
            logger.error(f"Environment setup failed: {str(e)}")
            raise

    def _initialize_metrics(self) -> None:
        """Initialize evaluation metrics."""
        self.base_metrics: TypeList[Any] = [
            answer_relevancy,
            faithfulness
        ]
        self.ground_truth_metrics: TypeList[Any] = [
            *self.base_metrics,
            context_precision
        ]

    def _initialize_evaluator(self) -> None:
        """Initialize the LLM evaluator if needed."""
        pass  # Initialization now happens in _setup_environment

    @staticmethod
    def _validate_inputs(
        questions: Sequence[str],
        contexts: Sequence[Sequence[str]],
        answers: Sequence[str],
        ground_truths: Optional[Sequence[str]] = None
    ) -> None:
        """Validate input parameters."""
        if not questions or not contexts or not answers:
            raise ValueError("Questions, contexts, and answers cannot be empty")
        
        if len(questions) != len(answers):
            raise ValueError("Number of questions must match number of answers")
            
        if ground_truths and len(ground_truths) != len(questions):
            raise ValueError("Number of ground truths must match number of questions")

    @staticmethod
    def _format_context(
        context: Union[str, List[str], List[List[str]]]
    ) -> List[List[str]]:
        """Format context into required structure with validation."""
        if not context:
            return [[""]]
            
        if isinstance(context, str):
            return [[context]]
            
        if isinstance(context[0], str):
            return [context]
            
        return context

    @lru_cache(maxsize=128)
    def _evaluate_batch(
        self,
        questions: tuple,
        contexts: tuple,
        answers: tuple,
        ground_truths: Optional[tuple] = None
    ) -> Dict[str, float]:
        """Evaluate a batch of queries with caching."""
        try:
            # Convert tuples back to lists
            questions_list = list(questions)
            contexts_list = [list(c) for c in contexts]
            answers_list = list(answers)
            ground_truths_list = list(ground_truths) if ground_truths else None

            # Prepare dataset
            eval_dataset = Dataset.from_dict({
                "question": questions_list,
                "answer": answers_list,
                "contexts": contexts_list,
                **({"ground_truth": ground_truths_list} if ground_truths_list else {})
            })

            # Select metrics
            metrics = self.ground_truth_metrics if ground_truths else self.base_metrics

            # Run evaluation
            if self.config.reuse_same_model:
                results = evaluate(
                    eval_dataset,
                    metrics=metrics,
                    llm=self.evaluator
                )
            else:
                results = evaluate(
                    eval_dataset,
                    metrics=metrics
                )

            return self._process_results(results, bool(ground_truths))

        except Exception as e:
            logger.error(f"Batch evaluation failed: {str(e)}")
            return self._create_empty_results(bool(ground_truths))

    def _process_results(
        self,
        results: Dict[str, Any],
        include_precision: bool
    ) -> Dict[str, float]:
        """Process and convert evaluation results to proper format."""
        try:
            processed_results = {
                "answer_relevancy": self._convert_to_float(results["answer_relevancy"]),
                "faithfulness": self._convert_to_float(results["faithfulness"])
            }
            
            if include_precision:
                processed_results["context_precision"] = self._convert_to_float(
                    results["context_precision"]
                )
                
            return processed_results
            
        except Exception as e:
            logger.error(f"Results processing failed: {str(e)}")
            return self._create_empty_results(include_precision)

    @staticmethod
    def _convert_to_float(value: Any) -> float:
        """Convert various numeric types to float with validation."""
        try:
            if isinstance(value, (np.ndarray, list)):
                value = value[0] if len(value) > 0 else 0.0
            if isinstance(value, (np.float32, np.float64)):
                return float(value)
            return float(value)
        except (TypeError, ValueError):
            logger.warning(f"Failed to convert value to float: {value}")
            return 0.0

    @staticmethod
    def _create_empty_results(include_precision: bool) -> Dict[str, float]:
        """Create empty results dictionary."""
        results = {
            "answer_relevancy": 0.0,
            "faithfulness": 0.0
        }
        if include_precision:
            results["context_precision"] = 0.0
        return results

    def evaluate_queries(
        self,
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple queries in batches with validation and error handling.
        
        Args:
            questions: List of query questions
            contexts: List of context passages for each question
            answers: List of generated answers
            ground_truths: Optional list of ground truth answers
            
        Returns:
            List of EvaluationResult objects containing scores
        """
        try:
            # Validate inputs
            self._validate_inputs(questions, contexts, answers, ground_truths)
            
            # Format contexts
            formatted_contexts = [self._format_context(ctx) for ctx in contexts]
            
            # Process in batches
            results = []
            for i in range(0, len(questions), self.config.batch_size):
                batch_end = min(i + self.config.batch_size, len(questions))
                
                # Convert lists to tuples for caching
                batch_results = self._evaluate_batch(
                    tuple(questions[i:batch_end]),
                    tuple(tuple(ctx) for ctx in formatted_contexts[i:batch_end]),
                    tuple(answers[i:batch_end]),
                    tuple(ground_truths[i:batch_end]) if ground_truths else None
                )
                
                results.append(
                    EvaluationResult(
                        answer_relevancy=batch_results["answer_relevancy"],
                        faithfulness=batch_results["faithfulness"],
                        context_precision=batch_results.get("context_precision")
                    )
                )
                
            return results
            
        except Exception as e:
            logger.error(f"Query evaluation failed: {str(e)}")
            return [EvaluationResult.create_empty(bool(ground_truths))]

def evaluate_single_query(
    question: str,
    answer: str,
    contexts: List[str],
    resuse_same_model_for_eval: bool,
    ground_truth: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate a single query response with improved error handling.
    
    Args:
        model_provider (str): The LLM provider (e.g., "openai", "claude", "gemini", "ollama")
        model_name (Optional[str]): Specific model name to use
        question: Query question
        answer: Generated answer
        contexts: Retrieved context passages
        resuse_same_model_for_eval: Whether to reuse same model for evaluation
        ground_truth: Optional ground truth answer
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation scores
    """
    try:
        # Initialize evaluator with configuration
        config = EvaluationConfig(
            reuse_same_model=resuse_same_model_for_eval
        )
        evaluator = RAGEvaluator(config)

        # Prepare evaluation data
        eval_dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            **({"ground_truth": [ground_truth]} if ground_truth else {})
        })

        # Select metrics based on ground truth availability
        metrics = evaluator.ground_truth_metrics if ground_truth else evaluator.base_metrics
        
        if resuse_same_model_for_eval: # Run eval when selected reusing same model
            results = evaluate(
                eval_dataset,
                metrics=metrics,
                llm=evaluator.evaluator
            )
        else: # Run eval when using OpenAI API
            results = evaluate(
                eval_dataset,
                metrics=metrics
            )

        # Process and return results
        return evaluator._process_results(results, bool(ground_truth))
        
    except Exception as e:
        raise e
        #logger.error(f"Single query evaluation failed: {str(e)}")
        #return evaluator._create_empty_results(bool(ground_truth))

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

