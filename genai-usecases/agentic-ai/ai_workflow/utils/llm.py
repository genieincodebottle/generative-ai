"""
LLM Integration Module
This module provides unified access to various LLM providers including Groq, OpenAI,
Anthropic, and Google's Gemini, with standardized error handling and response formatting.
"""
import os
import logging
from typing import Dict, Any
from datetime import datetime

from groq import Groq
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
# Configure logging
logging.basicConfig(
    level=os.getenv("LOGGING_LEVEL"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_groq_completion(
    prompt: str, 
    temperature: float, 
    model: str = "llama3-8b-8192"
) -> str:
    """
    Make a call to Groq's API with error handling.
    
    Args:
        prompt (str): Input text to send to the model
        temperature (float): Sampling temperature for generation
        model (str, optional): Groq model identifier. Defaults to "llama3-8b-8192"
        
    Returns:
        str: Model response or error message
    """
    logger.debug(f"Making Groq API call with model: {model}")
    
    try:
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
        client = Groq()
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            max_tokens=1024,
            timeout=30
        )
        response = chat_completion.choices[0].message.content
        logger.debug("Groq API call successful")
        logger.debug(f"Response: {response[:100]}... (truncated)")
        return response
    except Exception as e:
        logger.error(f"Error calling Groq: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

def get_openai_completion(
    prompt: str, 
    temperature: float, 
    model: str = "gpt-4o-2024-08-06"
) -> str:
    """
    Make a call to OpenAI's API with error handling.
    
    Args:
        prompt (str): Input text to send to the model
        temperature (float): Sampling temperature for generation
        model (str, optional): OpenAI model identifier. Defaults to "gpt-4o-2024-08-06"
        
    Returns:
        str: Model response or error message
    """
    logger.debug(f"Making OpenAI API call with model: {model}")
    
    try:
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        client = OpenAI()
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            max_tokens=1024
        )
        response = chat_completion.choices[0].message.content
        logger.debug("OpenAI API call successful")
        logger.debug(f"Response: {response[:100]}... (truncated)")
        return response
    except Exception as e:
        logger.error(f"Error calling OpenAI: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

def get_anthropic_completion(
    prompt: str, 
    temperature: float, 
    model: str = "claude-3-5-sonnet-20241022"
) -> str:
    """
    Make a call to Anthropic's API with error handling.
    
    Args:
        prompt (str): Input text to send to the model
        temperature (float): Sampling temperature for generation
        model (str, optional): Anthropic model identifier. Defaults to "claude-3-5-sonnet-20241022"
        
    Returns:
        str: Model response or error message
    """
    logger.debug(f"Making Anthropic API call with model: {model}")
    
    try:
        os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
        client = Anthropic()
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        response = message.content[0].text
        logger.debug("Anthropic API call successful")
        logger.debug(f"Response: {response[:100]}... (truncated)")
        return response
    except Exception as e:
        logger.error(f"Error calling Anthropic: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

def get_gemini_completion(
    prompt: str, 
    temperature: float, 
    model: str = "gemini-2.0-flash-exp"
) -> str:
    """
    Make a call to Google's Gemini API with error handling.
    
    Args:
        prompt (str): Input text to send to the model
        temperature (float): Sampling temperature for generation
        model (str, optional): Gemini model identifier. Defaults to "gemini-2.0-flash-exp"
        
    Returns:
        str: Model response or error message
    """
    logger.debug(f"Making Gemini API call with model: {model}")
    
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(model_name=model)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature
            )
        )
        text_response = response.text
        logger.debug("Gemini API call successful")
        logger.debug(f"Response: {text_response[:100]}... (truncated)")
        return text_response
    except Exception as e:
        logger.error(f"Error calling Gemini: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

# Model configurations with available models and corresponding functions
LLM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Groq": {
        "models": [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "llama3-8b-8192",
            "llama3-70b-8192",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "gemma2-9b-it",
            "mixtral-8x7b-32768"
        ],
        "function": get_groq_completion
    },
    "OpenAI": {
        "models": [
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18"
        ],
        "function": get_openai_completion
    },
    "Anthropic": {
        "models": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229"
        ],
        "function": get_anthropic_completion
    },
    "Gemini": {
        "models": [
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro"
        ],
        "function": get_gemini_completion
    }
}

def llm_call(prompt: str, temperature: float, provider: str, model: str = None) -> str:
    """
    Unified LLM call function that routes to the appropriate provider.
    
    Args:
        prompt (str): Input text to send to the model
        temperature (float): Sampling temperature for generation
        provider (str): Specific Model Provider
        model (str, optional): Specific model identifier.
        
    Returns:
        str: Model response or error message
    """
    logger.debug(f"Processing LLM call for model provider: {provider} and model {model}")
    
    llm_function = LLM_CONFIGS[provider]["function"]
    
    return llm_function(prompt, temperature, model)

def extract_xml(text: str, tag: str) -> str:
    """
    Extract content between XML tags with robust handling.
    
    Args:
        text (str): Input text containing XML tags
        tag (str): Tag name to extract content from
        
    Returns:
        str: Content between tags or empty string if not found properly
    
    Example:
        >>> extract_xml("<name>Llama</name>", "name")
        'Llama'
    """
    logger.debug(f"Extracting content for tag: {tag}")
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start_pos = text.find(start_tag) + len(start_tag)
    end_pos = text.find(end_tag)
    
    content = (
        text[start_pos:end_pos].strip() 
        if start_pos > len(start_tag) - 1 and end_pos != -1 
        else ""
    )
    
    if not content:
        logger.warning(f"No content found for tag: {tag}")
    
    return content