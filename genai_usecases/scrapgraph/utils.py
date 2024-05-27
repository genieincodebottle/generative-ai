from scrapegraphai.graphs import SmartScraperGraph, SearchGraph, SpeechGraph
from scrapegraphai.utils import convert_to_csv, convert_to_json, prettify_exec_info

import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(filename='app.log', encoding='utf-8', level=logging.INFO)


openai_key = os.getenv("OPENAI_APIKEY")
gemini_pro_key = os.getenv("GEMINI_PRO_API_KEY")

def smart_scraper_graph(prompt, source, config):
    smart_scraper_graph = SmartScraperGraph(
        prompt=prompt,
        source=source,
        config=config
    )
    result = smart_scraper_graph.run()
    # Run the graph
    result = smart_scraper_graph.run()
    graph_exec_info = smart_scraper_graph.get_execution_info()
    logger.info(prettify_exec_info(graph_exec_info))  
    result_format(result) 

    return result

def search_graph(prompt, config):
    search_graph = SearchGraph(
        prompt=prompt,
        config=config
    )
    # Run the graph
    result = search_graph.run() 
    graph_exec_info = search_graph.get_execution_info()
    logger.info(prettify_exec_info(graph_exec_info))  

    result_format(result) 
    return result

def speech_graph(prompt, config):
    speech_graph = SpeechGraph(
        prompt=prompt,
        config=config
    )
    # Run the graph
    result = speech_graph.run()
    graph_exec_info = speech_graph.get_execution_info()
    logger.info(prettify_exec_info(graph_exec_info))  

    return result

def result_format(result):
    convert_to_csv(result, "result")
    convert_to_json(result, "result")

def get_ollama_config(model_type, graph_type):
    base_config = {
        "llm": {
            "model": f"ollama/{model_type}",
            "temperature": 0,
            "format": "json",  # Ollama needs the format to be specified explicitly
            "base_url": "http://localhost:11434",  # set Ollama URL
        },
        "embeddings": {
            "model": "ollama/nomic-embed-text",
            "base_url": "http://localhost:11434",  # set Ollama URL
        },
        "verbose": True,
    }
    if graph_type == "SearchGraph":
        base_config["max_results"] = 5
    return base_config

def get_openai_config(model_type, openai_key):
    return {
        "llm": {
            "api_key": openai_key,
            "model": model_type,
        },
        "tts_model": {
            "api_key": openai_key,
            "model": "tts-1",
            "voice": "alloy"
        },
        "output_path": "audio_summary.mp3",
    }

def get_gemini_pro_config(graph_type, gemini_pro_key):
    base_config = {
        "llm": {
            "api_key": gemini_pro_key,
            "model": "gemini-pro",
        },
    }
    if graph_type == "SmartScraperGraph":
        base_config["embeddings"] = {
            "model": "ollama/nomic-embed-text",
            "base_url": "http://localhost:11434",  # set Ollama URL
        }
    elif graph_type == "SearchGraph":
        base_config.update({
            "temperature": 0,
            "streaming": True,
            "max_results": 5,
            "verbose": True,
        })
    return base_config

def main(model, model_type, graph_type, question, source=None):
    print(f"Model: {model}, Model Type: {model_type}, Type: {graph_type}, Question: {question}, Source: {source}")
    try:
        graph_config = None
        if model == "ollama":
            graph_config = get_ollama_config(model_type, graph_type)
        elif model == "OpenAI":
            graph_config = get_openai_config(model_type, openai_key)
        elif model == "gemini-pro":
            graph_config = get_gemini_pro_config(graph_type, gemini_pro_key)
        
        logger.info(f"Graph Config: {graph_config}")

        if graph_type == "SmartScraperGraph":
            result = smart_scraper_graph(question, source, graph_config)
        elif graph_type == "SearchGraph":
            result = search_graph(question, graph_config)
        elif graph_type == "SpeechGraph":
            result = speech_graph(question, graph_config)
        
        return result
    except Exception as e:
        raise e

#if __name__ == "__main__":
#    main('ollama', 'llama3', 'SearchGraph', 'List me all the traditional recipes from Chioggia', None )