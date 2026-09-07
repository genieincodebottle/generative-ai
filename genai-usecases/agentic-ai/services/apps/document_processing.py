"""
LangGraph - Document Processing Pipeline
Features:
- Multi-format document processing (PDF, DOCX, TXT, CSV)
- Information extraction and classification
- Content summarization and analysis
- Data validation and quality checks
- Structured output generation
- Batch processing capabilities
- Error handling and recovery workflows
"""

import json
import logging
import os
from pathlib import Path
from typing import Annotated, TypedDict, Literal, List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# Pydantic for structured outputs
from pydantic import BaseModel, Field
from services.llm_text import message_text

logger = logging.getLogger(__name__)

load_dotenv()

def setup_langgraph_directories():
    """Create necessary directories for LangGraph operations"""
    base_dir = Path(__file__).parent
    input_temp_dir = base_dir / "input" / "temp"
    input_temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return {"input_temp": input_temp_dir, "output": output_dir, "base": base_dir}

# State Management for Document Processing
class DocumentState(TypedDict):
    messages: Annotated[list, add_messages]
    document_content: str
    document_type: str
    file_name: str
    processing_stage: str
    extracted_data: Dict[str, Any]
    document_metadata: Dict[str, Any]
    validation_results: Dict[str, Any]
    processing_errors: List[str]
    output_format: str
    next_action: str

# Data Models
class DocumentMetadata(BaseModel):
    """Document metadata structure"""
    file_name: str = Field(description="Original file name")
    file_size: int = Field(description="File size in bytes")
    document_type: Literal["pdf", "docx", "txt", "csv", "json", "unknown"] = Field(description="Document type")
    page_count: Optional[int] = Field(description="Number of pages")
    word_count: Optional[int] = Field(description="Word count")
    creation_date: Optional[str] = Field(description="Document creation date")
    language: str = Field(default="english", description="Detected language")

class ExtractedInformation(BaseModel):
    """Extracted information structure"""
    title: Optional[str] = Field(description="Document title")
    summary: str = Field(description="Document summary")
    key_topics: List[str] = Field(description="Main topics covered")
    entities: Dict[str, List[str]] = Field(description="Named entities by type")
    key_phrases: List[str] = Field(description="Important phrases")
    sentiment: Literal["positive", "neutral", "negative"] = Field(description="Overall sentiment")
    confidence_score: float = Field(ge=0, le=1, description="Extraction confidence")

# Document Processing Tools
@tool
def document_parser(file_content: str, file_type: str, file_name: str) -> str:
    """Parse document content based on file type"""
    try:
        if file_type == "txt":
            return file_content
        elif file_type == "csv":
            # Simulate CSV parsing
            lines = file_content.split('\\n')[:10]  # First 10 lines for demo
            parsed_data = {
                "headers": lines[0].split(',') if lines else [],
                "row_count": len(lines) - 1,
                "sample_rows": lines[1:6] if len(lines) > 1 else [],
                "columns": len(lines[0].split(',')) if lines else 0
            }
            return json.dumps(parsed_data, indent=2)
        elif file_type == "json":
            try:
                parsed_json = json.loads(file_content)
                return json.dumps(parsed_json, indent=2)
            except json.JSONDecodeError:
                return "Invalid JSON format"
        else:
            # For PDF, DOCX - simulate extraction
            return f"Extracted text content from {file_type.upper()} file: {file_name}\\n\\n{file_content[:1000]}..."

    except Exception as e:
        return f"Document parsing error: {str(e)}"

@tool
def content_analyzer(text_content: str) -> str:
    """Analyze document content for key information"""
    try:
        # Simulate content analysis
        words = text_content.split()
        word_count = len(words)

        # Simple keyword extraction (in production, use NLP libraries)
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were']
        keywords = [word.lower().strip('.,!?') for word in words if word.lower() not in common_words and len(word) > 3]
        keyword_freq = {}
        for keyword in keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1

        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        # Detect entities (simplified)
        entities = {
            "organizations": [word for word in words if word.isupper() and len(word) > 2][:5],
            "dates": [word for word in words if any(char.isdigit() for char in word) and len(word) < 12][:5],
            "locations": []  # Would use NER in production
        }

        # Sentiment analysis (simplified)
        positive_words = ['good', 'excellent', 'great', 'positive', 'success', 'benefit', 'advantage']
        negative_words = ['bad', 'poor', 'negative', 'problem', 'issue', 'error', 'failure']

        positive_count = sum(1 for word in words if word.lower() in positive_words)
        negative_count = sum(1 for word in words if word.lower() in negative_words)

        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        analysis = {
            "word_count": word_count,
            "top_keywords": top_keywords,
            "entities": entities,
            "sentiment": sentiment,
            "readability_score": min(100, max(0, 100 - (word_count / 100))),
            "content_density": len(set(keywords)) / max(1, len(keywords)),
            "summary_length_recommendation": max(100, word_count // 10)
        }

        return json.dumps(analysis, indent=2)

    except Exception as e:
        return f"Content analysis error: {str(e)}"

@tool
def data_validator(extracted_data: str, validation_rules: str = "standard") -> str:
    """Validate extracted data quality and completeness"""
    try:
        validation_results = {
            "data_quality_score": 0.0,
            "completeness_score": 0.0,
            "validation_errors": [],
            "validation_warnings": [],
            "recommendations": []
        }

        try:
            data = json.loads(extracted_data)
        except json.JSONDecodeError:
            data = {"raw_content": extracted_data}

        # Check data completeness
        required_fields = ["word_count", "top_keywords", "sentiment"]
        present_fields = [field for field in required_fields if field in data]
        completeness_score = len(present_fields) / len(required_fields)

        validation_results["completeness_score"] = completeness_score

        # Check data quality
        quality_issues = 0

        if "word_count" in data:
            if data["word_count"] < 10:
                validation_results["validation_warnings"].append("Very short document (< 10 words)")
                quality_issues += 1
            elif data["word_count"] > 10000:
                validation_results["validation_warnings"].append("Very long document (> 10,000 words)")

        if "top_keywords" in data:
            if len(data["top_keywords"]) < 3:
                validation_results["validation_warnings"].append("Few keywords extracted")
                quality_issues += 1

        if "entities" in data:
            total_entities = sum(len(ent_list) for ent_list in data["entities"].values())
            if total_entities == 0:
                validation_results["validation_warnings"].append("No entities detected")

        # Calculate quality score
        validation_results["data_quality_score"] = max(0.0, 1.0 - (quality_issues * 0.2))

        # Generate recommendations
        if completeness_score < 0.8:
            validation_results["recommendations"].append("Consider re-processing with different extraction parameters")

        if validation_results["data_quality_score"] < 0.7:
            validation_results["recommendations"].append("Review document format and content quality")

        validation_results["overall_score"] = (completeness_score + validation_results["data_quality_score"]) / 2

        return json.dumps(validation_results, indent=2)

    except Exception as e:
        return f"Data validation error: {str(e)}"

@tool
def format_converter(data: str, output_format: str) -> str:
    """Convert processed data to specified output format"""
    try:
        # Parse input data
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError:
            parsed_data = {"content": data}

        if output_format == "json":
            return json.dumps(parsed_data, indent=2)

        elif output_format == "markdown":
            # Extract data from nested structure
            content_analysis = parsed_data.get('content_analysis', {})
            ai_summary = parsed_data.get('ai_summary', 'No summary available')

            # Get word count
            word_count = content_analysis.get('word_count', 'N/A')

            # Get sentiment
            sentiment = content_analysis.get('sentiment', 'N/A')

            # Get top keywords
            top_keywords = content_analysis.get('top_keywords', [])
            keywords_text = '\n'.join(f"- {kw[0]} ({kw[1]} occurrences)" for kw in top_keywords[:5]) if top_keywords else "No keywords found"

            # Get entities
            entities = content_analysis.get('entities', {})
            entities_text = '\n'.join(f"**{ent_type.title()}:** {', '.join(ent_list)}" for ent_type, ent_list in entities.items() if ent_list) if entities else "No entities found"

            # Get quality score from validation results if available
            quality_score = parsed_data.get('data_quality_score', 'N/A')

            markdown_output = f"""# Document Processing Results

## Summary
{ai_summary}

## Key Information
- **Word Count:** {word_count}
- **Sentiment:** {sentiment}
- **Quality Score:** {quality_score}
- **Readability:** {content_analysis.get('readability_score', 'N/A')}

## Top Keywords
{keywords_text}

## Entities
{entities_text}

## Processing Details
- **Content Density:** {content_analysis.get('content_density', 'N/A')}
- **Recommended Summary Length:** {content_analysis.get('summary_length_recommendation', 'N/A')} characters
"""
            return markdown_output

        elif output_format == "csv":
            # Convert to CSV format
            csv_lines = ["Field,Value"]
            for key, value in parsed_data.items():
                if isinstance(value, (str, int, float)):
                    csv_lines.append(f"{key},{value}")
            return "\\n".join(csv_lines)

        else:
            return str(parsed_data)

    except Exception as e:
        return f"Format conversion error: {str(e)}"

# Document Processing Functions
def create_document_llm(provider: str, model: str, **kwargs):
    """Create LLM instance for document processing with configurable parameters"""
    temperature = kwargs.get('temperature', 0.1)

    if provider == "Ollama":
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=kwargs.get('base_url', "http://localhost:11434"),
            timeout=kwargs.get('timeout', 120)
        )
    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature
        )
    elif provider == "Groq":
        return ChatGroq(
            model=model,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature
        )
    elif provider == "Anthropic":
        return ChatAnthropic(
            model=model,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=temperature
        )
    elif provider == "OpenAI":
        return ChatOpenAI(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature
        )

def document_intake_node(state: DocumentState) -> DocumentState:
    """Initial document intake and metadata extraction"""
    try:
        # Extract basic metadata
        content = state["document_content"]
        file_name = state["file_name"]
        doc_type = state["document_type"]

        metadata = {
            "file_name": file_name,
            "file_size": len(content.encode('utf-8')),
            "document_type": doc_type,
            "word_count": len(content.split()) if content else 0,
            "processing_timestamp": datetime.now().isoformat(),
            "language": "english"  # Default - could use language detection
        }

        state["document_metadata"] = metadata
        state["processing_stage"] = "intake_complete"
        state["next_action"] = "parse_document"

        return state

    except Exception as e:
        state["processing_errors"].append(f"Intake error: {str(e)}")
        state["next_action"] = "error_handler"
        return state

def document_parser_node(state: DocumentState) -> DocumentState:
    """Parse document content"""
    try:
        content = state["document_content"]
        doc_type = state["document_type"]
        file_name = state["file_name"]

        # Parse document using tool
        parsed_result = document_parser.invoke({
            "file_content": content,
            "file_type": doc_type,
            "file_name": file_name
        })

        state["extracted_data"]["parsed_content"] = parsed_result
        state["processing_stage"] = "parsing_complete"
        state["next_action"] = "analyze_content"

        return state

    except Exception as e:
        state["processing_errors"].append(f"Parsing error: {str(e)}")
        state["next_action"] = "error_handler"
        return state

def content_analysis_node(state: DocumentState) -> DocumentState:
    """Analyze document content for insights"""
    try:
        content = state["document_content"]

        # Analyze content using tool
        analysis_result = content_analyzer.invoke({"text_content": content})
        analysis_data = json.loads(analysis_result)

        state["extracted_data"]["content_analysis"] = analysis_data
        state["processing_stage"] = "analysis_complete"
        state["next_action"] = "validate_data"

        return state

    except Exception as e:
        state["processing_errors"].append(f"Analysis error: {str(e)}")
        state["next_action"] = "error_handler"
        return state

def data_validation_node(state: DocumentState) -> DocumentState:
    """Validate extracted data quality"""
    try:
        extracted_data = json.dumps(state["extracted_data"])

        # Validate data using tool
        validation_result = data_validator.invoke({
            "extracted_data": extracted_data,
            "validation_rules": "standard"
        })

        validation_data = json.loads(validation_result)
        state["validation_results"] = validation_data
        state["processing_stage"] = "validation_complete"

        # Decide next action based on validation results
        if validation_data["overall_score"] < 0.5:
            state["next_action"] = "quality_improvement"
        else:
            state["next_action"] = "generate_summary"

        return state

    except Exception as e:
        state["processing_errors"].append(f"Validation error: {str(e)}")
        state["next_action"] = "error_handler"
        return state

def summary_generator(state: DocumentState, llm) -> DocumentState:
    """Generate document summary and insights"""
    try:
        content = state["document_content"]
        analysis = state["extracted_data"].get("content_analysis", {})

        summary_prompt = f"""
        Analyze this document and provide a comprehensive summary:

        Document Content: {content[:2000]}...

        Content Analysis: {json.dumps(analysis, indent=2)}

        Please provide:
        1. A concise summary (2-3 paragraphs)
        2. Key insights and findings
        3. Main topics and themes
        4. Notable entities or important information
        5. Overall assessment of document quality and usefulness

        Format your response clearly and professionally.
        """

        response = llm.invoke([
            SystemMessage(content="You are an expert document analyst and summarizer."),
            HumanMessage(content=summary_prompt)
        ])

        state["extracted_data"]["ai_summary"] = message_text(response)
        state["processing_stage"] = "summary_complete"
        state["next_action"] = "format_output"

        return state

    except Exception as e:
        state["processing_errors"].append(f"Summary generation error: {str(e)}")
        state["next_action"] = "error_handler"
        return state

def output_formatter_node(state: DocumentState) -> DocumentState:
    """Format output according to specified format"""
    try:
        data_to_format = json.dumps(state["extracted_data"])
        output_format = state["output_format"]

        # Format output using tool
        formatted_output = format_converter.invoke({
            "data": data_to_format,
            "output_format": output_format
        })

        state["extracted_data"]["formatted_output"] = formatted_output
        state["processing_stage"] = "formatting_complete"
        state["next_action"] = "complete"

        return state

    except Exception as e:
        state["processing_errors"].append(f"Formatting error: {str(e)}")
        state["next_action"] = "error_handler"
        return state

def error_handler_node(state: DocumentState) -> DocumentState:
    """Handle processing errors"""
    state["processing_stage"] = "error"
    state["next_action"] = "complete"

    error_summary = {
        "error_count": len(state["processing_errors"]),
        "errors": state["processing_errors"],
        "partial_results": state["extracted_data"],
        "recovery_suggestions": [
            "Check document format and content quality",
            "Try different processing parameters",
            "Ensure document is not corrupted"
        ]
    }

    state["extracted_data"]["error_report"] = error_summary
    return state

def processing_router(state: DocumentState) -> str:
    """Route processing based on current state"""
    next_action = state.get("next_action", "parse_document")

    if next_action == "parse_document":
        return "document_parser"
    elif next_action == "analyze_content":
        return "content_analysis"
    elif next_action == "validate_data":
        return "data_validation"
    elif next_action == "generate_summary":
        return "summary_generator"
    elif next_action == "format_output":
        return "output_formatter"
    elif next_action == "error_handler":
        return "error_handler"
    elif next_action == "complete":
        return END
    else:
        return "content_analysis"

def create_document_processing_graph(llm):
    """Create the document processing workflow graph"""

    # Create workflow
    workflow = StateGraph(DocumentState)

    # Add nodes
    workflow.add_node("document_intake", document_intake_node)
    workflow.add_node("document_parser", document_parser_node)
    workflow.add_node("content_analysis", content_analysis_node)
    workflow.add_node("data_validation", data_validation_node)
    workflow.add_node("summary_generator", lambda state: summary_generator(state, llm))
    workflow.add_node("output_formatter", output_formatter_node)
    workflow.add_node("error_handler", error_handler_node)

    # Add edges
    workflow.add_edge(START, "document_intake")
    workflow.add_conditional_edges("document_intake", processing_router)
    workflow.add_conditional_edges("document_parser", processing_router)
    workflow.add_conditional_edges("content_analysis", processing_router)
    workflow.add_conditional_edges("data_validation", processing_router)
    workflow.add_conditional_edges("summary_generator", processing_router)
    workflow.add_conditional_edges("output_formatter", processing_router)
    workflow.add_edge("error_handler", END)

    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def process_document(content: str, filename: str, filetype: str, output_fmt: str,
                     llm_provider: str, model: str, temperature: float = 0.1,
                     ollama_base_url: str | None = None) -> dict:
    """Run a document through the pipeline and return the result.

    The original version of this function ran the graph and then rendered
    about 110 lines of Streamlit tabs, metrics and download buttons inline.
    That made the pipeline impossible to call from anything but a running
    Streamlit session. It now returns the state dict; drawing it is the UI's
    job.
    """
    initial_state = {
        "messages": [],
        "document_content": content,
        "document_type": filetype,
        "file_name": filename,
        "processing_stage": "initialized",
        "extracted_data": {},
        "document_metadata": {},
        "validation_results": {},
        "processing_errors": [],
        "output_format": output_fmt,
        "next_action": "parse_document",
    }

    llm_kwargs = {"temperature": temperature}
    if llm_provider == "Ollama" and ollama_base_url:
        llm_kwargs["base_url"] = ollama_base_url

    llm = create_document_llm(llm_provider, model, **llm_kwargs)
    graph = create_document_processing_graph(llm)

    config = {"configurable": {"thread_id": f"doc_{datetime.now().timestamp()}"}}
    return graph.invoke(initial_state, config)


def save_output(result: dict, filename: str, output_fmt: str) -> str | None:
    """Write the formatted output to the output directory, if there is any."""
    formatted = result.get("extracted_data", {}).get("formatted_output")
    if not formatted:
        return None
    dirs = setup_langgraph_directories()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"processed_{filename.split('.')[0]}_{stamp}.{output_fmt}"
    path = dirs["output"] / name
    path.write_text(formatted, encoding="utf-8")
    return name
