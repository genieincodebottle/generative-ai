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

import streamlit as st
import json
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

        state["extracted_data"]["ai_summary"] = response.content
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

def process_document(content: str, filename: str, filetype: str, output_fmt: str, llm_provider: str, model: str, temperature: float = 0.1, ollama_base_url: str = None):
    """Process document through the pipeline"""

    # Initialize document state
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
        "next_action": "parse_document"
    }

    with st.spinner("📄 Processing document through pipeline..."):
        try:
            # Create LLM and graph with configuration
            llm_kwargs = {
                'temperature': temperature
            }
            if llm_provider == "Ollama" and ollama_base_url:
                llm_kwargs['base_url'] = ollama_base_url
            llm = create_document_llm(llm_provider, model, **llm_kwargs)
            doc_graph = create_document_processing_graph(llm)

            # Execute processing pipeline
            config = {"configurable": {"thread_id": f"doc_{datetime.now().timestamp()}"}}
            result = doc_graph.invoke(initial_state, config)

            # Display results
            st.success("✅ Document processing completed!")

            # Results tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "📊 Analysis", "✅ Validation", "📄 Full Results"])

            with tab1:
                st.markdown("### 📋 Document Summary")
                if "ai_summary" in result["extracted_data"]:
                    st.markdown(result["extracted_data"]["ai_summary"])
                else:
                    st.info("Summary not available")

                # Key metrics
                col1, col2, col3 = st.columns(3)
                metadata = result["document_metadata"]

                with col1:
                    st.metric("File Size", f"{metadata.get('file_size', 0)} bytes")
                    st.metric("Document Type", metadata.get('document_type', 'Unknown'))

                with col2:
                    st.metric("Word Count", metadata.get('word_count', 0))
                    st.metric("Processing Stage", result.get('processing_stage', 'Unknown'))

                with col3:
                    validation = result.get("validation_results", {})
                    st.metric("Quality Score", f"{validation.get('overall_score', 0):.2f}")
                    st.metric("Completeness", f"{validation.get('completeness_score', 0):.2f}")

            with tab2:
                st.markdown("### 📊 Content Analysis")
                analysis = result["extracted_data"].get("content_analysis", {})

                if analysis:
                    # Keywords
                    if "top_keywords" in analysis:
                        st.markdown("**Top Keywords:**")
                        for keyword, count in analysis["top_keywords"][:10]:
                            st.write(f"• {keyword} ({count} times)")

                    # Entities
                    if "entities" in analysis:
                        st.markdown("**Detected Entities:**")
                        for entity_type, entities in analysis["entities"].items():
                            if entities:
                                st.write(f"**{entity_type.title()}:** {', '.join(entities)}")

                    # Sentiment
                    if "sentiment" in analysis:
                        sentiment = analysis["sentiment"]
                        sentiment_emoji = {"positive": "😊", "neutral": "😐", "negative": "😔"}.get(sentiment, "🤔")
                        st.metric("Sentiment", f"{sentiment_emoji} {sentiment.title()}")

            with tab3:
                st.markdown("### ✅ Validation Results")
                validation = result.get("validation_results", {})

                if validation:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Data Quality", f"{validation.get('data_quality_score', 0):.2f}")
                        st.metric("Completeness", f"{validation.get('completeness_score', 0):.2f}")

                    with col2:
                        if validation.get("validation_warnings"):
                            st.markdown("**Warnings:**")
                            for warning in validation["validation_warnings"]:
                                st.warning(warning)

                        if validation.get("recommendations"):
                            st.markdown("**Recommendations:**")
                            for rec in validation["recommendations"]:
                                st.info(rec)

            with tab4:
                st.markdown("### 📄 Full Processing Results")

                # Formatted output
                if "formatted_output" in result["extracted_data"]:
                    st.markdown("**Formatted Output:**")
                    st.code(result["extracted_data"]["formatted_output"], language=output_fmt)

                # Processing errors
                if result["processing_errors"]:
                    st.markdown("**Processing Errors:**")
                    for error in result["processing_errors"]:
                        st.error(error)

                # Save and download option
                if "formatted_output" in result["extracted_data"]:
                    output_content = result["extracted_data"]["formatted_output"]

                    # Save to output directory
                    dirs = setup_langgraph_directories()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"processed_{filename.split('.')[0]}_{timestamp}.{output_fmt}"
                    output_path = dirs["output"] / output_filename

                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(output_content)

                    st.success(f"✅ Results saved to: {output_filename}")

                    st.download_button(
                        label="📥 Download Results",
                        data=output_content,
                        file_name=f"processed_{filename.split('.')[0]}.{output_fmt}",
                        mime=f"text/{output_fmt}"
                    )

        except Exception as e:
            st.error(f"Document processing error: {str(e)}")

def render_document_processing_interface():
    """Render the Streamlit interface for Document Processing Pipeline"""
    st.header("📄 Document Processing Pipeline")

    # Sidebar configuration
    with st.sidebar:
        st.subheader("⚙️ Configuration")

        # LLM Selection
        llm_provider = st.selectbox(
            "LLM Provider",
            ["Gemini", "Ollama", "Groq", "Anthropic", "OpenAI"],
            key='doc_llm_provider'
        )

        model_options = {
            # Ollama: Free Open Models, runs on your local system (no API key required)
            "Ollama": ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "gemma2:2b", "gemma2:9b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "codestral:22b", "deepseek-coder:1.3b"],
            # Gemini: Google's Gemini models (requires API key)
            "Gemini": ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.0-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
            # Groq: Open Models (requires API key)
            "Groq": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "openai/gpt-oss-120b"],
            # Anthropic: Claude AI models, good at reasoning & Coding (requires API key)
            "Anthropic": ["claude-sonnet-4-20250514", "claude-opus-4-1-20250805", "claude-opus-4-20250514", "claude-3-7-sonnet-latest", "claude-3-5-haiku-latest"],
            # OpenAI: ChatGPT and GPT models, good at reasoning(requires API key)
            "OpenAI": ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-5", "gpt-5-mini", "gpt-5-nano"]
        }

        model = st.selectbox(
            "Model",
            model_options[llm_provider],
            key='doc_model'
        )

        # Ollama-specific configuration
        ollama_base_url = None
        if llm_provider == "Ollama":
            st.markdown("**🔧 Ollama Configuration**")
            ollama_base_url = st.text_input(
                "Ollama Base URL",
                value="http://localhost:11434",
                key='doc_ollama_url',
                help="URL where Ollama server is running"
            )

            # Check Ollama status
            try:
                import requests
                response = requests.get(f"{ollama_base_url}/api/tags", timeout=3)
                if response.status_code == 200:
                    st.success("✅ Ollama server is running")
                else:
                    st.error("❌ Ollama server not accessible")
            except Exception as e:
                st.error("❌ Cannot connect to Ollama server")
                st.markdown("**Setup Instructions:**")
                st.code(f"1. Install Ollama from https://ollama.com\n2. Run: ollama serve\n3. Pull model: ollama pull {model}")




    # Main interface
    # Model Settings (always visible)
    col1, col2 = st.columns(2)

    with col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            key='doc_temperature',
            help="Controls analysis precision (low for accuracy)"
        )

    with col2:
        output_fmt = st.selectbox(
            "Output Format",
            ["markdown", "json", "text"],
            key='output_format',
            help="Format for processed results"
        )

    # Advanced Configuration Section
    with st.expander("⚙️ Advanced Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🎛️ Additional Model Settings**")

        with col2:
            st.markdown("**📄 Processing Settings**")
            processing_depth = st.selectbox(
                "Processing Depth",
                ["Quick", "Standard", "Thorough"],
                index=1,
                key='processing_depth',
                help="Level of analysis detail"
            )

            extract_metadata = st.checkbox(
                "Extract Metadata",
                value=True,
                key='extract_metadata',
                help="Extract document metadata and structure"
            )

        with col3:
            st.markdown("**🔍 Analysis Options**")
            content_validation = st.checkbox(
                "Content Validation",
                value=True,
                key='content_validation',
                help="Validate and clean extracted content"
            )

            enable_summary = st.checkbox(
                "Generate Summary",
                value=True,
                key='enable_summary',
                help="Generate document summary"
            )

    # File upload options
    tab1, tab2 = st.tabs(["📁 File Upload", "✏️ Text Input"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['txt', 'csv', 'json'],
            key='doc_upload'
        )

        if uploaded_file is not None:
            file_content = uploaded_file.read().decode('utf-8')
            file_name = uploaded_file.name
            file_type = uploaded_file.name.split('.')[-1].lower()

            st.success(f"✅ File uploaded: {file_name} ({len(file_content)} characters)")

            if st.button("Process Document", type="primary", key='process_upload'):
                process_document(file_content, file_name, file_type, output_fmt, llm_provider, model, temperature, ollama_base_url)

    with tab2:
        col1, col2 = st.columns([3, 1])

        with col1:
            text_content = st.text_area(
                "Enter document content",
                placeholder="Paste your document content here...",
                height=200,
                key='text_content'
            )

        with col2:
            file_name = st.text_input("File Name", value="document.txt", key='text_filename')
            file_type = st.selectbox("Content Type", ["txt", "csv", "json"], key='text_filetype')

        if text_content and st.button("Process Text", type="primary", key='process_text'):
            process_document(text_content, file_name, file_type, output_fmt, llm_provider, model, temperature, ollama_base_url)

    # Processing pipeline info
    with st.expander("ℹ️ Document Processing Features", expanded=False):
        st.markdown("""
        **Advanced Document Processing Pipeline:**

        📤 **Multi-Format Support**
        - Text documents (TXT)
        - Comma-separated values (CSV)
        - JSON data files
        - Extensible for PDF, DOCX, and more

        🧠 **Intelligent Analysis**
        - Content extraction and parsing
        - Keyword and entity detection
        - Sentiment analysis
        - Document classification

        ✅ **Quality Assurance**
        - Data validation and quality scoring
        - Completeness assessment
        - Error detection and reporting
        - Automated quality recommendations

        📄 **Flexible Output**
        - JSON for structured data
        - Markdown for readable reports
        - CSV for data analysis
        - Downloadable results

        🔄 **Robust Processing**
        - Error handling and recovery
        - Stage-by-stage processing
        - Quality checkpoints
        - Detailed processing logs
        """)

if __name__ == "__main__":
    render_document_processing_interface()