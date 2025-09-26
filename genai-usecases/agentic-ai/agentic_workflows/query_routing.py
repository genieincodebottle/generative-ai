"""
Query Routing Workflow Pattern
Intelligent query classification and routing to specialized processors
- Dynamic routing based on query type/intent
- Specialized processor selection
- Conditional workflow execution
- Load balancing and optimization
"""

import asyncio
import streamlit as st
import re
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import os
from dataclasses import dataclass
from enum import Enum

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

load_dotenv()

class QueryType(Enum):
    """Supported query types for routing"""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    RESEARCH = "research"
    CODING = "coding"
    MATH = "math"
    GENERAL = "general"

@dataclass
class RoutingRule:
    """Defines a routing rule for queries"""
    name: str
    description: str
    keywords: List[str]
    patterns: List[str]
    query_type: QueryType
    processor: str
    priority: int = 1

class QueryRouter:
    """Intelligent query routing system"""

    def __init__(self, llm):
        self.llm = llm
        self.routing_rules = self._initialize_routing_rules()
        self.routing_history = []

    def _initialize_routing_rules(self) -> List[RoutingRule]:
        """Initialize default routing rules"""
        return [
            RoutingRule(
                name="Data Analysis",
                description="Statistical analysis, data processing, visualization",
                keywords=["analyze", "statistics", "data", "chart", "graph", "visualization", "trend"],
                patterns=[r"\\b(analyze|analysis|statistical?)\\b", r"\\b(data|dataset)\\b", r"\\b(chart|graph|plot)\\b"],
                query_type=QueryType.ANALYSIS,
                processor="analysis_processor",
                priority=2
            ),
            RoutingRule(
                name="Content Generation",
                description="Creative writing, content creation, copywriting",
                keywords=["write", "create", "generate", "compose", "draft", "content", "article"],
                patterns=[r"\\b(write|create|generate)\\b", r"\\b(article|blog|content)\\b", r"\\b(story|essay)\\b"],
                query_type=QueryType.GENERATION,
                processor="generation_processor",
                priority=2
            ),
            RoutingRule(
                name="Research",
                description="Information gathering, fact-checking, research tasks",
                keywords=["research", "find", "search", "investigate", "study", "explore", "learn"],
                patterns=[r"\\b(research|investigate)\\b", r"\\b(find|search).*\\binformation\\b", r"\\b(study|explore)\\b"],
                query_type=QueryType.RESEARCH,
                processor="research_processor",
                priority=2
            ),
            RoutingRule(
                name="Code Development",
                description="Programming, debugging, code review, technical tasks",
                keywords=["code", "program", "function", "debug", "api", "software", "algorithm"],
                patterns=[r"\\b(code|coding|program)\\b", r"\\b(function|algorithm)\\b", r"\\b(debug|error)\\b"],
                query_type=QueryType.CODING,
                processor="coding_processor",
                priority=3
            ),
            RoutingRule(
                name="Mathematical",
                description="Mathematical calculations, equations, problem solving",
                keywords=["calculate", "solve", "equation", "math", "formula", "compute"],
                patterns=[r"\\b(calculate|compute)\\b", r"\\b(equation|formula)\\b", r"\\b(solve|solution)\\b"],
                query_type=QueryType.MATH,
                processor="math_processor",
                priority=2
            ),
            RoutingRule(
                name="General Query",
                description="General questions and conversations",
                keywords=["help", "explain", "what", "how", "why", "question"],
                patterns=[r"\\b(help|explain)\\b", r"\\b(what|how|why)\\b"],
                query_type=QueryType.GENERAL,
                processor="general_processor",
                priority=1
            )
        ]

    async def classify_query(self, query: str) -> Dict[str, Any]:
        """Classify query using both rule-based and LLM-based approaches"""
        try:
            # Rule-based classification
            rule_scores = self._score_routing_rules(query)

            # LLM-based classification for validation
            llm_classification = await self._llm_classify_query(query)

            # Combine results
            best_rule = max(rule_scores, key=lambda x: x['score']) if rule_scores else None

            classification = {
                "query": query,
                "rule_based_result": best_rule,
                "llm_classification": llm_classification,
                "final_routing": best_rule if best_rule and best_rule['score'] > 0.3 else {
                    "name": "General Query",
                    "query_type": QueryType.GENERAL.value,
                    "processor": "general_processor",
                    "score": 0.5
                },
                "timestamp": datetime.now().isoformat()
            }

            return classification

        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "final_routing": {
                    "name": "General Query",
                    "query_type": QueryType.GENERAL.value,
                    "processor": "general_processor",
                    "score": 0.5
                }
            }

    def _score_routing_rules(self, query: str) -> List[Dict[str, Any]]:
        """Score query against routing rules"""
        query_lower = query.lower()
        scores = []

        for rule in self.routing_rules:
            score = 0

            # Keyword matching
            keyword_matches = sum(1 for keyword in rule.keywords if keyword in query_lower)
            keyword_score = (keyword_matches / len(rule.keywords)) * 0.6

            # Pattern matching
            pattern_matches = sum(1 for pattern in rule.patterns if re.search(pattern, query_lower))
            pattern_score = (pattern_matches / len(rule.patterns)) * 0.4 if rule.patterns else 0

            # Priority bonus
            priority_bonus = rule.priority * 0.1

            total_score = keyword_score + pattern_score + priority_bonus

            if total_score > 0:
                scores.append({
                    "name": rule.name,
                    "query_type": rule.query_type.value,
                    "processor": rule.processor,
                    "score": min(total_score, 1.0),
                    "keyword_matches": keyword_matches,
                    "pattern_matches": pattern_matches
                })

        return sorted(scores, key=lambda x: x['score'], reverse=True)

    async def _llm_classify_query(self, query: str) -> Dict[str, Any]:
        """Use LLM to classify query type"""
        try:
            classification_prompt = f"""
            Classify this query into one of these categories:
            - analysis: Data analysis, statistics, visualization
            - generation: Content creation, writing, creative tasks
            - research: Information gathering, fact-checking, investigation
            - coding: Programming, debugging, technical development
            - math: Mathematical calculations, equations, problem solving
            - general: General questions, conversations, explanations

            Query: "{query}"

            Respond with just the category name and a confidence score (0-1).
            Format: category:confidence
            """

            messages = [
                SystemMessage(content="You are an expert query classifier."),
                HumanMessage(content=classification_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            result = response.content.strip().lower()

            # Parse response
            if ":" in result:
                category, confidence = result.split(":", 1)
                return {
                    "category": category.strip(),
                    "confidence": float(confidence.strip())
                }
            else:
                return {"category": "general", "confidence": 0.5}

        except Exception as e:
            return {"category": "general", "confidence": 0.5, "error": str(e)}

    async def route_and_process(self, query: str) -> Dict[str, Any]:
        """Route query and process with appropriate processor"""
        # Classify the query
        classification = await self.classify_query(query)

        # Get routing decision
        routing = classification['final_routing']
        processor_name = routing['processor']

        # Process with selected processor
        result = await self._execute_processor(query, processor_name, routing)

        # Store routing history
        routing_record = {
            "query": query,
            "classification": classification,
            "processor_used": processor_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

        self.routing_history.append(routing_record)

        return routing_record

    async def _execute_processor(self, query: str, processor_name: str, routing: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the appropriate processor for the query"""
        processors = {
            "analysis_processor": self._analysis_processor,
            "generation_processor": self._generation_processor,
            "research_processor": self._research_processor,
            "coding_processor": self._coding_processor,
            "math_processor": self._math_processor,
            "general_processor": self._general_processor
        }

        processor = processors.get(processor_name, self._general_processor)
        return await processor(query, routing)

    async def _analysis_processor(self, query: str, routing: Dict[str, Any]) -> Dict[str, Any]:
        """Process analysis-related queries"""
        prompt = f"""
        You are a data analysis expert. Analyze this query and provide:
        1. What type of analysis is being requested
        2. What data or information would be needed
        3. Recommended approach or methodology
        4. Expected output format

        Query: {query}
        """

        response = await self.llm.ainvoke([
            SystemMessage(content="You are an expert data analyst."),
            HumanMessage(content=prompt)
        ])

        return {
            "processor": "analysis_processor",
            "response": response.content,
            "specialization": "Data Analysis & Statistics"
        }

    async def _generation_processor(self, query: str, routing: Dict[str, Any]) -> Dict[str, Any]:
        """Process content generation queries"""
        prompt = f"""
        You are a creative content specialist. For this request:
        1. Identify the type of content needed
        2. Suggest content structure and key elements
        3. Provide style and tone recommendations
        4. Create an outline or sample content

        Request: {query}
        """

        response = await self.llm.ainvoke([
            SystemMessage(content="You are an expert content creator and writer."),
            HumanMessage(content=prompt)
        ])

        return {
            "processor": "generation_processor",
            "response": response.content,
            "specialization": "Content Creation & Writing"
        }

    async def _research_processor(self, query: str, routing: Dict[str, Any]) -> Dict[str, Any]:
        """Process research-related queries"""
        prompt = f"""
        You are a research specialist. For this research request:
        1. Identify the research objectives
        2. Suggest research methodology and sources
        3. Outline key areas to investigate
        4. Provide a research plan and timeline

        Research Query: {query}
        """

        response = await self.llm.ainvoke([
            SystemMessage(content="You are an expert researcher and information specialist."),
            HumanMessage(content=prompt)
        ])

        return {
            "processor": "research_processor",
            "response": response.content,
            "specialization": "Research & Investigation"
        }

    async def _coding_processor(self, query: str, routing: Dict[str, Any]) -> Dict[str, Any]:
        """Process coding-related queries"""
        prompt = f"""
        You are a software development expert. For this coding request:
        1. Analyze the technical requirements
        2. Suggest appropriate technologies and approaches
        3. Provide code structure or pseudocode
        4. Include best practices and considerations

        Coding Query: {query}
        """

        response = await self.llm.ainvoke([
            SystemMessage(content="You are an expert software developer and architect."),
            HumanMessage(content=prompt)
        ])

        return {
            "processor": "coding_processor",
            "response": response.content,
            "specialization": "Software Development"
        }

    async def _math_processor(self, query: str, routing: Dict[str, Any]) -> Dict[str, Any]:
        """Process mathematical queries"""
        prompt = f"""
        You are a mathematics expert. For this mathematical query:
        1. Identify the mathematical concepts involved
        2. Provide step-by-step solution approach
        3. Show calculations and reasoning
        4. Verify and explain the result

        Mathematical Query: {query}
        """

        response = await self.llm.ainvoke([
            SystemMessage(content="You are an expert mathematician and problem solver."),
            HumanMessage(content=prompt)
        ])

        return {
            "processor": "math_processor",
            "response": response.content,
            "specialization": "Mathematics & Calculations"
        }

    async def _general_processor(self, query: str, routing: Dict[str, Any]) -> Dict[str, Any]:
        """Process general queries"""
        prompt = f"""
        Provide a helpful and comprehensive response to this query:
        {query}
        """

        response = await self.llm.ainvoke([
            SystemMessage(content="You are a helpful and knowledgeable assistant."),
            HumanMessage(content=prompt)
        ])

        return {
            "processor": "general_processor",
            "response": response.content,
            "specialization": "General Assistant"
        }

def create_llm(provider: str, model: str, base_url: str = None):
    """Create LLM instance based on provider"""
    if provider == "Ollama":
        return ChatOllama(
            model=model,
            base_url=base_url or 'http://localhost:11434',
            temperature=0.3,
            timeout=120
        )
    elif provider == "OpenAI":
        return ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"), temperature=0.3)
    elif provider == "Anthropic":
        return ChatAnthropic(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"), temperature=0.3)
    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(model=model, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.3)
    elif provider == "Groq":
        return ChatGroq(model=model, api_key=os.getenv("GROQ_API_KEY"), temperature=0.3)

def render_query_routing_interface():
    """Render the Streamlit interface for Query Routing workflow"""
    st.header("🎯 Query Routing Workflow")

    # Sidebar configuration
    with st.sidebar:
        st.subheader("⚙️ Configuration")

        # LLM Selection
        llm_provider = st.selectbox(
            "LLM Provider",
            ["Gemini", "Ollama", "Groq", "Anthropic", "OpenAI"],
            key='routing_llm_provider'
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
            key='routing_model'
        )

        # Ollama-specific configuration
        ollama_base_url = None
        if llm_provider == "Ollama":
            st.markdown("**🔧 Ollama Configuration**")
            ollama_base_url = st.text_input(
                "Ollama Base URL",
                value="http://localhost:11434",
                key='routing_ollama_url',
                help="URL where Ollama server is running (no API key required)"
            )

            # Check Ollama status
            try:
                import requests
                response = requests.get(f"{ollama_base_url}/api/tags", timeout=3)
                if response.status_code == 200:
                    st.success("✅ Ollama server is running")

                    # Show available models
                    try:
                        models_data = response.json()
                        if 'models' in models_data and models_data['models']:
                            st.info(f"📋 {len(models_data['models'])} models available")
                        else:
                            st.warning("⚠️ No models found. Pull a model first.")
                    except:
                        st.info("🔄 Ollama server connected")
                else:
                    st.error("❌ Ollama server not accessible")
            except Exception as e:
                st.error("❌ Cannot connect to Ollama server")
                st.markdown("**Setup Instructions:**")
                st.code(f"1. Install Ollama from https://ollama.com\n2. Run: ollama serve\n3. Pull model: ollama pull {model}")

        # API Key Status
        st.markdown("**🔑 API Key Status**")
        if llm_provider == "Ollama":
            st.success("✅ No API key required for Ollama")
        else:
            required_key = f"{llm_provider.upper()}_API_KEY"
            if os.getenv(required_key):
                st.success(f"✅ {required_key} configured")
            else:
                st.warning(f"⚠️ {required_key} not found")

        # Show routing rules
        with st.expander("📋 Routing Rules", expanded=False):
            st.markdown("""
            **Available Processors:**
            - 📊 **Analysis**: Data, statistics, visualization
            - ✍️ **Generation**: Content creation, writing
            - 🔬 **Research**: Information gathering, investigation
            - 💻 **Coding**: Programming, debugging, development
            - 🧮 **Math**: Calculations, equations, problem solving
            - 💬 **General**: General questions and conversations
            """)

    # Debug: Clear session state if needed
    if st.sidebar.button("🔄 Reset Router", help="Clear cached router to fix enum issues"):
        if 'query_router' in st.session_state:
            del st.session_state.query_router
        st.rerun()

    # Initialize router
    if 'query_router' not in st.session_state:
        llm = create_llm(llm_provider, model, ollama_base_url)
        st.session_state.query_router = QueryRouter(llm)

    # Main interface

    # Pattern information
    with st.expander("ℹ️ Query Routing Pattern", expanded=False):
        st.markdown("""
        **Query Routing Workflow Pattern:**

        🎯 **Intelligent Classification**
        - Rule-based keyword and pattern matching
        - LLM based query understanding
        - Confidence scoring and validation

        🔀 **Dynamic Routing**
        - Automatic processor selection
        - Specialized handling for different query types
        - Load balancing and optimization

        🧠 **Specialized Processing**
        - Domain-specific prompt engineering
        - Optimized responses for each query type
        - Context-aware processing

        📊 **Continuous Learning**
        - Routing history tracking
        - Performance monitoring
        - Pattern refinement and optimization

        **Use Cases:**
        - Multi-domain chatbots
        - Intelligent help desks
        - Content management systems
        - API request routing
        """)

    # Query input
    query = st.text_area(
        "Enter your query",
        placeholder="e.g. Analyze the sales data trends for Q3, Write a blog post about AI, Debug this Python function...",
        height=100,
        key='routing_query'
    )

    # Processing options
    col1, col2 = st.columns(2)
    with col1:
        show_classification = st.checkbox("Show Classification Details", value=True)
    with col2:
        show_routing_history = st.checkbox("Show Routing History", value=False)

    # Process query
    if st.button("Route and Process Query", type="primary", key='route_query'):
        if not query:
            st.error("Please enter a query.")
            return

        with st.spinner("🎯 Routing and processing query..."):
            try:
                # Update router LLM if changed
                current_llm = create_llm(llm_provider, model, ollama_base_url)
                st.session_state.query_router.llm = current_llm

                # Route and process
                result = asyncio.run(st.session_state.query_router.route_and_process(query))

                # Display results
                st.success("✅ Query processed successfully!")

                # Results tabs
                tab1, tab2, tab3 = st.tabs(["📋 Response", "🎯 Routing Details", "📊 Classification"])

                with tab1:
                    st.markdown("### 📋 Processed Response")

                    processor_info = result['result']
                    st.info(f"**Processor Used:** {processor_info['specialization']}")
                    st.markdown(processor_info['response'])

                with tab2:
                    if show_classification:
                        st.markdown("### 🎯 Routing Decision")

                        classification = result['classification']
                        routing = classification['final_routing']

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Selected Processor", routing['processor'])
                            st.metric("Confidence Score", f"{routing['score']:.2f}")

                        with col2:
                            st.metric("Query Type", routing.get('query_type', 'Unknown'))
                            st.metric("Rule Name", routing['name'])

                        # Rule-based results
                        if 'rule_based_result' in classification and classification['rule_based_result']:
                            st.markdown("**Rule-Based Classification:**")
                            rule_result = classification['rule_based_result']
                            st.write(f"- **Matches:** {rule_result.get('keyword_matches', 0)} keywords, {rule_result.get('pattern_matches', 0)} patterns")
                            st.write(f"- **Score:** {rule_result['score']:.3f}")

                        # LLM classification
                        if 'llm_classification' in classification:
                            st.markdown("**LLM Classification:**")
                            llm_result = classification['llm_classification']
                            st.write(f"- **Category:** {llm_result.get('category', 'unknown')}")
                            st.write(f"- **Confidence:** {llm_result.get('confidence', 0):.3f}")

                with tab3:
                    st.markdown("### 📊 Classification Analysis")

                    # Show all routing scores
                    if 'rule_based_result' in result['classification']:
                        router = st.session_state.query_router
                        all_scores = router._score_routing_rules(query)

                        if all_scores:
                            st.markdown("**All Processor Scores:**")
                            for score in all_scores[:5]:  # Top 5
                                st.write(f"- **{score['name']}**: {score['score']:.3f} (Keywords: {score['keyword_matches']}, Patterns: {score['pattern_matches']})")

            except Exception as e:
                st.error(f"Processing error: {str(e)}")

    # Routing history
    if show_routing_history and 'query_router' in st.session_state:
        history = st.session_state.query_router.routing_history
        if history:
            st.markdown("### 📚 Routing History")

            for i, record in enumerate(reversed(history[-5:]), 1):  # Last 5 queries
                with st.expander(f"Query {len(history) - i + 1}: {record['query'][:50]}...", expanded=False):
                    st.write(f"**Processor:** {record['processor_used']}")
                    st.write(f"**Timestamp:** {record['timestamp']}")

                    routing = record['classification']['final_routing']
                    st.write(f"**Score:** {routing['score']:.3f}")

                    # Display query type safely
                    query_type = routing.get('query_type', 'Unknown')
                    query_type_str = query_type.value if hasattr(query_type, 'value') else str(query_type)
                    st.write(f"**Query Type:** {query_type_str}")

    

if __name__ == "__main__":
    render_query_routing_interface()