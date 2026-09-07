"""
Query Routing Workflow Pattern
Intelligent query classification and routing to specialized processors
- Dynamic routing based on query type/intent
- Specialized processor selection
- Conditional workflow execution
- Load balancing and optimization
"""

import asyncio
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
from services.llm_text import message_text
from services.providers import create_llm

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

ROUTING_THRESHOLD = 0.3


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
                # Rounded above, so this compares the number a reader
                # expects rather than its floating-point neighbour.
                "final_routing": best_rule if best_rule and best_rule['score'] > ROUTING_THRESHOLD else {
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

            # Priority bonus - only for rules that actually matched
            # something. Applying it unconditionally gave every rule a
            # non-zero score, so on a query that matched nothing the
            # highest-priority rule won by default. A billing question
            # routed to the code-review processor that way, with
            # keyword_matches=0 and pattern_matches=0.
            matched = keyword_matches > 0 or pattern_matches > 0
            priority_bonus = rule.priority * 0.1 if matched else 0.0

            total_score = round(keyword_score + pattern_score + priority_bonus, 6)

            if matched and total_score > 0:
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
            result = message_text(response).strip().lower()

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
            "response": message_text(response),
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
            "response": message_text(response),
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
            "response": message_text(response),
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
            "response": message_text(response),
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
            "response": message_text(response),
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
            "response": message_text(response),
            "specialization": "General Assistant"
        }


