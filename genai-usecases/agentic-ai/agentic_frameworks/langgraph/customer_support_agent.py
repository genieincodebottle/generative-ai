"""
LangGraph - Customer Support Agent System
Features:
- Intelligent conversation flow management
- Multi-tier escalation patterns
- Sentiment analysis and emotion detection
- Knowledge base integration
- Ticket routing and prioritization
- Human handoff capabilities
- Real-time conversation context tracking
"""

import streamlit as st
import json
import os
from pathlib import Path
from typing import Annotated, TypedDict, Literal, List, Dict, Any
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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

# State Management for Customer Support
class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    customer_id: str
    issue_category: str
    priority_level: str
    sentiment_score: float
    escalation_level: int
    resolution_status: str
    conversation_context: Dict[str, Any]
    agent_notes: List[str]
    next_action: str

# Data Models
class CustomerIssue(BaseModel):
    """Customer issue classification"""
    category: Literal["technical", "billing", "account", "product", "general"] = Field(description="Issue category")
    priority: Literal["low", "medium", "high", "urgent"] = Field(description="Issue priority")
    sentiment: Literal["positive", "neutral", "negative", "frustrated"] = Field(description="Customer sentiment")
    urgency: int = Field(ge=1, le=10, description="Urgency score 1-10")
    summary: str = Field(description="Brief issue summary")

class EscalationDecision(BaseModel):
    """Escalation decision model"""
    should_escalate: bool = Field(description="Whether to escalate")
    escalation_reason: str = Field(description="Reason for escalation")
    recommended_department: str = Field(description="Recommended department")
    escalation_notes: str = Field(description="Additional notes for escalation")

# Support Tools
@tool
def knowledge_base_search(query: str, category: str = "general") -> str:
    """Search comprehensive knowledge base for relevant solutions"""
    try:
        # Comprehensive, practical knowledge base
        knowledge_base = {
            "technical": {
                "login_issues": {
                    "solution": "Login Problems Resolution: 1) Verify correct email/username 2) Check caps lock and typing 3) Clear browser cache and cookies 4) Try incognito/private mode 5) Reset password if needed",
                    "follow_up": "If still unable to login, I can help reset your password or check for account lockouts.",
                    "escalation": False
                },
                "password_reset": {
                    "solution": "Password Reset Process: 1) Click 'Forgot Password' on login page 2) Enter registered email address 3) Check email (including spam folder) 4) Click reset link within 24 hours 5) Create new strong password",
                    "follow_up": "Password reset emails are sent within 5 minutes. Let me know if you don't receive it.",
                    "escalation": False
                },
                "app_crashes": {
                    "solution": "App Crash Troubleshooting: 1) Force close and restart app 2) Check for app updates in store 3) Restart your device 4) Free up storage space (need 1GB free) 5) Reinstall app if problem persists",
                    "follow_up": "If crashes continue, I can escalate to our technical team with your device details.",
                    "escalation": True
                },
                "slow_performance": {
                    "solution": "Performance Optimization: 1) Close other apps running in background 2) Check internet connection speed 3) Clear app cache in settings 4) Update to latest app version 5) Restart device daily",
                    "follow_up": "Performance issues can also be network-related. What's your current internet speed?",
                    "escalation": False
                },
                "connectivity": {
                    "solution": "Connection Issues Fix: 1) Check internet connection (try other apps) 2) Switch between WiFi and mobile data 3) Restart router/modem 4) Clear app cache 5) Disable VPN temporarily 6) Check firewall settings",
                    "follow_up": "Let me know which step resolved it, or if you need help with network settings.",
                    "escalation": False
                }
            },
            "billing": {
                "payment_failed": {
                    "solution": "Payment Failure Resolution: 1) Verify card details and expiration date 2) Check available balance/credit limit 3) Contact bank about international/online restrictions 4) Try different payment method 5) Update billing address to match bank records",
                    "follow_up": "I can help update your payment method or process a manual payment over phone.",
                    "escalation": False
                },
                "refund_request": {
                    "solution": "Refund Process: 1) Refunds available within 30 days of purchase 2) Provide order/transaction number 3) Specify refund reason 4) Process takes 5-7 business days 5) Refund goes to original payment method",
                    "follow_up": "I can initiate your refund request now. What's your order number and reason?",
                    "escalation": False
                },
                "billing_dispute": {
                    "solution": "Billing Dispute Resolution: 1) Review detailed invoice in account settings 2) Check for pro-rated charges or upgrades 3) Verify billing cycle dates 4) Compare with previous invoices 5) Contact billing team for adjustments",
                    "follow_up": "I can review your billing history and explain any charges you're questioning.",
                    "escalation": True
                },
                "subscription_change": {
                    "solution": "Subscription Management: 1) Changes take effect next billing cycle 2) Upgrades are immediate, downgrades at cycle end 3) Cancel anytime in account settings 4) No cancellation fees 5) Keep access until period ends",
                    "follow_up": "Would you like me to help you change your plan or show you the options?",
                    "escalation": False
                }
            },
            "account": {
                "account_locked": {
                    "solution": "Account Lockout Resolution: 1) Wait 15 minutes after 3 failed attempts 2) Use password reset to unlock immediately 3) Check email for security alerts 4) Verify no unauthorized access 5) Contact security team if suspicious activity",
                    "follow_up": "I can unlock your account manually or help secure it if you suspect unauthorized access.",
                    "escalation": False
                },
                "data_privacy": {
                    "solution": "Data Privacy & Export: 1) Request data export in Privacy Settings 2) Processing takes 24-48 hours 3) Download link valid for 7 days 4) Includes all personal data and activity 5) Delete account option available",
                    "follow_up": "I can initiate your data export request or help with privacy settings.",
                    "escalation": False
                },
                "account_security": {
                    "solution": "Security Enhancement: 1) Enable two-factor authentication 2) Use unique, strong passwords 3) Review recent login activity 4) Update security questions 5) Monitor for suspicious activity alerts",
                    "follow_up": "Let me help you set up 2FA or review your recent account activity for security.",
                    "escalation": False
                },
                "profile_issues": {
                    "solution": "Profile Update Help: 1) Go to Account Settings > Profile 2) Edit information carefully 3) Verify email changes via confirmation link 4) Upload photos in JPG/PNG format 5) Save changes and refresh",
                    "follow_up": "What specific profile information do you need help updating?",
                    "escalation": False
                }
            },
            "general": {
                "feature_request": {
                    "solution": "Feature Requests: 1) Submit ideas in app feedback section 2) Vote on existing requests in community forum 3) Follow product updates blog 4) Join beta testing program 5) Contact product team directly",
                    "follow_up": "I can forward your feature request to our product team and add you to relevant update lists.",
                    "escalation": True
                },
                "service_outage": {
                    "solution": "Service Status: 1) Check status page at status.company.com 2) Follow @company_status on social media 3) Enable service notifications 4) Current issues posted in real-time 5) Estimated resolution times provided",
                    "follow_up": "Let me check current service status and provide updates on any ongoing issues.",
                    "escalation": False
                }
            }
        }

        # Advanced search with keyword matching
        results = []
        query_words = query.lower().split()

        for cat, items in knowledge_base.items():
            if category == "general" or cat == category:
                for key, data in items.items():
                    # Check for keyword matches
                    key_words = key.replace('_', ' ').lower().split()
                    solution_words = data["solution"].lower().split()

                    match_score = 0
                    for word in query_words:
                        if word in key_words:
                            match_score += 3
                        elif word in solution_words:
                            match_score += 1
                        elif any(word in kw for kw in key_words):
                            match_score += 2

                    if match_score > 0:
                        results.append({
                            "title": key.replace('_', ' ').title(),
                            "solution": data["solution"],
                            "follow_up": data["follow_up"],
                            "needs_escalation": data["escalation"],
                            "score": match_score
                        })

        # Sort by relevance score
        results.sort(key=lambda x: x["score"], reverse=True)

        if not results:
            return """No specific knowledge base articles found for your query.

            **How I can help you:**
            - Technical issues (login, app crashes, connectivity)
            - Billing questions (payments, refunds, subscriptions)
            - Account management (security, privacy, profile updates)
            - General inquiries (features, service status)

            Please provide more specific details about your issue, and I'll find the right solution."""

        # Format top 2 results
        formatted_results = []
        for result in results[:2]:
            formatted_results.append(f"""**{result['title']}**

{result['solution']}

*Next steps:* {result['follow_up']}""")

        return "\n\n---\n\n".join(formatted_results)

    except Exception as e:
        return f"Knowledge base search error: {str(e)}"

@tool
def create_support_ticket(customer_email: str, issue_category: str, priority: str, description: str) -> str:
    """Create a support ticket for tracking and follow-up"""
    try:
        import datetime
        import random

        # Generate ticket ID
        ticket_id = f"TKT-{datetime.datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"

        # Create ticket data
        ticket = {
            "ticket_id": ticket_id,
            "customer_email": customer_email,
            "category": issue_category,
            "priority": priority,
            "description": description,
            "status": "open",
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "assigned_agent": "AI Assistant",
            "estimated_resolution": "24-48 hours" if priority == "high" else "2-3 business days"
        }

        # In a real system, this would save to a database
        # For demo, we'll return the ticket info

        return f"""✅ **Support Ticket Created**

**Ticket ID:** {ticket_id}
**Customer:** {customer_email}
**Category:** {issue_category.title()}
**Priority:** {priority.title()}
**Status:** Open
**Created:** {ticket['created_at']}
**Estimated Resolution:** {ticket['estimated_resolution']}

**What happens next:**
1. You'll receive email confirmation within 5 minutes
2. A support agent will review your case
3. Initial response within {ticket['estimated_resolution']}
4. You can track progress at: support.company.com/tickets/{ticket_id}

**Reference this ticket ID in future communications for faster service.**"""

    except Exception as e:
        return f"Ticket creation error: {str(e)}"

@tool
def perform_account_action(action: str, customer_email: str, details: str = "") -> str:
    """Perform practical account actions like password reset, unlock, etc."""
    try:
        actions_available = {
            "password_reset": {
                "description": "Send password reset email",
                "message": "✅ Password reset email sent to {email}. Please check your inbox (and spam folder). The reset link expires in 24 hours."
            },
            "account_unlock": {
                "description": "Unlock temporarily locked account",
                "message": "✅ Account unlocked successfully. You can now log in normally. For security, consider enabling 2FA in your account settings."
            },
            "payment_retry": {
                "description": "Retry failed payment",
                "message": "✅ Payment retry initiated. You'll receive an email confirmation if successful, or further instructions if it fails again."
            },
            "subscription_pause": {
                "description": "Temporarily pause subscription",
                "message": "✅ Subscription paused. Your access continues until {date}. You can reactivate anytime in account settings."
            },
            "data_export": {
                "description": "Initiate personal data export",
                "message": "✅ Data export request submitted. You'll receive a download link within 24-48 hours. The link will be valid for 7 days."
            }
        }

        if action not in actions_available:
            return f"❌ Action '{action}' not available. Available actions: {', '.join(actions_available.keys())}"

        # In a real system, this would integrate with actual services
        action_info = actions_available[action]
        result_message = action_info["message"].format(
            email=customer_email,
            date="your next billing date" if "subscription" in action else ""
        )

        # Add action log entry
        import datetime
        log_entry = f"\n\n📝 **Action Log**: {action_info['description']} performed at {datetime.datetime.now().strftime('%H:%M:%S')} for {customer_email}"

        return result_message + log_entry

    except Exception as e:
        return f"Action execution error: {str(e)}"

@tool
def check_system_status(service: str = "all") -> str:
    """Check current system status and any ongoing issues"""
    try:
        # Simulate system status check
        import random

        services = {
            "login": {"status": "operational", "uptime": "99.9%"},
            "payments": {"status": "operational", "uptime": "99.8%"},
            "api": {"status": "operational", "uptime": "99.95%"},
            "mobile_app": {"status": "operational", "uptime": "99.7%"},
            "website": {"status": "operational", "uptime": "99.9%"}
        }

        # Occasionally simulate an issue (5% chance)
        if random.random() < 0.05:
            affected_service = random.choice(list(services.keys()))
            services[affected_service]["status"] = "degraded"
            services[affected_service]["issue"] = "Experiencing slower than normal response times"
            services[affected_service]["eta"] = "Resolution expected within 2 hours"

        if service != "all" and service in services:
            service_info = services[service]
            status_message = f"**{service.title()} Status**: {service_info['status'].title()}\n"
            status_message += f"**Uptime**: {service_info['uptime']}\n"
            if "issue" in service_info:
                status_message += f"**Current Issue**: {service_info['issue']}\n"
                status_message += f"**ETA**: {service_info['eta']}"
            return status_message

        # Return all services status
        status_message = "🟢 **System Status Overview**\n\n"
        for svc, info in services.items():
            icon = "🟡" if info["status"] == "degraded" else "🟢"
            status_message += f"{icon} **{svc.title()}**: {info['status'].title()} ({info['uptime']} uptime)\n"
            if "issue" in info:
                status_message += f"   ⚠️ {info['issue']} - {info['eta']}\n"

        status_message += f"\n📊 **Overall System Health**: {'All systems operational' if all(s['status'] == 'operational' for s in services.values()) else 'Some services experiencing issues'}"
        status_message += "\n🔗 **Live Status Page**: status.company.com"

        return status_message

    except Exception as e:
        return f"Status check error: {str(e)}"

@tool
def sentiment_analyzer(text: str) -> str:
    """Analyze customer sentiment from text"""
    try:
        # Simulate sentiment analysis
        negative_indicators = ["angry", "frustrated", "terrible", "worst", "hate", "awful", "disappointed"]
        positive_indicators = ["great", "excellent", "love", "amazing", "perfect", "wonderful", "satisfied"]
        urgency_indicators = ["urgent", "immediately", "asap", "emergency", "critical", "now"]

        text_lower = text.lower()

        negative_score = sum(1 for word in negative_indicators if word in text_lower)
        positive_score = sum(1 for word in positive_indicators if word in text_lower)
        urgency_score = sum(1 for word in urgency_indicators if word in text_lower)

        if negative_score > positive_score:
            sentiment = "negative"
            if urgency_score > 0:
                sentiment = "frustrated"
        elif positive_score > negative_score:
            sentiment = "positive"
        else:
            sentiment = "neutral"

        sentiment_score = (positive_score - negative_score) / max(1, positive_score + negative_score + 1)

        analysis = {
            "sentiment": sentiment,
            "sentiment_score": round(sentiment_score, 2),
            "urgency_level": min(10, urgency_score * 2 + 1),
            "key_emotions": [],
            "response_tone_recommendation": "empathetic" if sentiment in ["negative", "frustrated"] else "friendly"
        }

        return json.dumps(analysis)

    except Exception as e:
        return f"Sentiment analysis error: {str(e)}"

@tool
def escalation_checker(issue_summary: str, sentiment: str, attempts: int) -> str:
    """Check if issue should be escalated"""
    try:
        escalation_triggers = {
            "high_priority_keywords": ["billing", "payment", "refund", "legal", "security", "data loss"],
            "negative_sentiment_threshold": 3,
            "max_attempts": 2
        }

        should_escalate = False
        reasons = []

        # Check for high priority keywords
        for keyword in escalation_triggers["high_priority_keywords"]:
            if keyword in issue_summary.lower():
                should_escalate = True
                reasons.append(f"High priority issue detected: {keyword}")

        # Check sentiment
        if sentiment in ["frustrated", "negative"] and attempts >= escalation_triggers["negative_sentiment_threshold"]:
            should_escalate = True
            reasons.append("Customer frustration detected with multiple attempts")

        # Check attempt count
        if attempts > escalation_triggers["max_attempts"]:
            should_escalate = True
            reasons.append(f"Maximum support attempts exceeded ({attempts})")

        escalation_info = {
            "should_escalate": should_escalate,
            "reasons": reasons,
            "recommended_department": "tier2_support" if should_escalate else "continue_tier1",
            "escalation_priority": "high" if len(reasons) > 1 else "medium"
        }

        return json.dumps(escalation_info)

    except Exception as e:
        return f"Escalation check error: {str(e)}"

# Support Agent Functions
def create_support_llm(provider: str, model: str, **kwargs):
    """Create LLM instance for support agent with configurable parameters"""
    temperature = kwargs.get('temperature', 0.3)

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

def issue_classifier(state: SupportState, llm) -> SupportState:
    """Classify customer issue and determine priority"""
    messages = state["messages"]

    if not messages:
        return state

    latest_message = messages[-1].content if messages else ""

    classification_prompt = f"""
    Analyze this customer message and classify the issue:

    Customer Message: "{latest_message}"

    Classify into:
    1. Category: technical, billing, account, product, or general
    2. Priority: low, medium, high, or urgent
    3. Sentiment: positive, neutral, negative, or frustrated
    4. Urgency Score: 1-10
    5. Brief Summary: One sentence description

    Respond in JSON format matching the CustomerIssue schema.
    """

    response = llm.invoke([
        SystemMessage(content="You are an expert customer service issue classifier."),
        HumanMessage(content=classification_prompt)
    ])

    try:
        # Parse classification (simplified for demo)
        if "billing" in latest_message.lower() or "payment" in latest_message.lower():
            category = "billing"
            priority = "high"
        elif "technical" in latest_message.lower() or "error" in latest_message.lower():
            category = "technical"
            priority = "medium"
        elif "account" in latest_message.lower():
            category = "account"
            priority = "medium"
        else:
            category = "general"
            priority = "low"

        state["issue_category"] = category
        state["priority_level"] = priority
        state["next_action"] = "sentiment_analysis"

    except Exception as e:
        st.error(f"Classification error: {e}")
        state["next_action"] = "error_handler"

    return state

def sentiment_analysis_node(state: SupportState) -> SupportState:
    """Analyze customer sentiment"""
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""

    # Use sentiment analyzer tool
    sentiment_result = sentiment_analyzer.invoke({"text": latest_message})
    sentiment_data = json.loads(sentiment_result)

    state["sentiment_score"] = sentiment_data["sentiment_score"]
    state["conversation_context"]["sentiment"] = sentiment_data["sentiment"]
    state["conversation_context"]["urgency_level"] = sentiment_data["urgency_level"]
    state["next_action"] = "knowledge_search"

    return state

def knowledge_search_node(state: SupportState) -> SupportState:
    """Search knowledge base for solutions"""
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""
    category = state["issue_category"]

    # Search knowledge base
    kb_results = knowledge_base_search.invoke({"query": latest_message, "category": category})

    state["conversation_context"]["knowledge_results"] = kb_results
    state["next_action"] = "response_generation"

    return state

def response_generator(state: SupportState, llm) -> SupportState:
    """Generate appropriate response to customer"""
    messages = state["messages"]
    kb_results = state["conversation_context"].get("knowledge_results", "")
    sentiment = state["conversation_context"].get("sentiment", "neutral")
    category = state["issue_category"]
    priority = state["priority_level"]

    response_prompt = f"""
    You are a helpful customer support agent. Generate a response based on:

    Customer Issue Category: {category}
    Priority Level: {priority}
    Customer Sentiment: {sentiment}

    Knowledge Base Results: {kb_results}

    Guidelines:
    1. Be empathetic, especially if sentiment is negative/frustrated
    2. Provide clear, actionable solutions from knowledge base
    3. Ask clarifying questions if needed
    4. Maintain professional but friendly tone
    5. Offer additional help or escalation if appropriate

    Generate a helpful response that addresses the customer's concern.
    """

    response = llm.invoke([
        SystemMessage(content="You are an expert customer support agent focused on resolution."),
        HumanMessage(content=response_prompt)
    ])

    # Add AI response to conversation
    state["messages"].append(AIMessage(content=response.content))
    state["next_action"] = "escalation_check"

    return state

def escalation_check_node(state: SupportState) -> SupportState:
    """Check if issue should be escalated"""
    messages = state["messages"]
    issue_summary = f"{state['issue_category']}: {messages[-2].content if len(messages) > 1 else ''}"
    sentiment = state["conversation_context"].get("sentiment", "neutral")
    attempts = state["escalation_level"]

    # Check escalation criteria
    escalation_result = escalation_checker.invoke({
        "issue_summary": issue_summary,
        "sentiment": sentiment,
        "attempts": attempts
    })

    escalation_data = json.loads(escalation_result)

    if escalation_data["should_escalate"]:
        state["next_action"] = "escalate"
        state["escalation_level"] += 1
        state["agent_notes"].append(f"Escalation triggered: {', '.join(escalation_data['reasons'])}")
    else:
        state["next_action"] = "continue_conversation"

    return state

def escalation_handler(state: SupportState) -> SupportState:
    """Handle escalation to human agent"""
    escalation_message = f"""
    🔄 **Escalating to Human Agent**

    **Issue Summary:**
    - Category: {state["issue_category"]}
    - Priority: {state["priority_level"]}
    - Escalation Level: {state["escalation_level"]}

    **Agent Notes:**
    {chr(10).join(f"• {note}" for note in state["agent_notes"])}

    A human agent will be with you shortly. Thank you for your patience.
    """

    state["messages"].append(AIMessage(content=escalation_message))
    state["resolution_status"] = "escalated"
    state["next_action"] = "end"

    return state

def conversation_router(state: SupportState) -> str:
    """Route conversation based on current state"""
    next_action = state.get("next_action", "sentiment_analysis")

    if next_action == "sentiment_analysis":
        return "sentiment_analysis"
    elif next_action == "knowledge_search":
        return "knowledge_search"
    elif next_action == "response_generation":
        return "response_generation"
    elif next_action == "escalation_check":
        return "escalation_check"
    elif next_action == "escalate":
        return "escalation_handler"
    elif next_action == "end":
        return END
    else:
        return "continue_conversation"

def create_support_graph(llm):
    """Create the customer support workflow graph"""

    # Create workflow
    workflow = StateGraph(SupportState)

    # Add nodes
    workflow.add_node("issue_classifier", lambda state: issue_classifier(state, llm))
    workflow.add_node("sentiment_analysis", sentiment_analysis_node)
    workflow.add_node("knowledge_search", knowledge_search_node)
    workflow.add_node("response_generation", lambda state: response_generator(state, llm))
    workflow.add_node("escalation_check", escalation_check_node)
    workflow.add_node("escalation_handler", escalation_handler)

    # Add edges
    workflow.add_edge(START, "issue_classifier")
    workflow.add_conditional_edges("issue_classifier", conversation_router)
    workflow.add_conditional_edges("sentiment_analysis", conversation_router)
    workflow.add_conditional_edges("knowledge_search", conversation_router)
    workflow.add_conditional_edges("response_generation", conversation_router)
    workflow.add_conditional_edges("escalation_check", conversation_router)
    workflow.add_edge("escalation_handler", END)

    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

def render_customer_support_interface():
    """Render the Streamlit interface for Customer Support Agent"""
    st.header("🎧 Customer Support Agent")

    # Sidebar configuration
    with st.sidebar:
        st.subheader("⚙️ Configuration")

        # LLM Selection
        llm_provider = st.selectbox(
            "LLM Provider",
            ["Gemini", "Ollama", "Groq", "Anthropic", "OpenAI"],
            key='support_llm_provider'
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
            key='support_model'
        )

        # Ollama-specific configuration
        ollama_base_url = None
        if llm_provider == "Ollama":
            st.markdown("**🔧 Ollama Configuration**")
            ollama_base_url = st.text_input(
                "Ollama Base URL",
                value="http://localhost:11434",
                key='support_ollama_url',
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


        # Customer info
        st.markdown("### 👤 Customer Info")
        customer_id = st.text_input("Customer ID", value="CUST_001", key='customer_id')


        # Reset conversation
        if st.button("🔄 New Conversation", key='reset_support'):
            if 'support_conversation' in st.session_state:
                del st.session_state.support_conversation
            if 'support_state' in st.session_state:
                del st.session_state.support_state
            st.rerun()

    # Initialize conversation state
    if 'support_conversation' not in st.session_state:
        st.session_state.support_conversation = []

    if 'support_state' not in st.session_state:
        st.session_state.support_state = {
            "messages": [],
            "customer_id": customer_id,
            "issue_category": "",
            "priority_level": "",
            "sentiment_score": 0.0,
            "escalation_level": 0,
            "resolution_status": "open",
            "conversation_context": {},
            "agent_notes": [],
            "next_action": "issue_classifier"
        }

    # Advanced Configuration Section
    with st.expander("⚙️ Advanced Configuration", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🎛️ Model Settings**")
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                key='support_temperature',
                help="Controls response variability"
            )

        with col2:
            st.markdown("**🎧 Support Settings**")
            escalation_threshold = st.slider(
                "Escalation Threshold",
                min_value=1,
                max_value=10,
                value=7,
                key='escalation_threshold',
                help="Sentiment score threshold for escalation"
            )

            auto_resolution = st.checkbox(
                "Auto-Resolution",
                value=False,
                key='auto_resolution',
                help="Attempt automatic issue resolution"
            )

            sentiment_tracking = st.checkbox(
                "Real-time Sentiment",
                value=True,
                key='sentiment_tracking',
                help="Track customer sentiment in real-time"
            )

    # Show conversation history
    for i, msg in enumerate(st.session_state.support_conversation):
        if msg["role"] == "customer":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # Customer input
    customer_message = st.chat_input("Describe your issue or ask a question...")

    if customer_message:
        # Add customer message to conversation
        st.session_state.support_conversation.append({
            "role": "customer",
            "content": customer_message
        })

        # Add to graph state
        st.session_state.support_state["messages"].append(HumanMessage(content=customer_message))

        # Process with support agent
        with st.spinner("Processing your request..."):
            try:
                # Create LLM and graph with configuration
                llm_kwargs = {
                    'temperature': temperature
                }
                if llm_provider == "Ollama" and ollama_base_url:
                    llm_kwargs['base_url'] = ollama_base_url
                llm = create_support_llm(llm_provider, model, **llm_kwargs)
                support_graph = create_support_graph(llm)

                # Execute support workflow
                config = {"configurable": {"thread_id": customer_id}}
                result = support_graph.invoke(st.session_state.support_state, config)

                # Get agent response
                if result["messages"]:
                    latest_response = result["messages"][-1]
                    if isinstance(latest_response, AIMessage):
                        st.session_state.support_conversation.append({
                            "role": "agent",
                            "content": latest_response.content
                        })

                # Update state
                st.session_state.support_state = result

                # Save conversation log if needed
                dirs = setup_langgraph_directories()
                if result.get("resolution_status") == "escalated":
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_filename = f"support_conversation_{customer_id}_{timestamp}.json"
                    log_path = dirs["output"] / log_filename

                    conversation_log = {
                        "customer_id": customer_id,
                        "timestamp": timestamp,
                        "conversation": st.session_state.support_conversation,
                        "final_state": {
                            "issue_category": result["issue_category"],
                            "priority_level": result["priority_level"],
                            "escalation_level": result["escalation_level"],
                            "resolution_status": result["resolution_status"],
                            "agent_notes": result["agent_notes"]
                        }
                    }

                    with open(log_path, 'w') as f:
                        json.dump(conversation_log, f, indent=2)

                st.rerun()

            except Exception as e:
                st.error(f"Support processing error: {str(e)}")

    # Support dashboard
    if st.session_state.support_state["issue_category"]:
        with st.expander("📊 Support Dashboard", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Issue Category", st.session_state.support_state["issue_category"])
                st.metric("Priority Level", st.session_state.support_state["priority_level"])

            with col2:
                st.metric("Escalation Level", st.session_state.support_state["escalation_level"])
                st.metric("Resolution Status", st.session_state.support_state["resolution_status"])

            with col3:
                sentiment_score = st.session_state.support_state["sentiment_score"]
                st.metric("Sentiment Score", f"{sentiment_score:.2f}")

                sentiment = st.session_state.support_state["conversation_context"].get("sentiment", "unknown")
                sentiment_color = {"positive": "🟢", "neutral": "🟡", "negative": "🔴", "frustrated": "🔴"}.get(sentiment, "⚪")
                st.metric("Sentiment", f"{sentiment_color} {sentiment.title()}")

            # Agent notes
            if st.session_state.support_state["agent_notes"]:
                st.markdown("**Agent Notes:**")
                for note in st.session_state.support_state["agent_notes"]:
                    st.write(f"• {note}")

    # Support features info
    with st.expander("ℹ️ Support Agent Features", expanded=False):
        st.markdown("""
        **Intelligent Customer Support Features:**

        🧠 **Smart Issue Classification**
        - Automatic categorization of customer issues
        - Priority level assignment based on content
        - Context-aware routing to appropriate solutions

        😊 **Sentiment Analysis**
        - Real-time emotion detection
        - Adaptive response tone based on customer mood
        - Escalation triggers for frustrated customers

        📚 **Knowledge Base Integration**
        - Instant access to solution database
        - Category-specific article recommendations
        - Contextual help suggestions

        🔄 **Intelligent Escalation**
        - Automatic escalation based on complexity
        - Human handoff for unresolved issues
        - Comprehensive context transfer

        💾 **Conversation Memory**
        - Persistent conversation context
        - Customer history tracking
        - Agent notes and annotations
        """)

if __name__ == "__main__":
    render_customer_support_interface()