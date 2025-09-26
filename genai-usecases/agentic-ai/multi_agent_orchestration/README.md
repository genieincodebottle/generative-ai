# Multi-Agent Orchestration System

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/🦜🔗_LangChain-121011?logoColor=white)](https://langchain.com)
[![CrewAI](https://img.shields.io/badge/🚢_CrewAI-FF6B35?logoColor=white)](https://crewai.com)
[![LangGraph](https://img.shields.io/badge/🕸️_LangGraph-1C3A3A?logoColor=white)](https://langchain-ai.github.io/langgraph/)

> Advanced multi-agent coordination patterns showcasing cross-framework integration, hybrid orchestration strategies, and sophisticated inter-agent communication protocols.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Orchestration Architecture](#orchestration-architecture)
- [Available Patterns](#available-patterns)
- [Use Cases](#use-cases)
- [Advanced Features](#advanced-features)

## Overview

**Multi-Agent Orchestration** is the top level of AI teamwork, multiple frameworks working together to handle complex problems. This setup shows advanced patterns like hierarchical coordination, cross-framework integration and event-driven agents that adapt, collaborate and scale.

### Key Features

| Feature | Description |
|---------|-------------|
| **Cross-Framework Integration** | CrewAI + LangGraph hybrid workflows |
| **Hierarchical Coordination** | Manager-worker delegation with specialized roles |
| **Event-Driven Communication** | Real-time inter-agent message passing |
| **Performance Analytics** | Comprehensive execution metrics and monitoring |
| **Business Tool Integration** | Real-time market data and financial analysis |
| **Adaptive Orchestration** | Dynamic workflow adjustment based on results |

## ⚙️ Installation

Get up and running in under 5 minutes:

1. Clone the repository

    ```bash
    git clone https://github.com/genieincodebottle/generative-ai.git
    cd genai-usecases\agentic-ai
    ```

2. Open the Project in VS Code or any code editor.
3. Create a virtual environment by running the following command in the terminal:

    ```bash
    pip install uv #if uv not installed
    uv venv
    .venv\Scripts\activate # On Linux -> source venv/bin/activate
    ```

4. Install dependencies

    ```bash
    uv pip install -r requirements.txt
    ```
5. Configure Environment

    - **Option 1**: <-> Local LLM Setup

        1. **Ollama** - Run open models locally with zero API costs:

            ```bash
            # Install Ollama from https://ollama.ai
            curl -fsSL https://ollama.ai/install.sh | sh

            # Pull a lightweight models as per your system memory availability
            ollama pull llama3.2:3b # Options: gpt-oss:20b, llama3.2:1b, llama3.2:3b, llama3.1:8b, deepseek-r1:1.5b, deepseek-r1:7b, deepseek-r1:8b, gemma3:1b, gemma3:4b, gemma3:12b, phi3:3.8b

            # No API keys needed..
            ```
            Reference guide for memory requirement
            - **llama3.2:1b** (1B parameters) - ~0.7GB RAM
            - **llama3.2:3b** (3B parameters) - ~2GB RAM
            - **llama3.1:8b** (8B parameters) - ~4.5GB RAM
            - **gemma3:1b** (1B parameters) - ~0.7GB RAM
            - **gemma3:4b** (4B parameters) - ~2.5GB RAM

            **Note**: Ollama uses Q4_0 quantization (~0.5-0.7GB per billion parameters)
        2. **Run the following command to list the local open models available in Ollama**

            ```bash
            ollama list
            ```
        3. **Start Ollama Service** (if needed)
            ```bash
            ollama serve  # Only needed if Ollama isn't running automatically
            ```

            **Note**: Most desktop installations start Ollama automatically. Check if it's running by visiting `http://localhost:11434` in your browser or by looking for the Ollama icon in your system tray.

    - **Option 2**:  Cloud Providers

        - Configure Environment
            - rename ```.env.example``` to ```.env``` in your project root
            - Update with your keys:

                ```env
                # Choose your preferred providers
                GEMINI_API_KEY=your-gemini-key-here
                GROQ_API_KEY=your-groq-key-here
                ANTHROPIC_API_KEY=your-anthropic-key
                OPENAI_API_KEY=your-openai-key-here
                ```

6. Run the Multi-Agent Orchestration System

    ```bash
    streamlit run multi_agent_orchestration\multi_agent_orchestration.py
    ```

## Orchestration Architecture

### Core Components

#### **Orchestration Engine**
- **Pattern Selection**: Choose from multiple coordination strategies
- **Framework Integration**: Seamless CrewAI and LangGraph collaboration
- **Execution Management**: Comprehensive workflow orchestration
- **Performance Monitoring**: Real-time metrics and analytics

#### **Business Tools**
- **Market Analysis**: Real-time financial data and trend analysis
- **Competitive Intelligence**: Market positioning and competitor insights
- **Financial Modeling**: Revenue projections and business metrics
- **Risk Assessment**: Strategic risk evaluation and mitigation

#### **Communication Layer**
- **Inter-Framework Messaging**: CrewAI -> LangGraph communication
- **Event Broadcasting**: Real-time status updates and coordination
- **Result Aggregation**: Intelligent synthesis of multi-agent outputs
- **Error Handling**: Graceful failure recovery and coordination

## Available Orchestration Patterns

### 1. **Hierarchical CrewAI Coordination**
> **Structured multi-agent collaboration with role-based specialization**

**Perfect for beginners** - Clear hierarchy with visible delegation patterns.

Demonstrate traditional multi-agent coordination where a manager agent delegates specialized tasks to worker agents, each with distinct expertise and tools for comprehensive business analysis.

#### When to Use
- Complex business analysis requiring multiple expert perspectives
- Structured workflows with clear role definitions
- Quality assurance processes with hierarchical review

#### How It Works
```
Business Query -> Manager Coordinator -> Market Analyst -> Competitive Analyst -> Financial Modeler -> Strategic Synthesizer
```

#### Key Features
- **Role-Based Specialization**: Each agent has distinct expertise and tools
- **Hierarchical Delegation**: Manager coordinates and delegates tasks efficiently
- **Real-Time Data Integration**: Live market and financial data analysis
- **Quality Control**: Multi-tier review and validation processes

#### Agent Roles
| Agent | Responsibility | Tools & Expertise |
|-------|---------------|-----------|
| **Manager Coordinator** | Task delegation, workflow coordination | Business strategy, project management |
| **Market Research Analyst** | Market analysis, trend identification | Market data APIs, trend analysis tools |
| **Competitive Intelligence Analyst** | Competitor analysis, market positioning | Financial data, competitive benchmarking |
| **Financial Modeling Specialist** | Revenue projections, financial metrics | Financial modeling, risk assessment tools |
| **Strategic Business Synthesizer** | Final synthesis, strategic recommendations | Business intelligence, strategic planning |

#### Example Applications
```
1. Market Entry Analysis: Market sizing -> Competitor mapping -> Financial projections -> Strategic recommendations

2. Product Launch Planning: Market research -> Competitive analysis -> Revenue modeling -> Go-to-market strategy

3. Investment Evaluation: Market opportunity -> Competitive landscape -> Financial modeling -> Investment thesis

4. Business Expansion: Market assessment -> Competitive positioning -> Financial planning -> Expansion strategy
```

---

### 2. **Graph Workflow LangGraph**
> **State-driven multi-agent coordination with dynamic routing**

Execute sophisticated business analysis through graph-based workflows where agents coordinate through shared state management and conditional routing based on analysis results.

#### When to Use
- Complex workflows requiring conditional logic and dynamic routing
- State-dependent decision making processes
- Workflows requiring sophisticated error handling and recovery

#### How It Works
```
Task Input -> Research Coordinator -> Market Analyst -> Financial Analyst -> Strategic Synthesizer -> Report Generator
```

#### Key Features
- **Type-Safe State Management**: Structured state with automatic reducers
- **Dynamic Routing**: Conditional execution paths based on analysis results
- **Tool Integration**: Seamless integration with business analysis tools
- **Async Execution**: High-performance concurrent agent operations

#### Agent Roles
| Agent | Responsibility | State Management |
|-------|---------------|-----------|
| **Research Coordinator** | Task breakdown, research planning | Initializes analysis context and coordinates workflow |
| **Market Analysis Agent** | Market research, opportunity assessment | Updates market data and competitive intelligence |
| **Financial Analysis Agent** | Financial modeling, risk assessment | Maintains financial projections and risk metrics |
| **Strategic Synthesis Agent** | Strategic recommendations, final synthesis | Aggregates insights and generates strategic recommendations |

#### Example Applications
```
1. Strategic Planning: Goal definition -> Market analysis -> Financial modeling -> Strategic synthesis -> Implementation plan

2. Due Diligence: Target analysis -> Market assessment -> Financial review -> Risk evaluation -> Investment recommendation

3. Business Optimization: Current analysis -> Market opportunities -> Financial impact -> Strategic improvements -> Action plan

4. Competitive Response: Competitor analysis -> Market dynamics -> Financial implications -> Strategic counter-measures -> Response plan
```

---

### 3. **Hybrid Cross-Framework Integration**
> **Advanced coordination combining CrewAI hierarchy with LangGraph workflows**

Demonstrate the ultimate in multi-agent coordination by combining CrewAI's structured role-based collaboration with LangGraph's state-driven workflows for unprecedented analytical depth.

#### When to Use
- Maximum analytical depth requiring both structured roles and dynamic workflows
- Complex business challenges requiring multiple coordination patterns
- Enterprise-level analysis with comprehensive quality assurance

#### How It Works
```
CrewAI Analysis -> State Transfer -> LangGraph Processing -> Result Integration -> Hybrid Synthesis
```

#### Key Features
- **Best of Both Frameworks**: Combines CrewAI's role clarity with LangGraph's flexibility
- **Cross-Framework State Management**: Seamless data transfer between frameworks
- **Advanced Coordination**: Multiple coordination patterns in single workflow
- **Enterprise Scalability**: Handles complex, multi-faceted business challenges

#### Integration Architecture
| Phase | Framework | Coordination Pattern | Output |
|-------|-----------|---------------------|---------|
| **Initial Analysis** | CrewAI | Hierarchical delegation | Structured business insights |
| **Deep Processing** | LangGraph | State-driven workflow | Dynamic analysis results |
| **Cross-Framework Synthesis** | Hybrid | Intelligent aggregation | Comprehensive recommendations |
| **Final Integration** | Custom | Result harmonization | Unified strategic output |

#### Example Applications
```
1. Enterprise Strategy: CrewAI market analysis -> LangGraph scenario modeling -> Hybrid integration -> Strategic roadmap

2. M&A Analysis: CrewAI due diligence -> LangGraph financial modeling -> Hybrid risk assessment -> Investment decision

3. Digital Transformation: CrewAI current state -> LangGraph transformation planning -> Hybrid implementation -> Change strategy

4. Market Expansion: CrewAI opportunity analysis -> LangGraph expansion modeling -> Hybrid integration -> Global strategy
```

---

### 4. **Event-Driven Coordination**
> **Real-time reactive multi-agent coordination with intelligent event handling**

Implement sophisticated event-driven coordination where agents react to real-time events, market changes, and inter-agent communications for dynamic business intelligence.

#### When to Use
- Real-time business monitoring and response systems
- Dynamic market analysis with immediate reaction capabilities
- Interactive business intelligence with human-in-the-loop coordination

#### How It Works
```
1. Event Detection -> Agent Notification -> Parallel Analysis -> Event Aggregation -> Response Coordination -> Action Execution
```

#### Key Features
- **Real-Time Event Processing**: Immediate response to market and business events
- **Dynamic Agent Coordination**: Agents adapt behavior based on event patterns
- **Intelligent Event Routing**: Smart distribution of events to relevant agents
- **Coordinated Response**: Synchronized multi-agent responses to complex events

#### Event Types & Handlers
| Event Type | Trigger | Agent Response | Coordination Pattern |
|------------|---------|----------------|---------------------|
| **Market Events** | Price changes, news | Market analyst activation | Parallel analysis with result aggregation |
| **Competitive Events** | Competitor actions | Competitive intelligence | Sequential analysis with strategic response |
| **Business Events** | Performance metrics | Financial analyst engagement | Dynamic routing based on event severity |
| **User Events** | Interactive queries | Real-time response coordination | Human-in-the-loop with immediate feedback |

#### Example Applications
```
1. Market Monitoring: Real-time price feeds -> Market analysis -> Competitive response -> Strategic recommendations -> Action alerts

2. Business Intelligence: Performance metrics -> Analysis triggers -> Multi-agent insights -> Executive dashboards -> Decision support

3. Crisis Management: Event detection -> Rapid analysis -> Risk assessment -> Response coordination -> Action implementation

4. Opportunity Detection: Market signals -> Opportunity analysis -> Competitive assessment -> Strategic evaluation -> Investment recommendations
```

## Use Cases

### Business Applications
- **Strategic Planning**: Multi-framework analysis with comprehensive market intelligence
- **Investment Analysis**: Deep financial modeling with competitive intelligence
- **Market Research**: Real-time market analysis with event-driven updates
- **Business Intelligence**: Sophisticated analytics with cross-framework synthesis

### Enterprise Applications
- **Digital Transformation**: Complex change management with multi-agent coordination
- **M&A Analysis**: Comprehensive due diligence with hybrid analytical approaches
- **Risk Management**: Advanced risk assessment with real-time monitoring
- **Competitive Intelligence**: Dynamic competitor analysis with market event processing

### Research & Development
- **Market Innovation**: Advanced opportunity identification with predictive analytics
- **Technology Assessment**: Multi-perspective technology evaluation
- **Strategic Research**: Comprehensive research coordination across multiple domains
- **Business Experimentation**: A/B testing coordination with intelligent result synthesis

### Operations & Management
- **Performance Optimization**: Real-time performance monitoring with intelligent recommendations
- **Resource Allocation**: Dynamic resource optimization with multi-agent coordination
- **Quality Assurance**: Comprehensive quality management with hierarchical review
- **Process Improvement**: Continuous improvement with intelligent process analysis

## Advanced Features

### Cross-Framework Orchestration
- **Seamless Integration**: Natural coordination between CrewAI and LangGraph
- **State Synchronization**: Intelligent state transfer and management
- **Result Harmonization**: Advanced synthesis of multi-framework outputs
- **Performance Optimization**: Efficient resource utilization across frameworks

### Real-Time Business Intelligence
- **Live Market Data**: Real-time financial and market data integration
- **Dynamic Analysis**: Adaptive analysis based on changing market conditions
- **Intelligent Alerting**: Smart notification system for critical business events
- **Interactive Dashboards**: Real-time visualization of multi-agent analysis results

### Enterprise-Grade Coordination
- **Scalable Architecture**: Handle complex, multi-faceted business challenges
- **Quality Assurance**: Multi-tier validation and quality control processes
- **Error Recovery**: Sophisticated error handling and workflow recovery
- **Performance Analytics**: Comprehensive metrics and performance monitoring

### Advanced Communication Protocols
- **Inter-Agent Messaging**: Sophisticated communication between agents
- **Event Broadcasting**: Real-time event distribution and coordination
- **Result Aggregation**: Intelligent synthesis of distributed agent outputs
- **Human Integration**: Seamless human-in-the-loop coordination and oversight