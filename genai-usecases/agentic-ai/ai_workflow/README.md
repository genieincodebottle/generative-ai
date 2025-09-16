<div align="center">
   <img src="../../../images/ai_workflow.png" alt="AI Workflow Patterns" />
</div>

# 📊 AI Workflow - Business Intelligence System

A comprehensive business intelligence system demonstrating various agentic AI workflow patterns for analyzing business documents, generating insights, and providing strategic recommendations.

## 🎯 Overview

This application showcases six different AI workflow patterns applied to business document analysis, from simple prompt chaining to complex orchestration with tool calling. Each pattern demonstrates progressively more sophisticated approaches to handling business intelligence tasks.

## ✨ Features

- **Multi-Pattern Support**: Six different agentic workflow patterns
- **Business Document Analysis**: Analyze quarterly reports, business plans, market analysis, and financial documents
- **Real-time Processing**: Dynamic analysis with configurable AI models
- **Interactive Interface**: User-friendly Streamlit interface with validation and guidance
- **Export Capabilities**: Download analysis results in multiple formats
- **Flexible Input**: Support for custom documents or sample business reports

### ⚙️ Setup Instructions

- #### Prerequisites
   - Python 3.9 or higher
   - pip (Python package installer)

- #### Installation
   1. Clone the repository:
      ```bash
      git clone https://github.com/genieincodebottle/generative-ai.git
      cd generative-ai/genai-usecases/agentic-ai/ai_workflow
      ```
   2. Open the Project in VS Code or any code editor.
   3. Create a virtual environment by running the following command in the terminal:
      ```bash
      pip install uv #if uv not installed
      uv venv
      .venv\Scripts\activate # On Linux -> source venv/bin/activate
      ```
   4. Create a requirements.txt file and add the following libraries:
      
      ```bash
        streamlit>=1.47.1 
        langchain>=0.3.27 
        langchain-google-genai>=2.1.8 
        langchain-chroma>=0.2.5 
        langchain-community>=0.3.27
        nest-asyncio>=1.6.0
        pypdf>=5.9.0
        python-dotenv>=1.0.1
        flashrank>=0.2.10
        rank_bm25>=0.2.2
      ```
   5 Install dependencies:
      ```bash
      uv pip install -r requirements.txt
      ```
   6. Configure Environment
      * Rename .env.example → .env
      * Update with your keys:

      ```bash
      GROQ_API_KEY=your_api_key
      OPENAI_API_KEY=your_api_key
      GOOGLE_API_KEY=your_api_key
      ANTHROPIC_API_KEY=your_api_key
      ```
      - For **GROQ_API_KEY** follow this -> https://console.groq.com/keys
      - For **OPENAI_API_KEY** follow this -> https://platform.openai.com/api-keys
      - For **GOOGLE_API_KEY** follow this -> https://ai.google.dev/gemini-api/docs/api-key
      - For **ANTHROPIC_API_KEY** follow this -> https://console.anthropic.com/settings/keys
<hr>

### 💻 Running the Application
To start the application, run:
```bash
streamlit run app.py
```
<hr>

### AI Workflows 

Reference: [Anthropic's Guide to Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)

### 🔗 Prompt Chaining
- Sequential prompts for comprehensive business document analysis
- Use when tasks can be split into fixed subtasks, trading latency for accuracy by simplifying each LLM call.
- **Business Application**: 
   -  Extract business data → Analyze trends and patterns → Generate strategic recommendations
   -  Review financial metrics → Identify risks and opportunities → Create action items
   -  Process quarterly report → Assess performance → Provide executive summary
- **Implementation**: Three-step analysis chain for business documents:
   1. **Data Extraction**: Extract financial metrics, KPIs, and strategic elements
   2. **Pattern Analysis**: Identify trends, strengths, weaknesses, and root causes  
   3. **Report Generation**: Create comprehensive business analysis with recommendations
<br>

<img src="../../../images/prompt_chaining.png" alt="Prompt Chaining" />

### 🔀 Parallelization 
- Concurrent business analysis tasks for comprehensive evaluation
- Use when tasks can run in parallel for speed or when multiple perspectives improve accuracy.
- Types:
   -  **Sectioning**: Split business analysis into independent specialized reviews
   -  **Voting**: Multiple perspective analysis for diverse insights
- **Business Applications**:
   -  **Sectioning**: Parallel review for compliance, accuracy, safety, implementation risks, and ethics
   -  **Voting**: Multiple analysts review financial projections or strategic decisions for consensus
   -  **Specialized Analysis**: Concurrent financial, operational, and strategic assessments
<br>

<img src="../../../images/parallelization.png" alt="Parallelization" />

### 📡 Query Routing
- Dynamic business query distribution based on content and complexity
- Use when business documents fall into distinct categories best handled by specialized analyzers.
- **Business Applications**:
   - Route financial reports to financial analysis specialists, market research to competitive analysis
   - Direct simple quarterly updates to lightweight processors, complex merger analysis to sophisticated models  
   - Classify document types (financial, operational, strategic) for appropriate specialized handling
   - Route urgent business decisions to high-performance models, routine analysis to cost-effective alternatives
<br>

<img src="../../../images/routing.png" alt="Query Routing" />

### 📈 Evaluator/Optimizer
- Quality control and iterative improvement for business analysis
- Use when business insights can be enhanced through evaluation criteria and feedback loops.
- **Business Applications**:
   - Evaluate business recommendations for feasibility, impact, and alignment with company strategy
   - Iteratively refine financial projections based on accuracy and completeness criteria
   - Assess strategic plans for clarity, actionability, and risk mitigation
   - Optimize market analysis reports through multiple evaluation rounds for depth and insight quality
<br>

<img src="../../../images/eval.png" alt="Evaluator and Optimizer" />

### 🎼 Orchestrator 
- Complex business workflow management with dynamic task decomposition
- Use when business analysis requires dynamic breakdown with specialized teams handling different aspects.
- **Business Applications**:
   -  Orchestrate comprehensive due diligence with specialized teams for financial, legal, and operational analysis
   -  Coordinate market entry strategy with teams analyzing competition, regulations, and customer segments
   -  Manage complex business transformations by assigning workstreams to experts in different business functions
   -  Dynamic resource allocation for business cases based on complexity, industry, and strategic importance
<br>

<img src="../../../images/orchestrator.png" alt="Orchestartor" />

### 📞 Tool Calling 
- External business tool integration for enhanced analysis capabilities
- Use when business analysis requires real-time data, precise calculations, or external system interactions.
- **Business Applications**:
   - Integrate with financial APIs for real-time market data, stock prices, and economic indicators
   - Connect to CRM systems for customer data analysis and performance metrics
   - Access business intelligence tools for advanced analytics and data visualization
   - Utilize calculation engines for complex financial modeling, ROI analysis, and forecasting
   - Interface with project management systems to update schedules and resource allocation
   - Call external compliance and risk assessment tools for regulatory analysis
<br>

<img src="../../../images/tool_calling.png" alt="Tool Calling" />

---

## 📋 Supported Document Types & Use Cases

### 📊 **Quarterly/Annual Business Reviews**
- Financial performance analysis
- Department performance assessment
- KPI tracking and trend analysis
- Strategic goal evaluation

### 💼 **Strategic Planning Documents**
- Market opportunity analysis
- Competitive positioning assessment
- Resource allocation planning
- Risk identification and mitigation

### 💰 **Financial Reports**
- Revenue and profitability analysis
- Cost structure evaluation
- Cash flow assessment
- Financial ratio analysis

### 📈 **Market Analysis Reports**
- Competitive landscape evaluation
- Market trend identification
- Customer segment analysis
- Growth opportunity assessment

### ⚙️ **Operational Assessment Documents**
- Process efficiency analysis
- Resource utilization evaluation
- Performance bottleneck identification
- Operational improvement recommendations

---

## 🚀 Getting Started

1. **Start the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Select Workflow Pattern**: Choose from 6 different agentic patterns

3. **Input Business Document**: Use the sample document or upload your own

4. **Configure Analysis**: Set model parameters and analysis options

5. **Review Results**: Get comprehensive insights, recommendations, and downloadable reports

---

## 🎯 Key Benefits

- **Comprehensive Analysis**: Multi-dimensional business document evaluation
- **Scalable Patterns**: From simple chains to complex orchestration
- **Flexible Input**: Support for various business document types
- **Actionable Insights**: Strategic recommendations with clear action items
- **Export Ready**: Professional reports in multiple formats