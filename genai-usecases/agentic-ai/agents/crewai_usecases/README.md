

<div align="center">
   <img src="https://github.com/genieincodebottle/generative-ai/blob/main/images/crewai_header.png" alt="CrewAI Header" />
</div>

### 📄 [CrewAI Documentation](https://docs.crewai.com/introduction)

### 🧩 CrewAI Architecture

   <img src="https://github.com/genieincodebottle/generative-ai/blob/main/images/crewai-architecture.png" alt="CrewAI Architecture"/>

### 🔍 CrewAI Overview

- A framework for creating AI agent teams that work together on complex tasks
- Agents have specific roles and can collaborate using various tools
- Similar to a company with different departments working together

### 🔑 Core Components

- **Crew**: The top-level organization that manages teams of AI agents
- **Agents**: Team members with specific roles (like researcher, writer)
- **Tasks**: Individual assignments given to agents
- **Process**: Controls how tasks are executed (sequential or hierarchical)

### 🤖 Agents

- Have defined roles, goals, and backstories
- Can use tools to accomplish tasks
- Can maintain memory of interactions
- Can be configured with different language models
- Can be given specific knowledge sources

### 📋 Tasks

- Have clear descriptions and expected outputs
- Can be assigned to specific agents
- Can include tools and context from other tasks
- Can produce output in various formats (raw text, JSON, etc.)
- Can be executed sequentially or hierarchically

### 📖 Knowledge

- Allows agents to access external information
- Supports various formats (text, PDF, CSV, etc.)
- Can be shared across agents or specific to individual agents
- Uses embedding models for processing information

### 🧠 Memory

- **Short-term**: Remembers recent interactions
- **Long-term**: Stores valuable insights over time
- **Entity**: Tracks information about specific subjects
- Can be reset or customized as needed

### 🛠️ Tools

- Extend agent capabilities
- Include web searching, data analysis, file reading
- Can be custom-built or use existing ones
- Support caching for better performance
- Include error handling mechanisms

### 🎯 Planning

- Optional feature that helps agents plan tasks step by step
- Adds structure to task execution
- Can use different language models for planning

### 🤝 Collaboration

- Agents can share information and assist each other
- Supports delegation of tasks
- Enables complex workflows through agent interaction

### 🎓 Training

- Allows agents to learn from feedback
- Can be done through CLI or programmatically
- Helps improve agent performance over time