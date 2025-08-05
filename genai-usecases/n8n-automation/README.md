## n8n Automation Templates & Workflow Guide

[![n8n](https://img.shields.io/badge/n8n-FF6D5A?style=for-the-badge&logo=n8n&logoColor=white)](https://n8n.io)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Node.js](https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=nodedotjs&logoColor=white)](https://nodejs.org)

> **Explore 4,200+ n8n automation templates with detailed setup instructions for both self-hosted and cloud environments.**

---

### 📋 Table of Contents

- [Template Library](#-template-library)
- [Getting Started](#-getting-started)
- [AI-Enhanced Setup](#-ai-enhanced-setup)
- [Template Categories & Examples](#-template-categories--examples)
- [Workflow Creation Process](#-workflow-creation-process)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Sample Project Structure](#-sample-project-structure)
- [Resources](#-resources)
- [Contributing Templates](#-add-your-own-templates)

---

### 📚 Template Library

- **4,200+ Official Templates**: Browse the full n8n template library - spanning categories like AI, sales, marketing, document ops, support, and more → [n8n.io/workflows](https://n8n.io/workflows)
- **Community Collections**:
  - [Awesome n8n Templates](https://github.com/enescingoz/awesome-n8n-templates)
  - [Professional Collection](https://github.com/Zie619/n8n-workflows)

---

### 🛠 Getting Started

#### A. Prerequisites

  - Before starting, ensure you have:
    * **Node.js** (version 18.10 or higher) - [Link](https://nodejs.org/)
    * **Docker** (recommended method) - [Link](https://www.docker.com/get-started/)
    * **Git** (to clone workflows from GitHub) - [Link](https://git-scm.com/)
    * A code editor like **VS Code** (optional but helpful)

#### B. Local (Self-Hosted)
  - **Method 1: Try instantly** with NPX: 
  
    * **Step 1:** Install and Run n8n

      ```bash
      npx n8n
      ```
      * This will:
        * Download and install n8n automatically
        * Start the n8n server
        * Open your browser to http://localhost:5678
    * **Step 2:** Initial Setup
      * Create an admin account when prompted
      * Set your email and password
      * You'll see the n8n dashboard

    - **Common NPX Fixes for**

      **Error:** Cannot find module 'ajv/dist/core' or any other issue
      
      * **Option A: Global Install**
        
        ```bash
        npm install -g n8n
        n8n
        ```
      * **Option B: Clear NPX Cache**
        
        ```bash
        npx clear-npx-cache
        npm install ajv #If error: Cannot find module 'ajv/dist/core'
        npx n8n
        ```
      * **Option C: Use Specific Version**

        ```bash
        npm create n8n@latest
        ```
  - **Method 2: Run with Docker (Best for Production)**:
    * **Step 1:** Run n8n with Docker
     
      ```bash
      docker run -it --rm --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n
      ```
    * **Step 2:** Access n8n
      * Open your browser and go to: http://localhost:5678

  - Finding and Using JSON Workflow Templates

    * **Step 1:** Find JSON Templates

      Popular sources for n8n workflow templates:
      * Official n8n Templates: https://n8n.io/workflows/
      * GitHub Repositories:
        * https://github.com/n8n-io/n8n-workflow-template-examples
        * https://github.com/n8n-io/n8n/tree/master/packages/cli/templates
      * Community Templates: Search GitHub for "n8n workflow json"
    * **Step 2:** Download a Sample JSON Template

      Let's use a simple example. Create a file called sample-workflow.json with this content:

      ```bash
      {
        "name": "My First Workflow",
        "nodes": [
          {
            "parameters": {},
            "id": "start-node",
            "name": "When clicking Test workflow",
            "type": "n8n-nodes-base.manualTrigger",
            "typeVersion": 1,
            "position": [240, 300]
          },
          {
            "parameters": {
              "values": {
                "string": [
                  {
                    "name": "message",
                    "value": "Hello from n8n!"
                  }
                ]
              }
            },
            "id": "set-data",
            "name": "Set Data",
            "type": "n8n-nodes-base.set",
            "typeVersion": 1,
            "position": [460, 300]
          }
        ],
        "connections": {
          "When clicking Test workflow": {
            "main": [
              [
                {
                  "node": "Set Data",
                  "type": "main",
                  "index": 0
                }
              ]
            ]
          }
        }
      }
      ```
  - Importing JSON Workflows into n8n

    * **Method A:** Web UI (Easy)
      * Open http://localhost:5678
      * Click New Workflow
      * Click three dots → Import from file
      * Upload JSON or paste content
    * **Method B:** Copy-Paste
      * Open New Workflow
      * Press Ctrl+A → Ctrl+V (paste your JSON)
      * Press Enter

    * **Method C:** API Import (Advanced)
      ```bash
      curl -X POST http://localhost:5678/rest/workflows \
      -H "Content-Type: application/json" \
      -H "X-N8N-API-KEY: your-api-key" \
      -d @sample-workflow.json
      ```
  - Run Your First Workflow

    * Click Test workflow
    * Execute the manual trigger
    * View results as nodes execute
---
#### C. Cloud (n8n.io)
- Sign up at [n8n Cloud](https://app.n8n.cloud)
- Click **Create New Workflow → Template** to browse and import

---

### 🤖 AI-Enhanced Setup

#### AI Starter Kit

```bash
git clone https://github.com/n8n-io/self-hosted-ai-starter-kit.git
cd self-hosted-ai-starter-kit
cp .env.example .env
docker compose --profile cpu up  # or gpu-nvidia or ollama
```

Includes:
- Ollama (Llama 3.2)
- AI workflows
- Shared volume `/data/shared`

#### Comprehensive Setup

```bash
git clone https://github.com/kossakovsky/n8n-installer.git
cd n8n-installer
chmod +x install.sh && ./install.sh
```

Adds:
- Flowise, Supabase, Qdrant, Langfuse, OpenWebUI, Caddy, etc.

---

### 📦 Template Categories & Examples

| Category            | Count | Examples                                       |
|---------------------|-------|------------------------------------------------|
| 🤖 AI & LLMs         | 500+  | OpenAI, Claude, Gemini, chatbot, RAG           |
| 📧 Email / CRM       | 1,188 | Gmail triage, email-to-sheets                  |
| 📱 Social Media      | 800+  | X, IG, Telegram bots, LinkedIn posts           |
| 📄 Document Ops      | 477+  | PDF parsing, Excel tools, content extraction   |
| 🛠️ Support / Ops     | 397+  | Slack bots, SSL monitoring, feedback systems   |

---

### ✅ Workflow Creation Process

1. Pick a template or build from scratch
2. Import JSON or use cloud library
3. Authenticate API nodes
4. Set up triggers (cron, webhook, etc.)
5. Customize nodes
6. Test and activate
7. Optionally back up via GitHub

---

### ⚙️ Configuration

#### `.env` Basics

```env
N8N_ENCRYPTION_KEY=your-256-key
N8N_HOST=your-domain.com
DB_TYPE=postgresdb
DB_POSTGRESDB_USER=n8n_user
DB_POSTGRESDB_PASSWORD=secure_password
GENERIC_TIMEZONE=America/New_York
```

#### Optional Features

```env
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=secure_pass

EXECUTIONS_MODE=queue
QUEUE_BULL_REDIS_HOST=redis
```

---

### 🔍 Troubleshooting

#### Common Fixes
- Clear cache:

  ```bash
  docker exec -it n8n rm -rf /home/node/.n8n/cache/templates
  ```
- Validate workflows:

  ```bash
  curl -X POST http://localhost:5678/api/v1/workflows/validate -d @workflow.json
  ```

#### Debugging
```bash
export N8N_LOG_LEVEL=debug
export N8N_LOG_OUTPUT=console
```

---

### 📂 Sample Project Structure

```text
n8n-automation/
├── workflows/
│   ├── ai/
│   ├── business-ops/
│   ├── social-media/
│   └── integrations/
├── backup/
├── docs/
├── scripts/
├── .env
└── docker-compose.yml
```

---

### 📖 Resources

#### Official
- [Docs](https://docs.n8n.io)
- [Workflow Library](https://n8n.io/workflows)
- [GitHub](https://github.com/n8n-io/n8n)

#### Templates
- [Awesome Templates](https://github.com/enescingoz/awesome-n8n-templates)
- [Professional Collection](https://github.com/Zie619/n8n-workflows)

#### Community
- [Forum](https://community.n8n.io)
- [YouTube](https://youtube.com/@n8n_io)
- [Twitter](https://twitter.com/n8n_io)

---

### 🧪 Add Your Own Templates

#### Creator Program
- Join at [n8n.io/creators](https://n8n.io/creators)
- Submit workflows via docs portal
- Get listed and earn via affiliate program


