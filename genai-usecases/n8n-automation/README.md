# 🚀 n8n Automation Templates & Workflow Guide

[![n8n](https://img.shields.io/badge/n8n-FF6D5A?style=for-the-badge&logo=n8n&logoColor=white)](https://n8n.io)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Node.js](https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=nodedotjs&logoColor=white)](https://nodejs.org)

> **Explore 4,200+ n8n automation templates with detailed setup instructions for both self-hosted and cloud environments.**

---

## 📋 Table of Contents

- [Template Library Highlights](#-template-library-highlights)
- [Getting Started](#-getting-started)
- [Installation Methods](#-installation-methods)
- [AI-Enhanced Setup](#-ai-enhanced-setup)
- [Template Categories & Examples](#-template-categories--examples)
- [Workflow Creation Process](#-workflow-creation-process)
- [Configuration](#-configuration)
- [Contributing Templates](#-add-your-own-templates)
- [Troubleshooting](#-troubleshooting)
- [Project Structure](#-sample-project-structure)
- [Resources](#-resources)

---

## 📚 Template Library Highlights

- **4,200+ Official Templates**: Browse the full n8n template library - spanning categories like AI, sales, marketing, document ops, support, and more → [n8n.io/workflows](https://n8n.io/workflows)
- **Community Collections**:
  - [Awesome n8n Templates](https://github.com/enescingoz/awesome-n8n-templates)
  - [Professional Collection](https://github.com/Zie619/n8n-workflows)

---

## 🛠 Getting Started

### Local (Self-Hosted)
- **Try instantly** with NPX:
  ```bash
  npx n8n
  ```
- **Recommended (Docker)**:
  ```bash
  docker run -it --rm --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n
  ```
- **Import Workflows** via UI or API using `.json` files

### Cloud (n8n.io)
- Sign up at [n8n Cloud](https://app.n8n.cloud)
- Click **Create New Workflow → Template** to browse and import

---

## 🔧 Installation Methods

### Docker Compose (with PostgreSQL)
```yaml
# docker-compose.yml
services:
  n8n:
    image: n8nio/n8n:latest
    ports: ["5678:5678"]
    environment:
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_DATABASE=n8n
      - DB_POSTGRESDB_USER=${POSTGRES_USER}
      - DB_POSTGRESDB_PASSWORD=${POSTGRES_PASSWORD}
      - N8N_ENCRYPTION_KEY=${N8N_ENCRYPTION_KEY}
      - N8N_TEMPLATES_ENABLED=true
    volumes:
      - n8n_data:/home/node/.n8n
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=n8n
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
volumes:
  n8n_data:
  postgres_data:
```

---

## 🤖 AI-Enhanced Setup

### AI Starter Kit
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

### Comprehensive Setup
```bash
git clone https://github.com/kossakovsky/n8n-installer.git
cd n8n-installer
chmod +x install.sh && ./install.sh
```

Adds:
- Flowise, Supabase, Qdrant, Langfuse, OpenWebUI, Caddy, etc.

---

## 📦 Template Categories & Examples

| Category            | Count | Examples                                       |
|---------------------|-------|------------------------------------------------|
| 🤖 AI & LLMs         | 500+  | OpenAI, Claude, Gemini, chatbot, RAG           |
| 📧 Email / CRM       | 1,188 | Gmail triage, email-to-sheets                  |
| 📱 Social Media      | 800+  | X, IG, Telegram bots, LinkedIn posts           |
| 📄 Document Ops      | 477+  | PDF parsing, Excel tools, content extraction   |
| 🛠️ Support / Ops     | 397+  | Slack bots, SSL monitoring, feedback systems   |

---

## ✅ Workflow Creation Process

1. Pick a template or build from scratch
2. Import JSON or use cloud library
3. Authenticate API nodes
4. Set up triggers (cron, webhook, etc.)
5. Customize nodes
6. Test and activate
7. Optionally back up via GitHub

---

## ⚙️ Configuration

### `.env` Basics
```env
N8N_ENCRYPTION_KEY=your-256-key
N8N_HOST=your-domain.com
DB_TYPE=postgresdb
DB_POSTGRESDB_USER=n8n_user
DB_POSTGRESDB_PASSWORD=secure_password
GENERIC_TIMEZONE=America/New_York
```

### Optional Features
```env
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=secure_pass

EXECUTIONS_MODE=queue
QUEUE_BULL_REDIS_HOST=redis
```

---

## 🧪 Add Your Own Templates

### Creator Program
- Join at [n8n.io/creators](https://n8n.io/creators)
- Submit workflows via docs portal
- Get listed and earn via affiliate program

---

## 🔍 Troubleshooting

### Common Fixes
- Clear cache:
  ```bash
  docker exec -it n8n rm -rf /home/node/.n8n/cache/templates
  ```
- Validate workflows:
  ```bash
  curl -X POST http://localhost:5678/api/v1/workflows/validate -d @workflow.json
  ```

### Debugging
```bash
export N8N_LOG_LEVEL=debug
export N8N_LOG_OUTPUT=console
```

---

## 📂 Sample Project Structure

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

## 📖 Resources

### Official
- [Docs](https://docs.n8n.io)
- [Workflow Library](https://n8n.io/workflows)
- [GitHub](https://github.com/n8n-io/n8n)

### Templates
- [Awesome Templates](https://github.com/enescingoz/awesome-n8n-templates)
- [Professional Collection](https://github.com/Zie619/n8n-workflows)

### Community
- [Forum](https://community.n8n.io)
- [YouTube](https://youtube.com/@n8n_io)
- [Twitter](https://twitter.com/n8n_io)

---

## Next Steps

1. Set up n8n (Docker is best)
2. Configure `.env` and volumes
3. Import or build workflows
4. Run and customize
5. Share your templates 
