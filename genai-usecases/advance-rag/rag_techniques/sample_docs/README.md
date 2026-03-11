# Sample Documents for RAG Techniques

These documents let you start experimenting immediately — no need to find your own PDFs.

## Files

| File | Topic | Best questions to ask |
|------|-------|----------------------|
| `renewable_energy_report.txt` | Solar, wind, hydro, battery storage, policy | "What are the main renewable energy sources?", "What are the challenges of solar energy?" |
| `machine_learning_guide.txt` | ML types, algorithms, training, evaluation | "What is the difference between supervised and unsupervised learning?", "What is overfitting?" |

## How to use

1. Run any of the RAG technique apps (e.g. `streamlit run basic_rag.py`)
2. In the file uploader, select one or both `.txt` files from this folder
3. Type a question from the table above and click **Ask**

## Tips for testing each technique

- **Basic RAG** — use a simple factual question: *"What is a neural network?"*
- **Hybrid Search RAG** — use a keyword-heavy question: *"LSTM recurrent architecture gradient"*
- **Re-ranking RAG** — use a complex question to see how re-ranking improves answer quality
- **Corrective RAG** — compare the Initial Response vs Final Response tabs to see self-correction in action
- **Adaptive RAG** — try different question types and watch the complexity badge (🟢/🟡🔴) change
