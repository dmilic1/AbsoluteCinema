# AbsoluteCinema AI: Hybrid RAG Movie Agent

A movie recommendation engine using **Retrieval-Augmented Generation (RAG)**. This agent combines local semantic search with OpenAI's **GPT-5.4 Nano** (March 2026 release) to provide grounded, witty film suggestions.

Check out the app here:  
👉 [AbsoluteCinema AI](https://absolutecinema-ai-agent.streamlit.app/)
## 🚀 Key Features
- **Semantic Search:** Uses `sentence-transformers` (SBERT) and KNN to find movies based on meaning, not just keywords.
- **Hybrid Architecture:** Local vector retrieval + Cloud-based LLM reasoning for cost-effective performance.
- **Security-First:** Implemented prompt injection delimiters and token-usage guards.
- **Object-Oriented Design:** Encapsulated logic within a `AbsoluteCinemaAgent` class for modularity.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **LLM:** OpenAI GPT-5.4 Nano
- **Embeddings:** `all-MiniLM-L6-v2` (SBERT)
- **Vector Search:** Scikit-Learn (Nearest Neighbors)
- **Data:** Pandas (optimized for 2026 `StringDtype` compatibility)

## 🏗️ Architecture
The app follows a RAG pipeline:
1. **User Query** is embedded via SBERT.
2. **KNN Search** retrieves the top 3 matches from a local 5,000-movie metadata "soup."
3. **Context Injection:** Movie metadata is fed into GPT-5.4 Nano with strict system instructions.
4. **Grounded Response:** The AI generates a response strictly based on the retrieved data.
