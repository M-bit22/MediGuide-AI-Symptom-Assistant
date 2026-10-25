# MediGuide AI - Al-Powered Medical Symptom Assessment Assistant

An educational Streamlit application powered by LangChain and Google Gemini API that collects patient symptoms and outputs structured, safety-first preliminary health guidance.

## Features
- **Streamlit UI**: Complete sidebar settings, responsive widgets, metrics, expandable condition cards, and tabs.
- **LangChain Integration**: Utilizes `ChatGoogleGenerativeAI`, `ChatPromptTemplate`, and LCEL chains.
- **Streaming Support**: Live response streaming using `st.write_stream()`.
- **Caching**: Supports both `InMemoryCache` (RAM) and `SQLiteCache` (Disk) for fast identical query retrieval.
- **Safety-First**: Enforces strict medical disclaimers and avoids confirmed diagnoses.

## Setup Instructions
1. Clone the repository and create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt