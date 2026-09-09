# Multi-Source RAG with LangChain

A Retrieval-Augmented Generation (RAG) application that retrieves information from multiple sources:

- PubMed
- YouTube transcripts
- Tavily Search
- DuckDuckGo Search

The retrieved context is passed to GPT-4o-mini to generate grounded answers.

## Features

- PubMed medical research retrieval
- YouTube transcript search
- Web search via Tavily
- Web search via DuckDuckGo
- FAISS vector database
- OpenAI embeddings
- LangChain pipeline

## Architecture

User Query
     │
     ▼
 Selected Retriever
     │
     ├── PubMed
     ├── YouTube + FAISS
     ├── Tavily
     └── DuckDuckGo
     │
     ▼
 Retrieved Context
     │
     ▼
 GPT-4o-mini
     │
     ▼
 Final Answer

## Installation

pip install -r requirements.txt

## Environment Variables

Create a .env file:

OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key

## Run

python app.py

## Tech Stack

- Python
- LangChain
- OpenAI
- FAISS
- Tavily
- DuckDuckGo
- PubMed

## Future Improvements

- Router chain for automatic source selection
- Multi-source retrieval
- LangGraph agent workflow
- Streamlit UI