"""Simple settings loading from environment variables."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Provider settings
PRIMARY_LLM_PROVIDER = os.getenv("PRIMARY_LLM_PROVIDER", "ollama")
PRIMARY_EMBEDDING_PROVIDER = os.getenv("PRIMARY_EMBEDDING_PROVIDER", "ollama")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ollama settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# RAG settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.25"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "5"))

# Paths
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "./docs")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION = "documents"
