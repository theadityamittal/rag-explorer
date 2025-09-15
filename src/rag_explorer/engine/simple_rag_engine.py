"""Simplified RAG Engine for personal use."""

import os
import logging
from typing import Dict, List, Optional
from pathlib import Path

from ..utils.simple_settings import (
    PRIMARY_LLM_PROVIDER,
    PRIMARY_EMBEDDING_PROVIDER,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    OLLAMA_HOST,
    OLLAMA_LLM_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CONFIDENCE,
    MAX_CHUNKS,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION
)
from ..data.chunker import chunk_text
from ..data.store import get_client, get_collection, upsert_chunks, query_by_embedding

logger = logging.getLogger(__name__)

class SimpleRAGEngine:
    """Simplified RAG engine for personal document Q&A."""
    
    def __init__(self):
        """Initialize the RAG engine with available providers."""
        self.providers = self._init_providers()
        self._ensure_chroma_db()
    
    def _init_providers(self) -> Dict:
        """Initialize available providers based on configuration."""
        providers = {}
        
        # Initialize Ollama if available
        try:
            if self._check_ollama_available():
                providers['ollama'] = self._create_ollama_provider()
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
        
        # Initialize API providers if keys are available
        if OPENAI_API_KEY:
            try:
                providers['openai'] = self._create_openai_provider()
            except Exception as e:
                logger.warning(f"OpenAI provider failed: {e}")
        
        if ANTHROPIC_API_KEY:
            try:
                providers['anthropic'] = self._create_anthropic_provider()
            except Exception as e:
                logger.warning(f"Anthropic provider failed: {e}")
        
        if GOOGLE_API_KEY:
            try:
                providers['google'] = self._create_google_provider()
            except Exception as e:
                logger.warning(f"Google provider failed: {e}")
        
        if not providers:
            raise RuntimeError("No providers available. Please configure at least one provider.")
        
        return providers
    
    def _ensure_chroma_db(self):
        """Ensure ChromaDB directory exists."""
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            import requests
            response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _create_ollama_provider(self):
        """Create Ollama provider."""
        import ollama
        return {
            'type': 'ollama',
            'client': ollama.Client(host=OLLAMA_HOST),
            'llm_model': OLLAMA_LLM_MODEL,
            'embedding_model': OLLAMA_EMBEDDING_MODEL
        }
    
    def _create_openai_provider(self):
        """Create OpenAI provider."""
        import openai
        return {
            'type': 'openai',
            'client': openai.OpenAI(api_key=OPENAI_API_KEY),
            'llm_model': 'gpt-4o-mini',
            'embedding_model': 'text-embedding-3-small'
        }
    
    def _create_anthropic_provider(self):
        """Create Anthropic provider."""
        import anthropic
        return {
            'type': 'anthropic',
            'client': anthropic.Anthropic(api_key=ANTHROPIC_API_KEY),
            'llm_model': 'claude-3-haiku-20240307',
            'embedding_model': None  # Anthropic doesn't provide embeddings
        }
    
    def _create_google_provider(self):
        """Create Google provider."""
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        return {
            'type': 'google',
            'client': genai,
            'llm_model': 'gemini-1.5-flash',
            'embedding_model': 'text-embedding-004'
        }
    
    def _get_provider(self, provider_type: str):
        """Get provider with fallback logic."""
        if provider_type == 'llm':
            preferred = PRIMARY_LLM_PROVIDER
        else:
            preferred = PRIMARY_EMBEDDING_PROVIDER
        
        # Try preferred provider first
        if preferred in self.providers:
            provider = self.providers[preferred]
            if provider_type == 'embedding' and provider.get('embedding_model') is None:
                # Anthropic doesn't have embeddings, fall back
                pass
            else:
                return provider
        
        # Fallback to any available provider
        for name, provider in self.providers.items():
            if provider_type == 'embedding' and provider.get('embedding_model') is None:
                continue
            return provider
        
        raise RuntimeError(f"No {provider_type} provider available")
    
    def index_documents(self, docs_path: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> Dict:
        """Index documents from a directory."""
        docs_path_obj = Path(docs_path)
        if not docs_path_obj.exists():
            raise FileNotFoundError(f"Directory not found: {docs_path}")
        
        chunk_size = chunk_size or CHUNK_SIZE
        chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        
        # Load documents
        documents = self._load_documents(docs_path_obj)
        if not documents:
            raise ValueError(f"No documents found in {docs_path}")
        
        # Chunk documents
        all_chunks = []
        chunk_metadatas = []
        chunk_ids = []
        
        for file_path, content in documents.items():
            chunks = chunk_text(content, chunk_size=chunk_size, overlap=chunk_overlap)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_path}_{i}"
                all_chunks.append(chunk)
                chunk_metadatas.append({
                    'source': str(file_path),
                    'chunk_id': chunk_id,
                    'chunk_index': i
                })
                chunk_ids.append(chunk_id)
        
        # Generate embeddings
        embeddings = self._generate_embeddings(all_chunks)
        
        # Store in ChromaDB
        upsert_chunks(all_chunks, chunk_metadatas, chunk_ids)
        
        return {
            'count': len(all_chunks),
            'files_processed': len(documents)
        }
    
    def answer_question(self, question: str, min_confidence: float = None, max_chunks: int = None) -> Dict:
        """Answer a question using RAG."""
        min_confidence = min_confidence or MIN_CONFIDENCE
        max_chunks = max_chunks or MAX_CHUNKS
        
        # Generate query embedding
        query_embedding = self._generate_query_embedding(question)
        
        # Retrieve relevant chunks
        hits = query_by_embedding(query_embedding, k=max_chunks)
        
        if not hits:
            return {
                'answer': "I don't have any documents indexed to answer questions.",
                'confidence': 0.0,
                'sources': []
            }
        
        # Calculate confidence
        confidence = self._calculate_confidence(hits, question)
        
        # Generate answer if confidence is high enough
        if confidence >= min_confidence:
            context = self._format_context(hits)
            answer = self._generate_answer(question, context)
        else:
            answer = "I don't have enough information to answer that question."
        
        return {
            'answer': answer,
            'confidence': confidence,
            'sources': [hit['meta']['source'] for hit in hits[:3]]
        }
    
    def get_status(self) -> Dict:
        """Get system status."""
        try:
            client = get_client()
            collection = get_collection(client)
            doc_count = collection.count()
            db_connected = True
        except:
            doc_count = 0
            db_connected = False
        
        available_providers = {
            'llm': [],
            'embedding': []
        }
        
        for name, provider in self.providers.items():
            available_providers['llm'].append(name)
            if provider.get('embedding_model'):
                available_providers['embedding'].append(name)
        
        return {
            'llm_provider': PRIMARY_LLM_PROVIDER,
            'embedding_provider': PRIMARY_EMBEDDING_PROVIDER,
            'db_connected': db_connected,
            'doc_count': doc_count,
            'available_providers': available_providers
        }
    
    def _load_documents(self, docs_path: Path) -> Dict[str, str]:
        """Load documents from directory."""
        documents = {}
        
        for file_path in docs_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.md', '.rst']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            documents[str(file_path.relative_to(docs_path))] = content
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
        
        return documents
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        provider = self._get_provider('embedding')
        
        if provider['type'] == 'ollama':
            embeddings = []
            for text in texts:
                response = provider['client'].embeddings(
                    model=provider['embedding_model'],
                    prompt=text
                )
                embeddings.append(response['embedding'])
            return embeddings
        
        elif provider['type'] == 'openai':
            response = provider['client'].embeddings.create(
                model=provider['embedding_model'],
                input=texts
            )
            return [item.embedding for item in response.data]
        
        elif provider['type'] == 'google':
            embeddings = []
            for text in texts:
                result = provider['client'].embed_content(
                    model=f"models/{provider['embedding_model']}",
                    content=text
                )
                embeddings.append(result['embedding'])
            return embeddings
        
        else:
            raise RuntimeError(f"Embedding not supported for provider: {provider['type']}")
    
    def _generate_query_embedding(self, question: str) -> List[float]:
        """Generate embedding for query."""
        return self._generate_embeddings([question])[0]
    
    def _calculate_confidence(self, hits: List[Dict], question: str) -> float:
        """Calculate confidence score based on similarity."""
        if not hits:
            return 0.0
        
        # Use the distance from the top hit
        top_hit = hits[0]
        distance = top_hit.get('distance', 1.0)
        
        # Convert distance to similarity (lower distance = higher similarity)
        similarity = 1.0 / (1.0 + max(0.0, distance))
        
        return round(similarity, 3)
    
    def _format_context(self, hits: List[Dict]) -> str:
        """Format retrieved chunks into context."""
        context_parts = []
        for i, hit in enumerate(hits, 1):
            source = hit['meta']['source']
            text = hit['text'][:500]  # Limit chunk size
            context_parts.append(f"[{i}] From {source}:\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM."""
        provider = self._get_provider('llm')
        
        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Use only the information in the context to answer. If the context doesn't contain "
            "enough information to answer the question, say so clearly."
        )
        
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        
        try:
            if provider['type'] == 'ollama':
                response = provider['client'].chat(
                    model=provider['llm_model'],
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ]
                )
                return response['message']['content']
            
            elif provider['type'] == 'openai':
                response = provider['client'].chat.completions.create(
                    model=provider['llm_model'],
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    temperature=0.0
                )
                return response.choices[0].message.content
            
            elif provider['type'] == 'anthropic':
                response = provider['client'].messages.create(
                    model=provider['llm_model'],
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[
                        {'role': 'user', 'content': user_prompt}
                    ]
                )
                return response.content[0].text
            
            elif provider['type'] == 'google':
                model = provider['client'].GenerativeModel(provider['llm_model'])
                response = model.generate_content(f"{system_prompt}\n\n{user_prompt}")
                return response.text
            
            else:
                return "Error: Unsupported LLM provider"
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
