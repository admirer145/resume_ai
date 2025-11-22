from typing import List
import config

class EmbeddingGenerator:
    def __init__(self, provider: str = None, api_key: str = None, base_url: str = None):
        """
        Initialize embedding generator with support for multiple providers.
        
        Args:
            provider: "openai" or "ollama" (defaults to config.MODEL_PROVIDER)
            api_key: API key for OpenAI (only needed if provider is "openai")
            base_url: Base URL for Ollama (only needed if provider is "ollama")
        """
        self.provider = provider or config.MODEL_PROVIDER
        self.model = config.get_embedding_model()
        
        if self.provider == "openai":
            import openai
            openai.api_key = api_key or config.OPENAI_API_KEY
            self.client = openai
        else:  # ollama
            import ollama
            self.base_url = base_url or config.OLLAMA_BASE_URL
            self.client = ollama.Client(host=self.base_url)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if self.provider == "openai":
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        else:  # ollama
            response = self.client.embeddings(
                model=self.model,
                prompt=text
            )
            return response['embedding']
    
    def batch_generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if self.provider == "openai":
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [item.embedding for item in response.data]
        else:  # ollama
            # Ollama doesn't support batch, so we process one by one
            embeddings = []
            for text in texts:
                response = self.client.embeddings(
                    model=self.model,
                    prompt=text
                )
                embeddings.append(response['embedding'])
            return embeddings
    
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into semantic chunks"""
        # Simple sentence-based chunking
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
