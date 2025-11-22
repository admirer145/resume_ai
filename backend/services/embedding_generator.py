import openai
from typing import List, Dict
import numpy as np

class EmbeddingGenerator:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.model = "text-embedding-3-small"
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        response = openai.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
    
    def batch_generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        response = openai.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]
    
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
