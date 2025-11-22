import weaviate
from typing import List, Dict


class WeaviateManager:
    def __init__(self, url: str):
        self.client = weaviate.connect_to_local()
        print("Weaviate client initialized", self.client.is_ready())
        self._create_schema()
    
    def _create_schema(self):
        """Create Weaviate schema for resume data"""
        schema = {
            "classes": [
                {
                    "class": "ResumeChunk",
                    "description": "A chunk of resume text",
                    "vectorizer": "none",  # We provide embeddings
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "The actual text content"
                        },
                        {
                            "name": "section",
                            "dataType": ["string"],
                            "description": "Resume section (skills, experience, etc.)"
                        },
                        {
                            "name": "resume_id",
                            "dataType": ["string"],
                            "description": "Unique resume identifier"
                        },
                        {
                            "name": "metadata",
                            "dataType": ["text"],
                            "description": "Additional context as JSON"
                        }
                    ]
                },
                {
                    "class": "JobRequirement",
                    "description": "A requirement from job description",
                    "vectorizer": "none",
                    "properties": [
                        {
                            "name": "requirement",
                            "dataType": ["text"],
                            "description": "The requirement text"
                        },
                        {
                            "name": "category",
                            "dataType": ["string"],
                            "description": "Type: skill, responsibility, qualification"
                        },
                        {
                            "name": "jd_id",
                            "dataType": ["string"],
                            "description": "Job description identifier"
                        }
                    ]
                }
            ]
        }
        
        # Create schema if not exists
        try:
            self.client.schema.create(schema)
        except:
            pass  # Schema already exists
    
    def add_resume_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Add resume chunks with embeddings"""
        with self.client.batch as batch:
            for chunk, embedding in zip(chunks, embeddings):
                batch.add_data_object(
                    data_object={
                        "content": chunk['content'],
                        "section": chunk['section'],
                        "resume_id": chunk['resume_id'],
                        "metadata": chunk.get('metadata', '{}')
                    },
                    class_name="ResumeChunk",
                    vector=embedding
                )
    
    def semantic_search(self, query_embedding: List[float], 
                       section_filter: str = None, 
                       limit: int = 5) -> List[Dict]:
        """Search for relevant resume chunks"""
        query = (
            self.client.query
            .get("ResumeChunk", ["content", "section", "metadata"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(limit)
        )
        
        if section_filter:
            query = query.with_where({
                "path": ["section"],
                "operator": "Equal",
                "valueString": section_filter
            })
        
        result = query.do()
        return result.get('data', {}).get('Get', {}).get('ResumeChunk', [])
