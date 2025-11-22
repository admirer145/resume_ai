import weaviate
import weaviate.classes as wvc
from typing import List, Dict
from utils.logger import logger


class WeaviateManager:
    def __init__(self, url: str):
        # Connect to local Weaviate instance
        self.client = weaviate.connect_to_local()
        logger.info(f"Weaviate client initialized, ready: {self.client.is_ready()}")
        self._create_collections()
    
    def _create_collections(self):
        """Create Weaviate collections for resume data"""
        try:
            # Check if ResumeChunk collection exists
            if not self.client.collections.exists("ResumeChunk"):
                self.client.collections.create(
                    name="ResumeChunk",
                    description="A chunk of resume text",
                    vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # We provide embeddings
                    properties=[
                        wvc.config.Property(
                            name="content",
                            data_type=wvc.config.DataType.TEXT,
                            description="The actual text content"
                        ),
                        wvc.config.Property(
                            name="section",
                            data_type=wvc.config.DataType.TEXT,
                            description="Resume section (skills, experience, etc.)"
                        ),
                        wvc.config.Property(
                            name="resume_id",
                            data_type=wvc.config.DataType.TEXT,
                            description="Unique resume identifier"
                        ),
                        wvc.config.Property(
                            name="metadata",
                            data_type=wvc.config.DataType.TEXT,
                            description="Additional context as JSON"
                        )
                    ]
                )
                logger.info("Created ResumeChunk collection")
            
            # Check if JobRequirement collection exists
            if not self.client.collections.exists("JobRequirement"):
                self.client.collections.create(
                    name="JobRequirement",
                    description="A requirement from job description",
                    vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                    properties=[
                        wvc.config.Property(
                            name="requirement",
                            data_type=wvc.config.DataType.TEXT,
                            description="The requirement text"
                        ),
                        wvc.config.Property(
                            name="category",
                            data_type=wvc.config.DataType.TEXT,
                            description="Type: skill, responsibility, qualification"
                        ),
                        wvc.config.Property(
                            name="jd_id",
                            data_type=wvc.config.DataType.TEXT,
                            description="Job description identifier"
                        )
                    ]
                )
                logger.info("Created JobRequirement collection")
                
        except Exception as e:
            logger.warning(f"Collection creation skipped (may already exist): {e}")
    
    def add_resume_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Add resume chunks with embeddings using Weaviate v4 API"""
        try:
            collection = self.client.collections.get("ResumeChunk")
            
            # Prepare data objects for batch insert
            data_objects = []
            for chunk, embedding in zip(chunks, embeddings):
                data_objects.append(
                    wvc.data.DataObject(
                        properties={
                            "content": chunk['content'],
                            "section": chunk['section'],
                            "resume_id": chunk['resume_id'],
                            "metadata": chunk.get('metadata', '{}')
                        },
                        vector=embedding
                    )
                )
            
            # Batch insert
            response = collection.data.insert_many(data_objects)
            
            # Check for errors
            if response.has_errors:
                logger.error(f"Batch insert had errors: {response.errors}")
            else:
                logger.info(f"Successfully inserted {len(data_objects)} resume chunks")
                
        except Exception as e:
            logger.exception(f"Error adding resume chunks: {e}")
            raise e
    
    def semantic_search(self, query_embedding: List[float], 
                       section_filter: str = None, 
                       limit: int = 5) -> List[Dict]:
        """Search for relevant resume chunks using Weaviate v4 API"""
        try:
            collection = self.client.collections.get("ResumeChunk")
            
            # Build query
            if section_filter:
                response = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=limit,
                    filters=wvc.query.Filter.by_property("section").equal(section_filter)
                )
            else:
                response = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=limit
                )
            
            # Convert response to expected format
            results = []
            for obj in response.objects:
                results.append({
                    "content": obj.properties.get("content", ""),
                    "section": obj.properties.get("section", ""),
                    "metadata": obj.properties.get("metadata", "{}")
                })
            
            return results
            
        except Exception as e:
            logger.exception(f"Error in semantic search: {e}")
            return []
    
    def __del__(self):
        """Close Weaviate connection"""
        try:
            if hasattr(self, 'client'):
                self.client.close()
        except:
            pass
