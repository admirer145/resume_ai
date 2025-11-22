
from fastapi import UploadFile
import tempfile
import os
import config
from services.pdf_extractor import PDFExtractor
from services.embedding_generator import EmbeddingGenerator
from services.weaviate_manager import WeaviateManager
from services.llm_analyzer import LLMAnalyzer
from services.scoring_engine import ScoringEngine
from services.highlight_generator import HighlightGenerator
from utils.logger import logger
from uuid import uuid4

# Initialize services with config
pdf_extractor = PDFExtractor()
embedding_gen = EmbeddingGenerator()  # Uses config internally
weaviate_mgr = WeaviateManager(url=config.WEAVIATE_URL)
llm_analyzer = LLMAnalyzer()  # Uses config internally
scoring_engine = ScoringEngine()

# Log configuration on startup
logger.info(f"Resume Service initialized with config: {config.get_config_summary()}")



async def analyze_resume_main(resume: UploadFile, jd_text: str):
    """Main endpoint for resume analysis"""
    logger.info(f"Analysis started for resume: {resume.filename}")
    # 1. Extract resume text
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(await resume.read())
        tmp_path = tmp.name
    
    resume_sections = pdf_extractor.extract_text(tmp_path)
    logger.info(resume_sections)
    os.unlink(tmp_path)
    
    # 2. Generate embeddings
    logger.info(f"Generating embeddings...")
    resume_chunks = []
    res_id = str(uuid4())
    for section, content in resume_sections.items():
        if section != 'raw_text' and content.strip():
            chunks = embedding_gen.chunk_text(content)
            for chunk in chunks:
                resume_chunks.append({
                    'content': chunk,
                    'section': section,
                    'resume_id': res_id
                })
    
    chunk_texts = [c['content'] for c in resume_chunks]
    embeddings = embedding_gen.batch_generate(chunk_texts)
    
    # 3. Store in Weaviate
    logger.info(f"Storing embeddings in Weaviate")
    weaviate_mgr.add_resume_chunks(resume_chunks, embeddings)
    
    # 4. Analyze each dimension
    logger.info(f"Analyzing each dimensions...")
    analyses = {}
    
    # Skills
    logger.info("Analyzing skills")
    skill_query_emb = embedding_gen.generate_embedding(jd_text)
    skill_matches = weaviate_mgr.semantic_search(
        skill_query_emb, 
        section_filter='skills',
        limit=10
    )
    analyses['skills'] = llm_analyzer.analyze_skills(
        resume_sections['raw_text'],
        jd_text,
        [m['content'] for m in skill_matches]
    )
    
    # Impact
    logger.info("Analyzing impact")
    analyses['impact'] = llm_analyzer.analyze_impact(
        resume_sections['raw_text'],
        jd_text
    )
    
    # Roles & Responsibilities
    logger.info("Analyzing roles & responsibilities")
    analyses['roles_responsibilities'] = llm_analyzer.analyze_roles_responsibilities(
        resume_sections['raw_text'],
        jd_text
    )
    
    # Achievements
    logger.info("Analyzing achievements")
    analyses['achievements'] = llm_analyzer.analyze_achievements(
        resume_sections['raw_text'],
        jd_text
    )
    
    # Experience
    logger.info("Analyzing experience")
    analyses['experience'] = llm_analyzer.analyze_experience(
        resume_sections['raw_text'],
        jd_text
    )
    
    # Personal Details
    logger.info("Analyzing personal details")
    analyses['personal_details'] = llm_analyzer.analyze_personal_details(
        resume_sections['raw_text'],
        jd_text
    )
    
    # Hobbies
    logger.info("Analyzing hobbies")
    analyses['hobbies'] = llm_analyzer.analyze_hobbies(
        resume_sections['raw_text'],
        jd_text
    )
    
    # 5. Calculate overall score
    logger.info("Calculating overall score")
    dimension_scores = {k: v['score'] for k, v in analyses.items()}
    overall_score = scoring_engine.calculate_overall_score(dimension_scores)
    summary = scoring_engine.generate_summary(overall_score, dimension_scores)
    
    # 6. Generate highlights
    logger.info("Generating highlights")
    highlight_gen = HighlightGenerator()
    full_result = {
        'overall_match_score': overall_score,
        'summary': summary,
        'sections': analyses
    }
    highlights = highlight_gen.generate_highlights(full_result)
    
    # 7. Return complete report
    logger.info("Returning complete report")
    return {
        **full_result,
        'highlight_instructions': highlights
    }
