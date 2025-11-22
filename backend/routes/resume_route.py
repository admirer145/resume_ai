from fastapi import APIRouter, UploadFile, File, Form, Body
from typing import List, Dict

from services.resume_service import analyze_resume_main
from utils.logger import logger


router = APIRouter()


@router.post("/analyze")
async def analyze_resume(
        resume: UploadFile = File(...),
        jd_text: str = Form(...)
    ):
    try:
        resp = await analyze_resume_main(resume, jd_text)
    except Exception as e:
        logger.error(f"Error analyzing resume: {str(e)}")
        return {"error": str(e)}
    return resp


@router.post("/batch-analyze")
async def batch_analyze(
    resumes: List[UploadFile] = File(...),
    jd_text: str = Form(...)
):
    """Analyze multiple resumes against one JD"""
    results = []
    
    for resume in resumes:
        result = await analyze_resume_main(resume, jd_text)
        results.append({
            'filename': resume.filename,
            'score': result['overall_match_score'],
            'summary': result['summary'],
            'full_analysis': result
        })
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        'total_resumes': len(results),
        'top_candidates': results[:5],
        'all_results': results
    }


@router.post("/compare")
async def compare_resumes(
    resume1: UploadFile = File(...),
    resume2: UploadFile = File(...),
    jd_text: str = Form(...)
):
    """Compare two resumes side-by-side"""
    result1 = await analyze_resume_main(resume1, jd_text)
    result2 = await analyze_resume_main(resume2, jd_text)
    
    comparison = {
        'resume1': {
            'name': resume1.filename,
            'overall_score': result1['overall_match_score'],
            'dimension_scores': {k: v['score'] for k, v in result1['sections'].items()}
        },
        'resume2': {
            'name': resume2.filename,
            'overall_score': result2['overall_match_score'],
            'dimension_scores': {k: v['score'] for k, v in result2['sections'].items()}
        },
        'winner': resume1.filename if result1['overall_match_score'] > result2['overall_match_score'] else resume2.filename,
        'dimension_comparison': {}
    }
    
    # Compare each dimension
    for dim in result1['sections'].keys():
        score1 = result1['sections'][dim]['score']
        score2 = result2['sections'][dim]['score']
        comparison['dimension_comparison'][dim] = {
            'resume1_score': score1,
            'resume2_score': score2,
            'difference': score1 - score2,
            'winner': resume1.filename if score1 > score2 else resume2.filename
        }
    
    return comparison


@router.post("/suggest-improvements")
async def suggest_improvements(
    analysis_result: Dict = Body(...)
):
    """Generate detailed improvement suggestions"""
    
    prompt = f"""
    Based on the resume analysis, generate a prioritized action plan for improvement.

    Analysis:
    {json.dumps(analysis_result, indent=2)}

    Provide:
    1. Top 3 high-impact changes (quick wins)
    2. Top 3 medium-term improvements
    3. Sample rewrites for weak sections
    4. Skills to add based on career trajectory

    Return as JSON with prioritized, actionable steps.
    """
    
    response = openai.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a professional resume coach."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)


