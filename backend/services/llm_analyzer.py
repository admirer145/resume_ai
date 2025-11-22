import openai
import json
from typing import Dict, List


class LLMAnalyzer:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.model = "gpt-4o-mini"
    
    def analyze_skills(self, resume_text: str, jd_text: str, 
                       semantic_matches: List[str]) -> Dict:
        """Analyze skills match"""
        prompt = f"""
        You are a resume analysis expert. Analyze the skills match between this resume and job description.

        RESUME:
        {resume_text}

        JOB DESCRIPTION:
        {jd_text}

        SEMANTIC MATCHES FOUND:
        {json.dumps(semantic_matches, indent=2)}

        Provide a JSON response with:
        {{
        "score": 0-100,
        "matched_skills": ["skill1", "skill2"],
        "missing_skills": ["skill3", "skill4"],
        "mentioned_but_not_listed": ["skill5"],
        "comment": "Brief explanation",
        "suggestions": ["suggestion1", "suggestion2"]
        }}

        Be specific and evidence-based. Reference exact phrases from the resume.
        """
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a resume analysis expert."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    
    def analyze_impact(self, resume_text: str, jd_text: str) -> Dict:
        """Analyze impact and contributions"""
        prompt = f"""
        Analyze the impact and contributions in this resume against the job requirements.

        RESUME:
        {resume_text}

        JOB DESCRIPTION:
        {jd_text}

        Look for:
        - Quantifiable achievements (numbers, percentages, metrics)
        - Business impact (revenue, cost savings, efficiency)
        - Scale indicators (team size, user base, data volume)
        - Leadership impact (mentoring, process improvements)

        Return JSON:
        {{
        "score": 0-100,
        "summary": "Overall impact assessment",
        "examples_found": [
            {{"achievement": "...", "impact": "...", "relevance": "high/medium/low"}}
        ],
        "missing_elements": ["What's missing"],
        "suggestions": ["How to improve"]
        }}
        """
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a resume analysis expert."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    
    def analyze_roles_responsibilities(self, resume_text: str, jd_text: str) -> Dict:
        """Analyze roles and responsibilities fit"""
        prompt = f"""
        Extract key roles and responsibilities from the JD and match them to the resume.

        RESUME:
        {resume_text}

        JOB DESCRIPTION:
        {jd_text}

        Return JSON:
        {{
        "score": 0-100,
        "matched": [
            {{"jd_requirement": "...", "resume_evidence": "...", "match_quality": "strong/moderate/weak"}}
        ],
        "missing": ["Responsibility not covered"],
        "suggestions": ["How to address gaps"]
        }}
        """
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a resume analysis expert."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    
    def analyze_achievements(self, resume_text: str, jd_text: str) -> Dict:
        """Analyze achievements relevance"""
        prompt = f"""
        Evaluate the achievements in the resume for relevance to this job.

        RESUME:
        {resume_text}

        JOB DESCRIPTION:
        {jd_text}

        Assess:
        - Relevance to the target role
        - Recency (recent achievements more valuable)
        - Quantifiability and impact
        - Alignment with JD priorities

        Return JSON:
        {{
        "score": 0-100,
        "relevant": [
            {{"achievement": "...", "why_relevant": "...", "strength": "high/medium/low"}}
        ],
        "irrelevant": [
            {{"achievement": "...", "why_irrelevant": "..."}}
        ],
        "suggestions": ["What to add/emphasize/remove"]
        }}
        """
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a resume analysis expert."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    
    def analyze_experience(self, resume_text: str, jd_text: str) -> Dict:
        """Analyze experience fit"""
        prompt = f"""
        Analyze the candidate's experience against job requirements.

        RESUME:
        {resume_text}

        JOB DESCRIPTION:
        {jd_text}

        Evaluate:
        - Total years of experience
        - Relevant domain experience
        - Seniority level match
        - Technology/tool experience
        - Industry experience

        Return JSON:
        {{
        "score": 0-100,
        "resume_experience_summary": "X years total, Y years in domain Z",
        "jd_experience_requirement": "What JD asks for",
        "gap_analysis": "Where experience falls short or exceeds",
        "suggestions": ["How to better present experience"]
        }}
        """
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a resume analysis expert."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    
    def analyze_personal_details(self, resume_text: str, jd_text: str) -> Dict:
        """Analyze personal details completeness"""
        prompt = f"""
        Evaluate the personal/contact details section.

        RESUME:
        {resume_text}

        JOB DESCRIPTION:
        {jd_text}

        Check for:
        - Email, phone (essential)
        - LinkedIn profile (highly recommended)
        - GitHub/portfolio (for tech roles)
        - Location (if relevant)
        - Irrelevant info (DOB, gender, photo, nationality - unless required)

        Return JSON:
        {{
            "score": 0-100,
            "missing": ["LinkedIn", "GitHub"],
            "irrelevant": ["Date of birth"],
            "suggestions": ["Add professional LinkedIn URL", "Remove DOB"]
        }}
        """
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a resume analysis expert."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    
    def analyze_hobbies(self, resume_text: str, jd_text: str) -> Dict:
        """Analyze hobbies and extracurriculars"""
        prompt = f"""
        Evaluate hobbies and extracurricular activities for job relevance.

        RESUME:
        {resume_text}

        JOB DESCRIPTION:
        {jd_text}

        Consider:
        - Soft skills demonstrated (leadership, teamwork, creativity)
        - Cultural fit indicators
        - Relevance to role (e.g., tech blogging for developer role)
        - Generic vs. distinctive activities

        Return JSON:
        {{
        "score": 0-100,
        "relevant": [
            {{"activity": "...", "why_relevant": "..."}}
        ],
        "irrelevant": [
            {{"activity": "...", "why_irrelevant": "..."}}
        ],
        "suggestions": ["Keep/remove/add specific activities"]
        }}
        """
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a resume analysis expert."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
