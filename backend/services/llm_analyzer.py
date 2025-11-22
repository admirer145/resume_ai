import json
from typing import Dict, List
import config


class LLMAnalyzer:
    def __init__(self, provider: str = None, api_key: str = None, base_url: str = None):
        """
        Initialize LLM analyzer with support for multiple providers.
        
        Args:
            provider: "openai" or "ollama" (defaults to config.MODEL_PROVIDER)
            api_key: API key for OpenAI (only needed if provider is "openai")
            base_url: Base URL for Ollama (only needed if provider is "ollama")
        """
        self.provider = provider or config.MODEL_PROVIDER
        self.model = config.get_llm_model()
        
        if self.provider == "openai":
            import openai
            openai.api_key = api_key or config.OPENAI_API_KEY
            self.client = openai
        else:  # ollama
            import ollama
            self.base_url = base_url or config.OLLAMA_BASE_URL
            self.client = ollama.Client(host=self.base_url)
    
    def _validate_response(self, response: Dict, expected_keys: List[str]) -> Dict:
        """
        Validate and fill missing keys in LLM response.
        Ensures all expected keys exist with sensible defaults.
        """
        # Always ensure 'score' exists
        if 'score' not in response:
            response['score'] = 50  # Default neutral score
        
        # Fill in missing keys with appropriate defaults
        defaults = {
            'matched_skills': [],
            'missing_skills': [],
            'mentioned_but_not_listed': [],
            'comment': 'Analysis completed',
            'suggestions': [],
            'summary': 'Analysis completed',
            'examples_found': [],
            'missing_elements': [],
            'matched': [],
            'missing': [],
            'relevant': [],
            'irrelevant': [],
            'resume_experience_summary': 'Not specified',
            'jd_experience_requirement': 'Not specified',
            'gap_analysis': 'Not specified'
        }
        
        for key in expected_keys:
            if key not in response:
                response[key] = defaults.get(key, [])
        
        return response
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict:
        """
        Call LLM with provider-specific logic.
        Returns parsed JSON response.
        """
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            return json.loads(response.choices[0].message.content)
        else:  # ollama
            # Combine system and user prompts for Ollama
            full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nIMPORTANT: Respond ONLY with valid JSON. No other text."
            
            response = self.client.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 1024  # Limit response length
                }
            )
            
            # Extract and parse JSON from response
            response_text = response['response'].strip()
            
            # Try to find JSON in the response
            try:
                # First, try direct parsing
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                
                # Try parsing again
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    # Return a default error response
                    return {
                        "score": 50,
                        "error": "Failed to parse LLM response",
                        "raw_response": response_text[:200]
                    }

    
    def analyze_skills(self, resume_text: str, jd_text: str, 
                       semantic_matches: List[str]) -> Dict:
        """Analyze skills match"""
        system_prompt = "You are a resume analysis expert."
        user_prompt = f"""
        Analyze the skills match between this resume and job description.

        RESUME:
        {resume_text[:2000]}

        JOB DESCRIPTION:
        {jd_text[:1000]}

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

        Be specific and evidence-based.
        """
        response = self._call_llm(system_prompt, user_prompt)
        return self._validate_response(response, [
            'score', 'matched_skills', 'missing_skills', 
            'mentioned_but_not_listed', 'comment', 'suggestions'
        ])

    
    def analyze_impact(self, resume_text: str, jd_text: str) -> Dict:
        """Analyze impact and contributions"""
        system_prompt = "You are a resume analysis expert."
        user_prompt = f"""
        Analyze impact and contributions in this resume against job requirements.

        RESUME:
        {resume_text[:2000]}

        JOB DESCRIPTION:
        {jd_text[:1000]}

        Look for:
        - Quantifiable achievements (numbers, percentages, metrics)
        - Business impact (revenue, cost savings, efficiency)
        - Scale indicators (team size, user base, data volume)
        - Leadership impact (mentoring, process improvements)

        Return JSON:
        {{
        "score": 0-100,
        "summary": "Overall impact assessment",
        "examples_found": [{{"achievement": "...", "impact": "...", "relevance": "high/medium/low"}}],
        "missing_elements": ["What's missing"],
        "suggestions": ["How to improve"]
        }}
        """
        response = self._call_llm(system_prompt, user_prompt)
        return self._validate_response(response, [
            'score', 'summary', 'examples_found', 'missing_elements', 'suggestions'
        ])

    
    def analyze_roles_responsibilities(self, resume_text: str, jd_text: str) -> Dict:
        """Analyze roles and responsibilities fit"""
        system_prompt = "You are a resume analysis expert."
        user_prompt = f"""
        Extract key roles and responsibilities from the JD and match them to the resume.

        RESUME:
        {resume_text[:2000]}

        JOB DESCRIPTION:
        {jd_text[:1000]}

        Return JSON:
        {{
        "score": 0-100,
        "matched": [{{"jd_requirement": "...", "resume_evidence": "...", "match_quality": "strong/moderate/weak"}}],
        "missing": ["Responsibility not covered"],
        "suggestions": ["How to address gaps"]
}}
"""
        response = self._call_llm(system_prompt, user_prompt)
        return self._validate_response(response, ['score', 'matched', 'missing', 'suggestions'])
    
    def analyze_achievements(self, resume_text: str, jd_text: str) -> Dict:
        """Analyze achievements relevance"""
        system_prompt = "You are a resume analysis expert."
        user_prompt = f"""
        Evaluate the achievements in the resume for relevance to this job.

        RESUME:
        {resume_text[:2000]}

        JOB DESCRIPTION:
        {jd_text[:1000]}

        Assess:
        - Relevance to the target role
        - Recency (recent achievements more valuable)
        - Quantifiability and impact
        - Alignment with JD priorities

        Return JSON:
        {{
        "score": 0-100,
        "relevant": [{{"achievement": "...", "why_relevant": "...", "strength": "high/medium/low"}}],
        "irrelevant": [{{"achievement": "...", "why_irrelevant": "..."}}],
        "suggestions": ["What to add/emphasize/remove"]
        }}
        """
        response = self._call_llm(system_prompt, user_prompt)
        return self._validate_response(response, ['score', 'relevant', 'irrelevant', 'suggestions'])
    
    def analyze_experience(self, resume_text: str, jd_text: str) -> Dict:
        """Analyze experience fit"""
        system_prompt = "You are a resume analysis expert."
        user_prompt = f"""
        Analyze the candidate's experience against job requirements.

        RESUME:
        {resume_text[:2000]}

        JOB DESCRIPTION:
        {jd_text[:1000]}

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
        response = self._call_llm(system_prompt, user_prompt)
        return self._validate_response(response, [
            'score', 'resume_experience_summary', 'jd_experience_requirement', 
            'gap_analysis', 'suggestions'
        ])
    
    def analyze_personal_details(self, resume_text: str, jd_text: str) -> Dict:
        """Analyze personal details completeness"""
        system_prompt = "You are a resume analysis expert."
        user_prompt = f"""
        Evaluate the personal/contact details section.

        RESUME:
        {resume_text[:2000]}

        JOB DESCRIPTION:
        {jd_text[:1000]}

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
        response = self._call_llm(system_prompt, user_prompt)
        return self._validate_response(response, ['score', 'missing', 'irrelevant', 'suggestions'])
    
    def analyze_hobbies(self, resume_text: str, jd_text: str) -> Dict:
        """Analyze hobbies and extracurriculars"""
        system_prompt = "You are a resume analysis expert."
        user_prompt = f"""
        Evaluate hobbies and extracurricular activities for job relevance.

        RESUME:
        {resume_text[:2000]}

        JOB DESCRIPTION:
        {jd_text[:1000]}

        Consider:
        - Soft skills demonstrated (leadership, teamwork, creativity)
        - Cultural fit indicators
        - Relevance to role (e.g., tech blogging for developer role)
        - Generic vs. distinctive activities

        Return JSON:
        {{
            "score": 0-100,
            "relevant": [{{"activity": "...", "why_relevant": "..."}}],
            "irrelevant": [{{"activity": "...", "why_irrelevant": "..."}}],
            "suggestions": ["Keep/remove/add specific activities"]
        }}
        """
        response = self._call_llm(system_prompt, user_prompt)
        return self._validate_response(response, ['score', 'relevant', 'irrelevant', 'suggestions'])
