from typing import Dict


class HighlightGenerator:
    def generate_highlights(self, analysis_result: Dict) -> Dict:
        """Generate highlighting instructions with defensive key access"""
        highlights = {
            'green': [],
            'yellow': [],
            'red': []
        }
        
        # Safely access nested dictionaries with defaults
        sections = analysis_result.get('sections', {})
        
        # Green: Strong matches
        skills = sections.get('skills', {})
        highlights['green'].extend([
            f"Skill: {skill}" for skill in skills.get('matched_skills', [])
        ])
        
        achievements = sections.get('achievements', {})
        for ach in achievements.get('relevant', []):
            if isinstance(ach, dict) and ach.get('strength') == 'high':
                highlights['green'].append(f"Achievement: {ach.get('achievement', 'N/A')}")
        
        # Yellow: Improvements needed
        highlights['yellow'].extend([
            f"Add to Skills section: {skill}" 
            for skill in skills.get('mentioned_but_not_listed', [])
        ])
        
        impact = sections.get('impact', {})
        for ex in impact.get('examples_found', []):
            if isinstance(ex, dict):
                suggestion = ex.get('suggestion', '')
                if suggestion and 'quantify' in suggestion.lower():
                    highlights['yellow'].append(f"Quantify: {ex.get('achievement', 'N/A')}")
        
        # Red: Remove
        hobbies = sections.get('hobbies', {})
        for item in hobbies.get('irrelevant', []):
            if isinstance(item, dict):
                highlights['red'].append(f"Remove irrelevant: {item.get('activity', 'N/A')}")
            elif isinstance(item, str):
                highlights['red'].append(f"Remove irrelevant: {item}")
        
        personal = sections.get('personal_details', {})
        for detail in personal.get('irrelevant', []):
            if isinstance(detail, str):
                highlights['red'].append(f"Remove: {detail}")
        
        return highlights

