from typing import Dict


class HighlightGenerator:
    def generate_highlights(self, analysis_result: Dict) -> Dict:
        """Generate highlighting instructions"""
        highlights = {
            'green': [],
            'yellow': [],
            'red': []
        }
        
        # Green: Strong matches
        highlights['green'].extend([
            f"Skill: {skill}" for skill in analysis_result['sections']['skills']['matched_skills']
        ])
        highlights['green'].extend([
            f"Achievement: {ach['achievement']}" 
            for ach in analysis_result['sections']['achievements']['relevant']
            if ach.get('strength') == 'high'
        ])
        
        # Yellow: Improvements needed
        highlights['yellow'].extend([
            f"Add to Skills section: {skill}" 
            for skill in analysis_result['sections']['skills']['mentioned_but_not_listed']
        ])
        highlights['yellow'].extend([
            f"Quantify: {ex['achievement']}" 
            for ex in analysis_result['sections']['impact']['examples_found']
            if 'quantify' in ex.get('suggestion', '').lower()
        ])
        
        # Red: Remove
        highlights['red'].extend([
            f"Remove irrelevant: {item}" 
            for item in analysis_result['sections']['hobbies']['irrelevant']
        ])
        highlights['red'].extend([
            f"Remove: {detail}" 
            for detail in analysis_result['sections']['personal_details']['irrelevant']
        ])
        
        return highlights
