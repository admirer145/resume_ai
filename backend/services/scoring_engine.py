from typing import Dict

class ScoringEngine:
    def __init__(self):
        # Weights for different dimensions (must sum to 1.0)
        self.weights = {
            'skills': 0.25,           # 25% - Most critical
            'experience': 0.20,        # 20% - Very important
            'roles_responsibilities': 0.20,  # 20% - Core fit
            'impact': 0.15,           # 15% - Differentiator
            'achievements': 0.10,      # 10% - Nice to have
            'personal_details': 0.05,  # 5% - Basic requirement
            'hobbies': 0.05           # 5% - Cultural fit
        }
    
    def calculate_overall_score(self, dimension_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        overall = 0.0
        
        for dimension, weight in self.weights.items():
            score = dimension_scores.get(dimension, 0)
            overall += score * weight
        
        return round(overall, 1)
    
    def get_match_category(self, score: float) -> str:
        """Categorize match strength"""
        if score >= 85:
            return "Excellent Match"
        elif score >= 70:
            return "Strong Match"
        elif score >= 55:
            return "Moderate Match"
        elif score >= 40:
            return "Weak Match"
        else:
            return "Poor Match"
    
    def generate_summary(self, overall_score: float, 
                        dimension_scores: Dict[str, float]) -> str:
        """Generate executive summary"""
        category = self.get_match_category(overall_score)
        
        # Find strengths and weaknesses
        strengths = [k for k, v in dimension_scores.items() if v >= 75]
        weaknesses = [k for k, v in dimension_scores.items() if v < 50]
        
        summary = f"{category} ({overall_score}%). "
        
        if strengths:
            summary += f"Strong in {', '.join(strengths)}. "
        
        if weaknesses:
            summary += f"Needs improvement in {', '.join(weaknesses)}."
        
        return summary
