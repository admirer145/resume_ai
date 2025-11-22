import fitz  # PyMuPDF
from typing import Dict, List


class PDFExtractor:
    def __init__(self):
        self.section_keywords = {
            'skills': ['skills', 'technical skills', 'competencies'],
            'experience': ['experience', 'work history', 'employment'],
            'education': ['education', 'academic', 'qualifications'],
            'achievements': ['achievements', 'accomplishments', 'awards'],
            'projects': ['projects', 'portfolio'],
            'personal': ['contact', 'personal details', 'profile'],
            'hobbies': ['hobbies', 'interests', 'activities']
        }
    
    def extract_text(self, pdf_path: str) -> Dict[str, str]:
        """Extract text with section detection"""
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page in doc:
            full_text += page.get_text()
        
        return self._parse_sections(full_text)
    
    def _parse_sections(self, text: str) -> Dict[str, str]:
        """Detect and separate resume sections"""
        sections = {
            'raw_text': text,
            'skills': '',
            'experience': '',
            'education': '',
            'achievements': '',
            'projects': '',
            'personal': '',
            'hobbies': ''
        }
        
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect section headers
            for section_name, keywords in self.section_keywords.items():
                if any(keyword in line_lower for keyword in keywords):
                    current_section = section_name
                    break
            
            # Append to current section
            if current_section:
                sections[current_section] += line + '\n'
        
        return sections
