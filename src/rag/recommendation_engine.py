from typing import List

class RecommendationEngine:
    @staticmethod
    def generate_recommendations(mutation_info: str) -> List[str]:
        recs = [
            "PCR Confirmation Recommended to validate AI findings.",
            "Veterinary Examination Recommended.",
            "Requires Laboratory Validation."
        ]
        
        if "HA" in mutation_info.upper() or "HEMAGGLUTININ" in mutation_info.upper():
            recs.append("Assess receptor-binding specificity for HA mutations.")
            
        if "NA" in mutation_info.upper() or "NEURAMINIDASE" in mutation_info.upper():
            recs.append("Monitor for potential antiviral resistance mutations (e.g., oseltamivir resistance).")
            
        return recs
