import os
from openai import OpenAI
from typing import List, Dict

class LLMSummarizer:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        self.use_mock = not api_key
        if not self.use_mock:
            self.client = OpenAI(api_key=api_key)

    def summarize(self, mutation_context: str, top_papers: List[Dict]) -> str:
        if self.use_mock:
            return f"The detected mutation {mutation_context} is reported in multiple H5N1 studies and has been associated with increased viral pathogenicity. (Note: This is a simulated response due to missing OpenAI API Key)."
            
        context_texts = []
        for p in top_papers:
            context_texts.append(f"Title: {p.get('title')}\nAbstract: {p.get('abstract')}\n")
            
        context_str = "\n".join(context_texts)
        
        prompt = f"""
You are a Bioinformatics Expert. 
A sequence was predicted as infected with the following mutation context:
{mutation_context}

Based on the following retrieved PubMed abstracts:
{context_str}

Provide a comprehensive scientific summary. Include:
1. Short Summary
2. Biological significance
3. Relation with the detected mutation
4. Disease relevance (H5/H7 Avian Influenza)
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a scientific AI research assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating summary: {e}"
