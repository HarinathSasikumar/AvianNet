from .pubmed_search import PubMedSearcher
from .paper_parser import PubMedParser
from .embedding_engine import EmbeddingEngine
from .vector_store import VectorStore
from typing import List, Dict

class Retriever:
    def __init__(self):
        self.searcher = PubMedSearcher()
        self.parser = PubMedParser()
        self.embedding_engine = EmbeddingEngine()
        
    def run_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        # 1. Search PubMed
        pmids = self.searcher.search_query(query, max_results=20)
        
        # 2. Fetch and Parse XML
        xml_data = self.searcher.fetch_papers_xml(pmids)
        papers = self.parser.parse_xml(xml_data)
        
        if not papers:
            return []
            
        # 3. Create fresh vector store for this session
        vstore = VectorStore()
        
        # 4. Embed abstracts
        abstracts = [p.get("abstract", "") for p in papers]
        doc_embeddings = self.embedding_engine.encode(abstracts)
        
        if len(doc_embeddings) == 0:
            return papers[:top_k]
            
        # 5. Populate vector store
        vstore.add_papers(doc_embeddings, papers)
        
        # 6. Embed query and search
        query_emb = self.embedding_engine.encode([query])[0]
        results = vstore.search(query_emb, top_k=top_k)
        
        return results
