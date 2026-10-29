import requests
import xml.etree.ElementTree as ET
import urllib.parse
from typing import List, Dict

class PubMedSearcher:
    def __init__(self, email="researcher@example.com"):
        self.base_url_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.base_url_fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        self.email = email

    def search_query(self, query: str, max_results: int = 15) -> List[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "email": self.email
        }
        response = requests.get(self.base_url_search, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get("esearchresult", {}).get("idlist", [])
        return []

    def fetch_papers_xml(self, pmids: List[str]) -> str:
        if not pmids:
            return ""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "email": self.email
        }
        response = requests.get(self.base_url_fetch, params=params)
        if response.status_code == 200:
            return response.text
        return ""
