import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

class PubMedParser:
    @staticmethod
    def parse_xml(xml_data: str) -> List[Dict]:
        if not xml_data:
            return []
        
        papers = []
        try:
            root = ET.fromstring(xml_data)
            for article in root.findall(".//PubmedArticle"):
                paper = {}
                # PMID
                pmid_node = article.find(".//PMID")
                paper["pmid"] = pmid_node.text if pmid_node is not None else ""
                
                # Title
                title_node = article.find(".//ArticleTitle")
                paper["title"] = "".join(title_node.itertext()) if title_node is not None else "No Title"
                
                # Abstract
                abstract_node = article.find(".//AbstractText")
                paper["abstract"] = "".join(abstract_node.itertext()) if abstract_node is not None else "No Abstract"
                
                # Journal
                journal_node = article.find(".//Journal/Title")
                paper["journal"] = journal_node.text if journal_node is not None else "Unknown Journal"
                
                # Year
                year_node = article.find(".//PubDate/Year")
                if year_node is None:
                    year_node = article.find(".//ArticleDate/Year")
                paper["year"] = year_node.text if year_node is not None else "Unknown"
                
                # Authors
                authors = []
                for author_node in article.findall(".//Author"):
                    last = author_node.find("LastName")
                    fore = author_node.find("ForeName")
                    if last is not None and fore is not None:
                        authors.append(f"{fore.text} {last.text}")
                paper["authors"] = ", ".join(authors) if authors else "Unknown"
                
                # DOI
                doi_node = article.find(".//ArticleId[@IdType='doi']")
                paper["doi"] = doi_node.text if doi_node is not None else ""
                
                paper["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/" if paper["pmid"] else ""
                
                papers.append(paper)
        except Exception as e:
            print(f"Error parsing XML: {e}")
            
        return papers
