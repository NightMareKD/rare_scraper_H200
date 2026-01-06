"""
Scraper Module for Clinical Case Sources
=========================================
Handles data ingestion from verified medical literature sources.
"""

import os
import time
import json
import hashlib
import logging
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


@dataclass
class ScrapedCase:
    """Represents a scraped case report."""
    case_id: str
    source: str
    url: str
    title: str
    content: str
    publication_date: Optional[str]
    journal: Optional[str]
    doi: Optional[str]
    scraped_at: str
    raw_data: Dict[str, Any]


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, requests_per_second: float = 0.5):
        self.min_interval = 1.0 / requests_per_second
        self.last_request = 0.0
    
    def wait(self) -> None:
        """Wait if necessary to respect rate limit."""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()


class BaseScraper:
    """Base class for source scrapers."""
    
    def __init__(self, config: Dict[str, Any], checkpoint_manager, data_root: Path = None):
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.data_root = Path(data_root) if data_root else Path("data")
        self.rate_limiter = RateLimiter(
            config.get("rate_limit", {}).get("requests_per_second", 0.5)
        )
        default_dir = self.data_root / "raw"
        self.output_dir = Path(config.get("output", {}).get("directory", str(default_dir)))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scrape(self) -> Generator[ScrapedCase, None, None]:
        """Override in subclass to implement scraping."""
        raise NotImplementedError
    
    def save_case(self, case: ScrapedCase) -> None:
        """Save scraped case to disk."""
        output_file = self.output_dir / f"{case.case_id}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "case_id": case.case_id,
                "source": case.source,
                "url": case.url,
                "title": case.title,
                "content": case.content,
                "publication_date": case.publication_date,
                "journal": case.journal,
                "doi": case.doi,
                "scraped_at": case.scraped_at,
                "raw_data": case.raw_data,
            }, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved case: {case.case_id}")


class PubMedScraper(BaseScraper):
    """
    Scraper for PubMed Central open access case reports.
    Uses NCBI E-utilities API.
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def __init__(self, config: Dict[str, Any], checkpoint_manager, data_root: Path = None):
        super().__init__(config, checkpoint_manager, data_root=data_root)
        self.api_key = os.environ.get("NCBI_API_KEY")
        self.output_dir = self.data_root / "raw" / "pubmed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scrape(self) -> Generator[ScrapedCase, None, None]:
        """Scrape PubMed case reports."""
        scraped_urls = self.checkpoint_manager.get_processed_ids("scraped_urls")
        
        search_config = self.config.get("search", {})
        query_terms = search_config.get("query_terms", ["case report[pt]"])
        
        for query in query_terms:
            logger.info(f"Searching PubMed: {query}")
            
            # Search for PMIDs
            pmids = self._search(query, max_results=1000)
            
            for pmid in pmids:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                
                if url in scraped_urls:
                    continue
                
                try:
                    case = self._fetch_case(pmid)
                    if case:
                        self.save_case(case)
                        self.checkpoint_manager.mark_processed("scraped_urls", url)
                        yield case
                        
                except Exception as e:
                    logger.error(f"Error fetching PMID {pmid}: {e}")
                    continue
    
    def _search(self, query: str, max_results: int = 1000) -> List[str]:
        """Search PubMed and return PMIDs."""
        self.rate_limiter.wait()
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "usehistory": "y",
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = requests.get(
                f"{self.BASE_URL}esearch.fcgi",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("esearchresult", {}).get("idlist", [])
            
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []
    
    def _fetch_case(self, pmid: str) -> Optional[ScrapedCase]:
        """Fetch full case details for a PMID."""
        self.rate_limiter.wait()
        
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = requests.get(
                f"{self.BASE_URL}efetch.fcgi",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            article = root.find(".//PubmedArticle")
            
            if article is None:
                return None
            
            # Extract fields
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            
            abstract_elem = article.find(".//AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""
            
            year_elem = article.find(".//PubDate/Year")
            year = year_elem.text if year_elem is not None else ""
            
            doi_elem = article.find(".//ArticleId[@IdType='doi']")
            doi = doi_elem.text if doi_elem is not None else None
            
            return ScrapedCase(
                case_id=f"PMID_{pmid}",
                source="pubmed",
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                title=title,
                content=abstract,
                publication_date=year,
                journal=journal,
                doi=doi,
                scraped_at=datetime.now().isoformat(),
                raw_data={"pmid": pmid, "xml": response.text[:5000]}
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch PMID {pmid}: {e}")
            return None


class ESCScraper(BaseScraper):
    """Scraper for European Society of Cardiology case gallery."""
    
    def __init__(self, config: Dict[str, Any], checkpoint_manager, data_root: Path = None):
        super().__init__(config, checkpoint_manager, data_root=data_root)
        self.output_dir = self.data_root / "raw" / "esc"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scrape(self) -> Generator[ScrapedCase, None, None]:
        """Scrape ESC cases."""
        # Placeholder - actual implementation would use web scraping
        logger.info("ESC scraper not yet implemented")
        return
        yield  # Make it a generator


class AHAScraper(BaseScraper):
    """Scraper for American Heart Association case reports."""
    
    def __init__(self, config: Dict[str, Any], checkpoint_manager, data_root: Path = None):
        super().__init__(config, checkpoint_manager, data_root=data_root)
        self.output_dir = self.data_root / "raw" / "aha"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scrape(self) -> Generator[ScrapedCase, None, None]:
        """Scrape AHA cases."""
        # Placeholder - actual implementation would use web scraping
        logger.info("AHA scraper not yet implemented")
        return
        yield


class JournalScraper(BaseScraper):
    """Scraper for open access journal case reports."""
    
    def __init__(self, config: Dict[str, Any], checkpoint_manager, data_root: Path = None):
        super().__init__(config, checkpoint_manager, data_root=data_root)
        self.output_dir = self.data_root / "raw" / "journals"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scrape(self) -> Generator[ScrapedCase, None, None]:
        """Scrape open access journals."""
        # Placeholder - actual implementation would use web scraping
        logger.info("Journal scraper not yet implemented")
        return
        yield


class ScraperFactory:
    """Factory for creating source-specific scrapers."""
    
    SCRAPERS = {
        "pubmed": PubMedScraper,
        "esc": ESCScraper,
        "aha": AHAScraper,
        "journals": JournalScraper,
    }
    
    @classmethod
    def create(
        cls,
        source_name: str,
        config: Dict[str, Any],
        checkpoint_manager,
        data_root: Path = None
    ) -> Optional[BaseScraper]:
        """Create scraper for given source."""
        scraper_class = cls.SCRAPERS.get(source_name)
        
        if scraper_class is None:
            logger.warning(f"Unknown source: {source_name}")
            return None
        
        return scraper_class(config, checkpoint_manager, data_root=data_root)


class IngestionManager:
    """
    Manages the complete ingestion pipeline.
    Coordinates scraping across all sources.
    """
    
    def __init__(self, sources_config: Dict[str, Any], checkpoint_manager, data_root: Path = None):
        self.sources_config = sources_config
        self.checkpoint_manager = checkpoint_manager
        self.data_root = Path(data_root) if data_root else Path("data")
        self.scrapers: Dict[str, BaseScraper] = {}
        
        self._initialize_scrapers()
    
    def _initialize_scrapers(self) -> None:
        """Initialize scrapers for all enabled sources."""
        sources = self.sources_config.get("sources", {})
        
        for source_name, source_config in sources.items():
            if source_config.get("enabled", False):
                scraper = ScraperFactory.create(
                    source_name,
                    source_config,
                    self.checkpoint_manager,
                    data_root=self.data_root
                )
                if scraper:
                    self.scrapers[source_name] = scraper
                    logger.info(f"Initialized scraper: {source_name}")
    
    def run_ingestion(self, max_cases_per_source: int = 1000) -> int:
        """
        Run ingestion across all sources.
        Returns total number of cases scraped.
        """
        total_scraped = 0
        
        for source_name, scraper in self.scrapers.items():
            logger.info(f"Starting ingestion for: {source_name}")
            
            source_count = 0
            try:
                for case in scraper.scrape():
                    source_count += 1
                    total_scraped += 1
                    
                    if source_count >= max_cases_per_source:
                        logger.info(f"Reached limit for {source_name}: {source_count}")
                        break
                        
            except Exception as e:
                logger.error(f"Ingestion failed for {source_name}: {e}")
                continue
            
            logger.info(f"Completed {source_name}: {source_count} cases")
        
        logger.info(f"Total ingestion complete: {total_scraped} cases")
        return total_scraped
