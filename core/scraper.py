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
from typing import Dict, Any, List, Optional, Generator, Callable
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

    def wait_or_stop(self, should_stop: Callable[[], bool]) -> bool:
        """Wait if needed unless a stop is requested. Returns False if stopped."""
        if should_stop():
            return False
        elapsed = time.time() - self.last_request
        remaining = self.min_interval - elapsed
        if remaining > 0:
            end_time = time.time() + remaining
            while time.time() < end_time:
                if should_stop():
                    return False
                time.sleep(min(0.1, end_time - time.time()))
        self.last_request = time.time()
        return True


class BaseScraper:
    """Base class for source scrapers."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_manager,
        data_root: Path = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.data_root = Path(data_root) if data_root else Path("data")
        self.should_stop = should_stop or (lambda: False)
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
    
    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_manager,
        data_root: Path = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        super().__init__(config, checkpoint_manager, data_root=data_root, should_stop=should_stop)
        self.api_key = os.environ.get("NCBI_API_KEY")
        self.output_dir = self.data_root / "raw" / "pubmed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scrape(self) -> Generator[ScrapedCase, None, None]:
        """Scrape PubMed case reports."""
        scraped_urls = self.checkpoint_manager.get_processed_ids("scraped_urls")
        
        search_config = self.config.get("search", {})
        query_terms = search_config.get("query_terms", ["case report[pt]"])
        
        for query in query_terms:
            if self.should_stop():
                logger.info("Stop requested; ending PubMed scraping")
                return
            logger.info(f"Searching PubMed: {query}")
            
            # Search for PMIDs
            pmids = self._search(query, max_results=1000)
            
            for pmid in pmids:
                if self.should_stop():
                    logger.info("Stop requested; ending PubMed scraping")
                    return
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
        if not self.rate_limiter.wait_or_stop(self.should_stop):
            return []
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "usehistory": "y",
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        return self._request_json_with_backoff("esearch.fcgi", params).get("esearchresult", {}).get("idlist", [])
    
    def _fetch_case(self, pmid: str) -> Optional[ScrapedCase]:
        """Fetch full case details for a PMID."""
        if not self.rate_limiter.wait_or_stop(self.should_stop):
            return None
        
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = self._request_with_backoff("efetch.fcgi", params)
            if response is None:
                return None

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

    def _request_json_with_backoff(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        response = self._request_with_backoff(endpoint, params)
        if response is None:
            return {}
        try:
            return response.json()
        except Exception:
            return {}

    def _request_with_backoff(self, endpoint: str, params: Dict[str, Any]) -> Optional[requests.Response]:
        """NCBI requests with 429 backoff; returns None if stop requested."""
        url = f"{self.BASE_URL}{endpoint}"
        backoff_seconds = 1.0
        max_retries = 5

        for attempt in range(max_retries):
            if self.should_stop():
                return None
            try:
                resp = requests.get(url, params=params, timeout=30)

                # Handle rate limiting
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait_s = float(retry_after) if retry_after else backoff_seconds
                    logger.warning(f"NCBI 429 rate limit; backing off {wait_s:.1f}s")
                    end_time = time.time() + wait_s
                    while time.time() < end_time:
                        if self.should_stop():
                            return None
                        time.sleep(min(0.2, end_time - time.time()))
                    backoff_seconds = min(backoff_seconds * 2, 30.0)
                    continue

                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                # Non-429 errors: retry a couple times, but stop if requested
                if self.should_stop():
                    return None
                if attempt >= max_retries - 1:
                    raise
                logger.warning(f"Request failed ({attempt + 1}/{max_retries}): {e}")
                end_time = time.time() + backoff_seconds
                while time.time() < end_time:
                    if self.should_stop():
                        return None
                    time.sleep(min(0.2, end_time - time.time()))
                backoff_seconds = min(backoff_seconds * 2, 30.0)

        return None


class ESCScraper(BaseScraper):
    """Scraper for European Society of Cardiology case gallery."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_manager,
        data_root: Path = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        super().__init__(config, checkpoint_manager, data_root=data_root, should_stop=should_stop)
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
    
    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_manager,
        data_root: Path = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        super().__init__(config, checkpoint_manager, data_root=data_root, should_stop=should_stop)
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
    
    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_manager,
        data_root: Path = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        super().__init__(config, checkpoint_manager, data_root=data_root, should_stop=should_stop)
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
        data_root: Path = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Optional[BaseScraper]:
        """Create scraper for given source."""
        scraper_class = cls.SCRAPERS.get(source_name)
        
        if scraper_class is None:
            logger.warning(f"Unknown source: {source_name}")
            return None
        
        return scraper_class(config, checkpoint_manager, data_root=data_root, should_stop=should_stop)


class IngestionManager:
    """
    Manages the complete ingestion pipeline.
    Coordinates scraping across all sources.
    """
    
    def __init__(
        self,
        sources_config: Dict[str, Any],
        checkpoint_manager,
        data_root: Path = None,
        should_stop: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ):
        self.sources_config = sources_config
        self.checkpoint_manager = checkpoint_manager
        self.data_root = Path(data_root) if data_root else Path("data")
        self.should_stop = should_stop or (lambda: False)
        self.progress_callback = progress_callback  # callback(source_name, current, total)
        self.scrapers: Dict[str, BaseScraper] = {}

        # Optional UI-tunable limit
        self.max_cases_per_source: int = 10000
        
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
                    ,
                    should_stop=self.should_stop
                )
                if scraper:
                    self.scrapers[source_name] = scraper
                    logger.info(f"Initialized scraper: {source_name}")
    
    def run_ingestion(self, max_cases_per_source: int = None) -> int:
        """
        Run ingestion across all sources.
        Returns total number of cases scraped.
        """
        import time
        
        if max_cases_per_source is None:
            max_cases_per_source = self.max_cases_per_source
        total_scraped = 0
        
        for source_name, scraper in self.scrapers.items():
            if self.should_stop():
                logger.info("Stop requested; ending ingestion")
                break
            logger.info(f"Starting ingestion for: {source_name}")
            
            source_count = 0
            source_start_time = time.time()
            last_update_time = source_start_time
            
            try:
                for case in scraper.scrape():
                    if self.should_stop():
                        logger.info("Stop requested; ending ingestion")
                        break
                    source_count += 1
                    total_scraped += 1
                    
                    # Report progress every second or every 10 cases
                    current_time = time.time()
                    if (self.progress_callback and 
                        (current_time - last_update_time >= 1.0 or source_count % 10 == 0)):
                        elapsed = current_time - source_start_time
                        rate = source_count / elapsed if elapsed > 0 else 0
                        eta = (max_cases_per_source - source_count) / rate if rate > 0 else 0
                        self.progress_callback(source_name, source_count, max_cases_per_source, rate, eta)
                        last_update_time = current_time
                    
                    if source_count >= max_cases_per_source:
                        logger.info(f"Reached limit for {source_name}: {source_count}")
                        break
                        
            except Exception as e:
                logger.error(f"Ingestion failed for {source_name}: {e}")
                continue
            
            logger.info(f"Completed {source_name}: {source_count} cases")
        
        logger.info(f"Total ingestion complete: {total_scraped} cases")
        return total_scraped
