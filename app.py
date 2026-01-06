"""
Medical Case Scraper & Similarity Detection System
===================================================
Orchestrator for Hugging Face Spaces deployment.

This module is the ENTRY POINT only - it contains NO business logic.
All processing is delegated to specialized modules.

Author: Clinical AI Team
License: MIT
"""

import os
import sys
import time
import logging
import yaml
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum

def get_storage_root() -> Path:
    """Return persistent storage root (prefers HF Spaces /data when available)."""
    env_root = os.environ.get("SCRAPER_STORAGE_ROOT")
    if env_root:
        return Path(env_root)

    # Hugging Face Spaces persistent storage mount (when enabled)
    if Path("/data").exists():
        return Path("/data") / "scraper"

    # Fallback: project directory
    return Path(".")


STORAGE_ROOT = get_storage_root()
DATA_ROOT = STORAGE_ROOT / "data"
LOG_ROOT = STORAGE_ROOT / "logs"
CHECKPOINT_ROOT = STORAGE_ROOT / "checkpoints"

# Ensure directories exist before configuring logging
LOG_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(LOG_ROOT / 'ingestion.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import core modules
try:
    from core import (
        SystemState,
        StateManager,
        RarityClassifier,
        EmbeddingManager,
        FAISSIndexManager,
        SimilarityEngine,
        SimilarityResult,
        FlagDecision,
    )
    from core.scraper import IngestionManager, ScrapedCase
    from core.normalizer import NormalizationPipeline, NormalizedCase
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Core modules not fully available: {e}")
    CORE_MODULES_AVAILABLE = False


class ExecutionMode(Enum):
    """Execution modes based on available hardware."""
    CPU_ONLY = "cpu"
    GPU_AVAILABLE = "gpu"
    GPU_LIMITED = "gpu_limited"


class PipelineStage(Enum):
    """Pipeline execution stages."""
    SCRAPING = "scraping"
    NORMALIZATION = "normalization"
    DISTILLATION = "distillation"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    SERVING = "serving"
    IDLE = "idle"


class ConfigLoader:
    """Centralized configuration management."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, Any] = {}
    
    def load(self, config_name: str) -> Dict[str, Any]:
        """Load a YAML configuration file with caching."""
        if config_name in self._cache:
            return self._cache[config_name]
        
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Missing config: {config_name}.yaml")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self._cache[config_name] = config
        logger.info(f"Loaded configuration: {config_name}")
        return config
    
    def reload_all(self) -> None:
        """Clear cache and reload all configurations."""
        self._cache.clear()
        for config_file in self.config_dir.glob("*.yaml"):
            self.load(config_file.stem)


class HardwareDetector:
    """Detect available hardware and set execution mode."""
    
    @staticmethod
    def detect_gpu() -> Tuple[bool, Optional[str]]:
        """Detect if GPU is available and return device info."""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU detected: {device_name} ({gpu_memory:.1f}GB)")
                return True, device_name
        except ImportError:
            logger.warning("PyTorch not installed, GPU detection skipped")
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
        
        return False, None
    
    @staticmethod
    def get_execution_mode(config: Dict[str, Any]) -> ExecutionMode:
        """Determine execution mode based on hardware and policy."""
        gpu_available, _ = HardwareDetector.detect_gpu()
        
        if not gpu_available:
            return ExecutionMode.CPU_ONLY
        
        # Check GPU quota
        gpu_policy = config.get('gpu_policy', {})
        max_minutes = gpu_policy.get('max_gpu_minutes_per_day', 60)
        
        usage_file = Path("logs/gpu_usage.log")
        if usage_file.exists():
            today = datetime.now().date()
            daily_usage = 0.0
            
            with open(usage_file, 'r') as f:
                for line in f:
                    try:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            date_str, minutes = parts[0], float(parts[1])
                            if datetime.fromisoformat(date_str).date() == today:
                                daily_usage += minutes
                    except (ValueError, IndexError):
                        continue
            
            if daily_usage >= max_minutes:
                logger.warning(f"GPU quota exhausted: {daily_usage:.1f}/{max_minutes} minutes")
                return ExecutionMode.GPU_LIMITED
        
        return ExecutionMode.GPU_AVAILABLE


class CheckpointManager:
    """Manage pipeline checkpoints for resumability."""
    
    def __init__(self, checkpoint_dir: str = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_ROOT
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_processed_ids(self, checkpoint_name: str) -> set:
        """Get set of already processed IDs."""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.txt"
        if not checkpoint_file.exists():
            return set()
        
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    
    def mark_processed(self, checkpoint_name: str, item_id: str) -> None:
        """Mark an item as processed."""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.txt"
        with open(checkpoint_file, 'a', encoding='utf-8') as f:
            f.write(f"{item_id}\n")
    
    def mark_batch_processed(self, checkpoint_name: str, item_ids: list) -> None:
        """Mark multiple items as processed."""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.txt"
        with open(checkpoint_file, 'a', encoding='utf-8') as f:
            for item_id in item_ids:
                f.write(f"{item_id}\n")
    
    def get_faiss_version(self) -> Dict[str, Any]:
        """Get current FAISS index version info."""
        version_file = self.checkpoint_dir / "faiss_version.json"
        if not version_file.exists():
            return {"version": 0, "last_updated": None, "case_count": 0}
        
        with open(version_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def update_faiss_version(self, case_count: int) -> None:
        """Update FAISS version after index rebuild."""
        version_file = self.checkpoint_dir / "faiss_version.json"
        current = self.get_faiss_version()
        
        new_version = {
            "version": current.get("version", 0) + 1,
            "last_updated": datetime.now().isoformat(),
            "case_count": case_count,
            "previous_count": current.get("case_count", 0)
        }
        
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(new_version, f, indent=2)


class AuditLogger:
    """Clinical audit logging for traceability."""
    
    def __init__(self, log_file: str = None):
        self.log_file = Path(log_file) if log_file else (LOG_ROOT / "audit.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_similarity_match(
        self,
        query_case_id: str,
        matched_case_id: str,
        source: str,
        scores: Dict[str, float],
        composite_score: float,
        flagged: bool
    ) -> None:
        """Log a similarity match for audit purposes."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query_case_id": query_case_id,
            "matched_case_id": matched_case_id,
            "source": source,
            "domain_scores": scores,
            "composite_score": composite_score,
            "flagged": flagged
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")
    
    def log_processing_event(
        self,
        event_type: str,
        case_id: str,
        details: Dict[str, Any]
    ) -> None:
        """Log a processing event."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "case_id": case_id,
            "details": details
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")


class GPUTracker:
    """Track GPU usage to prevent quota exhaustion."""
    
    def __init__(self, log_file: str = None):
        self.log_file = Path(log_file) if log_file else (LOG_ROOT / "gpu_usage.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._start_time: Optional[datetime] = None
    
    def start_session(self) -> None:
        """Mark start of GPU usage session."""
        self._start_time = datetime.now()
        logger.info("GPU session started")
    
    def end_session(self) -> float:
        """Mark end of GPU session and log usage."""
        if self._start_time is None:
            return 0.0
        
        duration = (datetime.now() - self._start_time).total_seconds() / 60
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().date().isoformat()},{duration:.2f}\n")
        
        logger.info(f"GPU session ended: {duration:.2f} minutes")
        self._start_time = None
        return duration
    
    def get_daily_usage(self) -> float:
        """Get total GPU usage for today."""
        if not self.log_file.exists():
            return 0.0
        
        today = datetime.now().date()
        total = 0.0
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        date_str, minutes = parts[0], float(parts[1])
                        if datetime.fromisoformat(date_str).date() == today:
                            total += minutes
                except (ValueError, IndexError):
                    continue
        
        return total


class PipelineOrchestrator:
    """
    Main orchestrator for the clinical case processing pipeline.
    
    This class coordinates all pipeline stages but contains NO business logic.
    Each stage is handled by specialized modules.
    """
    
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.storage_root = STORAGE_ROOT
        self.data_root = DATA_ROOT
        self.log_root = LOG_ROOT
        self.checkpoint_root = CHECKPOINT_ROOT

        self.checkpoint_manager = CheckpointManager(str(self.checkpoint_root))
        self.audit_logger = AuditLogger(str(self.log_root / "audit.log"))
        self.gpu_tracker = GPUTracker(str(self.log_root / "gpu_usage.log"))
        
        # Ensure directory structure exists
        self._ensure_directories()
        
        # Load configurations
        self._load_configs()
        
        # Initialize core modules if available
        self._initialize_core_modules()
    
    def _ensure_directories(self) -> None:
        """Create required directory structure."""
        directories = [
            self.data_root / "raw" / "pubmed",
            self.data_root / "raw" / "ncbi",
            self.data_root / "raw" / "esc",
            self.data_root / "raw" / "aha",
            self.data_root / "raw" / "journals",
            self.data_root / "structured",
            self.data_root / "distilled",
            self.data_root / "embeddings",
            self.data_root / "faiss",
            self.checkpoint_root,
            self.log_root,
        ]

        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Directory structure verified")
    
    def _load_configs(self) -> None:
        """Load all configuration files."""
        try:
            self.sources_config = self.config_loader.load("sources")
            self.schema_config = self.config_loader.load("clinical_schema")
            self.weights_config = self.config_loader.load("weights")
            self.thresholds_config = self.config_loader.load("thresholds")
            self.gpu_policy = self.config_loader.load("gpu_policy")
            
            # Load new configs
            try:
                self.state_machine_config = self.config_loader.load("state_machine")
                self.llm_distillation_config = self.config_loader.load("llm_distillation")
                self.embedding_policy_config = self.config_loader.load("embedding_policy")
                self.faiss_lifecycle_config = self.config_loader.load("faiss_lifecycle")
                self.similarity_engine_config = self.config_loader.load("similarity_engine")
            except FileNotFoundError:
                logger.warning("Some extended configs not found, using defaults")
                
            logger.info("All configurations loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Configuration loading failed: {e}")
            raise
    
    def _initialize_core_modules(self) -> None:
        """Initialize core processing modules."""
        if not CORE_MODULES_AVAILABLE:
            logger.warning("Core modules not available, using placeholder implementations")
            self.state_manager = None
            self.rarity_classifier = None
            self.embedding_manager = None
            self.faiss_manager = None
            self.similarity_engine = None
            self.ingestion_manager = None
            self.normalization_pipeline = None
            return
        
        try:
            # State manager
            self.state_manager = StateManager(
                checkpoint_dir=str(self.checkpoint_root),
                gpu_tracker=self.gpu_tracker
            )
            
            # Rarity classifier
            self.rarity_classifier = RarityClassifier(
                model_name="meta-llama/Llama-2-7b-chat-hf",
                use_quantization=True
            )
            
            # Embedding manager
            self.embedding_manager = EmbeddingManager(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=str(self.data_root / "embeddings")
            )
            
            # FAISS manager
            self.faiss_manager = FAISSIndexManager(
                index_dir=str(self.data_root / "faiss"),
                dimension=384
            )
            
            # Similarity engine
            self.similarity_engine = SimilarityEngine(
                weights=self.weights_config,
                thresholds=self.thresholds_config
            )
            
            # Ingestion manager
            self.ingestion_manager = IngestionManager(
                sources_config=self.sources_config,
                checkpoint_manager=self.checkpoint_manager,
                data_root=self.data_root
            )
            
            # Normalization pipeline
            self.normalization_pipeline = NormalizationPipeline(
                schema_config=self.schema_config,
                checkpoint_manager=self.checkpoint_manager,
                data_root=self.data_root
            )
            
            logger.info("Core modules initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize core modules: {e}")
            self.state_manager = None
            self.rarity_classifier = None
            self.embedding_manager = None
            self.faiss_manager = None
            self.similarity_engine = None
            self.ingestion_manager = None
            self.normalization_pipeline = None
    
    def get_current_stage(self) -> PipelineStage:
        """Determine current pipeline stage based on checkpoints."""
        scraped = self.checkpoint_manager.get_processed_ids("scraped_urls")
        normalized = self.checkpoint_manager.get_processed_ids("normalized_ids")
        reviewed = self.checkpoint_manager.get_processed_ids("llm_reviewed_ids")
        embedded = self.checkpoint_manager.get_processed_ids("embedded_ids")
        
        # Determine stage based on completion
        if len(scraped) == 0:
            return PipelineStage.SCRAPING
        
        if len(normalized) < len(scraped):
            return PipelineStage.NORMALIZATION
        
        if len(reviewed) < len(normalized):
            return PipelineStage.DISTILLATION
        
        if len(embedded) < len(reviewed):
            return PipelineStage.EMBEDDING
        
        faiss_version = self.checkpoint_manager.get_faiss_version()
        if faiss_version.get("case_count", 0) < len(embedded):
            return PipelineStage.INDEXING
        
        return PipelineStage.SERVING
    
    def run_stage(self, stage: PipelineStage, mode: ExecutionMode) -> bool:
        """
        Execute a specific pipeline stage.
        
        Returns True if stage completed successfully.
        """
        logger.info(f"Starting stage: {stage.value} in {mode.value} mode")
        
        try:
            if stage == PipelineStage.SCRAPING:
                return self._run_scraping_stage()
            
            elif stage == PipelineStage.NORMALIZATION:
                return self._run_normalization_stage()
            
            elif stage == PipelineStage.DISTILLATION:
                if mode == ExecutionMode.CPU_ONLY:
                    logger.info("Distillation requires GPU, deferring...")
                    return False
                return self._run_distillation_stage()
            
            elif stage == PipelineStage.EMBEDDING:
                if mode == ExecutionMode.CPU_ONLY:
                    logger.info("Embedding requires GPU, deferring...")
                    return False
                return self._run_embedding_stage()
            
            elif stage == PipelineStage.INDEXING:
                return self._run_indexing_stage()
            
            elif stage == PipelineStage.SERVING:
                return self._run_serving_stage()
            
            else:
                logger.warning(f"Unknown stage: {stage}")
                return False
                
        except Exception as e:
            logger.error(f"Stage {stage.value} failed: {e}", exc_info=True)
            return False
    
    def _run_scraping_stage(self) -> bool:
        """Execute scraping stage (CPU-safe)."""
        logger.info("=== SCRAPING STAGE ===")
        
        if self.ingestion_manager is not None:
            try:
                scraped_count = self.ingestion_manager.run_ingestion(max_cases_per_source=2000)
                logger.info(f"Scraped {scraped_count} new cases")
            except Exception as e:
                logger.error(f"Ingestion failed: {e}")
                return False
        else:
            # Fallback: placeholder logic
            scraped_urls = self.checkpoint_manager.get_processed_ids("scraped_urls")
            
            for source_name, source_config in self.sources_config.get("sources", {}).items():
                if not source_config.get("enabled", False):
                    logger.info(f"Skipping disabled source: {source_name}")
                    continue
                
                logger.info(f"Processing source: {source_name}")
                # Placeholder - actual scraping would happen here
        
        logger.info("Scraping stage completed")
        return True
    
    def _run_normalization_stage(self) -> bool:
        """Execute normalization stage (CPU-safe)."""
        logger.info("=== NORMALIZATION STAGE ===")
        
        if self.normalization_pipeline is not None:
            try:
                normalized_count = self.normalization_pipeline.run()
                logger.info(f"Normalized {normalized_count} cases")
            except Exception as e:
                logger.error(f"Normalization failed: {e}")
                return False
        else:
            # Fallback: placeholder logic
            normalized_ids = self.checkpoint_manager.get_processed_ids("normalized_ids")
            logger.info(f"Normalization placeholder - {len(normalized_ids)} already processed")
        
        logger.info("Normalization stage completed")
        return True
    
    def _run_distillation_stage(self) -> bool:
        """Execute LLM distillation stage (GPU required)."""
        logger.info("=== DISTILLATION STAGE ===")
        
        # Check GPU quota BEFORE starting
        max_minutes = self.gpu_policy.get("max_gpu_minutes_per_day", 20)
        allocation = self.gpu_policy.get("allocation", {})
        distill_budget = allocation.get("distillation", 15)
        current_usage = self.gpu_tracker.get_daily_usage()
        
        if current_usage >= max_minutes:
            logger.warning(f"GPU quota exhausted: {current_usage:.1f}/{max_minutes} min")
            return False
        
        self.gpu_tracker.start_session()
        
        try:
            if self.rarity_classifier is not None:
                # Load model (2 min budget)
                logger.info("Loading LLM for distillation...")
                self.rarity_classifier.load_model()
                
                # Get cases to process
                reviewed_ids = self.checkpoint_manager.get_processed_ids("llm_reviewed_ids")
                structured_dir = self.data_root / "structured"

                distilled_dir = self.data_root / "distilled"
                distilled_dir.mkdir(parents=True, exist_ok=True)
                
                processed = 0
                start_time = time.time()
                
                for case_file in structured_dir.glob("*_normalized.json"):
                    # Check time budget (15 min for distillation)
                    elapsed = (time.time() - start_time) / 60
                    if elapsed >= distill_budget:
                        logger.info(f"Distillation budget exhausted: {elapsed:.1f} min")
                        break
                    
                    case_id = case_file.stem.replace("_normalized", "")
                    if case_id in reviewed_ids:
                        continue
                    
                    try:
                        with open(case_file, 'r', encoding='utf-8') as f:
                            case_data = json.load(f)
                        
                        # Classify rarity
                        classification = self.rarity_classifier.classify(case_data)
                        
                        # Save result
                        result = {
                            "case_id": case_id,
                            "is_rare": classification.is_rare,
                            "rarity_criteria": classification.criteria_met,
                            "confidence": classification.confidence,
                            "explanation": classification.explanation,
                            "classified_at": datetime.now().isoformat()
                        }
                        
                        with open(distilled_dir / f"{case_id}_distilled.json", 'w') as f:
                            json.dump(result, f, indent=2)
                        
                        self.checkpoint_manager.mark_processed("llm_reviewed_ids", case_id)
                        processed += 1
                        
                        # Log to audit
                        self.audit_logger.log_processing_event(
                            "distillation",
                            case_id,
                            {"is_rare": classification.is_rare, "confidence": classification.confidence}
                        )
                        
                    except Exception as e:
                        logger.error(f"Distillation failed for {case_id}: {e}")
                        continue
                
                logger.info(f"Distilled {processed} cases")
                
            else:
                # Placeholder
                reviewed_ids = self.checkpoint_manager.get_processed_ids("llm_reviewed_ids")
                logger.info(f"Distillation placeholder - {len(reviewed_ids)} already reviewed")
            
            return True
            
        finally:
            self.gpu_tracker.end_session()
    
    def _run_embedding_stage(self) -> bool:
        """Execute embedding generation stage (GPU required)."""
        logger.info("=== EMBEDDING STAGE ===")
        
        # Check GPU quota
        max_minutes = self.gpu_policy.get("max_gpu_minutes_per_day", 20)
        current_usage = self.gpu_tracker.get_daily_usage()
        
        if current_usage >= max_minutes:
            logger.warning(f"GPU quota exhausted: {current_usage:.1f}/{max_minutes} min")
            return False
        
        self.gpu_tracker.start_session()
        
        try:
            if self.embedding_manager is not None:
                # Load model
                self.embedding_manager.load_model()
                
                embedded_ids = self.checkpoint_manager.get_processed_ids("embedded_ids")
                distilled_dir = self.data_root / "distilled"
                embeddings_dir = self.data_root / "embeddings"
                embeddings_dir.mkdir(parents=True, exist_ok=True)
                
                processed = 0
                
                for distilled_file in distilled_dir.glob("*_distilled.json"):
                    case_id = distilled_file.stem.replace("_distilled", "")
                    
                    if case_id in embedded_ids:
                        continue
                    
                    try:
                        # Load normalized case
                        normalized_file = self.data_root / "structured" / f"{case_id}_normalized.json"
                        if not normalized_file.exists():
                            continue
                        
                        with open(normalized_file, 'r', encoding='utf-8') as f:
                            case_data = json.load(f)
                        
                        # Generate embeddings per domain
                        embeddings = self.embedding_manager.generate_domain_embeddings(case_data)
                        
                        # Save embeddings
                        import numpy as np
                        np.savez(
                            embeddings_dir / f"{case_id}_embeddings.npz",
                            **embeddings
                        )
                        
                        self.checkpoint_manager.mark_processed("embedded_ids", case_id)
                        processed += 1
                        
                    except Exception as e:
                        logger.error(f"Embedding failed for {case_id}: {e}")
                        continue
                
                logger.info(f"Generated embeddings for {processed} cases")
                
            else:
                embedded_ids = self.checkpoint_manager.get_processed_ids("embedded_ids")
                logger.info(f"Embedding placeholder - {len(embedded_ids)} already embedded")
            
            return True
            
        finally:
            self.gpu_tracker.end_session()
    
    def _run_indexing_stage(self) -> bool:
        """Execute FAISS indexing stage (CPU or GPU)."""
        logger.info("=== INDEXING STAGE ===")
        
        if self.faiss_manager is not None:
            try:
                embeddings_dir = self.data_root / "embeddings"
                
                # Load existing index or create new
                self.faiss_manager.load_index()
                
                indexed_ids = self.checkpoint_manager.get_processed_ids("indexed_ids")
                added = 0
                
                import numpy as np
                
                for emb_file in embeddings_dir.glob("*_embeddings.npz"):
                    case_id = emb_file.stem.replace("_embeddings", "")
                    
                    if case_id in indexed_ids:
                        continue
                    
                    try:
                        data = np.load(emb_file)
                        
                        # Add to index (using combined embedding)
                        if "combined" in data:
                            self.faiss_manager.add_vector(case_id, data["combined"])
                        
                        self.checkpoint_manager.mark_processed("indexed_ids", case_id)
                        added += 1
                        
                    except Exception as e:
                        logger.error(f"Indexing failed for {case_id}: {e}")
                        continue
                
                # Save index
                self.faiss_manager.save_index()
                
                # Update version
                total_indexed = len(self.checkpoint_manager.get_processed_ids("indexed_ids"))
                self.checkpoint_manager.update_faiss_version(total_indexed)
                
                logger.info(f"Added {added} vectors to index (total: {total_indexed})")
                
            except Exception as e:
                logger.error(f"Indexing failed: {e}")
                return False
        else:
            logger.info("FAISS indexing placeholder")
        
        logger.info("Indexing stage completed")
        return True

    def _ensure_vector_db_initialized(self) -> None:
        """Ensure a FAISS index file exists in persistent storage."""
        if self.faiss_manager is None:
            return
        try:
            self.faiss_manager.load_index()
            # Persist even if empty, so the "vector DB" exists on disk.
            self.faiss_manager.save_index()
        except Exception as e:
            logger.warning(f"Vector DB initialization failed: {e}")
    
    def _run_serving_stage(self) -> bool:
        """Enter serving mode for similarity queries."""
        logger.info("=== SERVING STAGE ===")
        logger.info("System ready for similarity queries")
        return True
    
    def run_daily_pipeline(self) -> None:
        """Execute the complete daily pipeline based on current state."""
        logger.info("=" * 60)
        logger.info("Starting daily pipeline execution")
        logger.info("=" * 60)
        
        # Detect execution mode
        mode = HardwareDetector.get_execution_mode(self.gpu_policy)
        logger.info(f"Execution mode: {mode.value}")
        
        # Get current stage
        current_stage = self.get_current_stage()
        logger.info(f"Current stage: {current_stage.value}")
        
        # Execute stages in order
        stages = [
            PipelineStage.SCRAPING,
            PipelineStage.NORMALIZATION,
            PipelineStage.DISTILLATION,
            PipelineStage.EMBEDDING,
            PipelineStage.INDEXING,
            PipelineStage.SERVING,
        ]
        
        stage_index = stages.index(current_stage) if current_stage in stages else 0
        
        for stage in stages[stage_index:]:
            success = self.run_stage(stage, mode)
            
            if not success:
                if stage in [PipelineStage.DISTILLATION, PipelineStage.EMBEDDING]:
                    if mode == ExecutionMode.CPU_ONLY:
                        logger.info(f"Stage {stage.value} deferred (no GPU)")
                        break
                    else:
                        logger.error(f"Stage {stage.value} failed")
                        break
                else:
                    logger.error(f"Stage {stage.value} failed")
                    break
        
        logger.info("=" * 60)
        logger.info("Daily pipeline execution completed")
        logger.info("=" * 60)

        # Always ensure the vector database exists on disk (HF Spaces persistent storage).
        self._ensure_vector_db_initialized()


def create_gradio_interface(orchestrator: PipelineOrchestrator):
    """Create Gradio interface for Hugging Face Spaces."""
    try:
        import gradio as gr
    except ImportError:
        logger.warning("Gradio not installed, skipping interface creation")
        return None
    
    def query_similar_cases(
        symptoms: str,
        ecg_findings: str,
        lab_values: str,
        demographics: str,
        imaging: str
    ) -> str:
        """Query for similar cases based on clinical input."""
        if orchestrator.similarity_engine is not None and orchestrator.faiss_manager is not None:
            try:
                # Build query case
                query_case = {
                    "case_id": f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "presenting_symptoms": [s.strip() for s in symptoms.split(",") if s.strip()],
                    "chief_complaint": symptoms[:200] if symptoms else None,
                    "ecg_abnormalities": [e.strip() for e in ecg_findings.split(",") if e.strip()],
                    "ecg_interpretation": ecg_findings,
                    "raw_text": f"{symptoms} {ecg_findings} {lab_values} {demographics} {imaging}",
                }
                
                # Parse demographics
                if demographics:
                    import re
                    age_match = re.search(r'(\d+)\s*(year|yr|y)', demographics.lower())
                    if age_match:
                        query_case["age"] = int(age_match.group(1))
                    if "male" in demographics.lower() or " m " in f" {demographics.lower()} ":
                        query_case["sex"] = "M"
                    elif "female" in demographics.lower() or " f " in f" {demographics.lower()} ":
                        query_case["sex"] = "F"
                
                # Generate embeddings for query
                if orchestrator.embedding_manager is not None:
                    orchestrator.embedding_manager.load_model()
                    query_embeddings = orchestrator.embedding_manager.generate_domain_embeddings(query_case)
                    
                    # Search FAISS
                    if "combined" in query_embeddings:
                        similar_ids = orchestrator.faiss_manager.search(
                            query_embeddings["combined"],
                            k=10
                        )
                        
                        results = []
                        for case_id, distance in similar_ids:
                            # Load case details
                            case_file = orchestrator.data_root / "structured" / f"{case_id}_normalized.json"
                            if case_file.exists():
                                with open(case_file, 'r') as f:
                                    case_data = json.load(f)
                                
                                # Compute domain similarity
                                result = orchestrator.similarity_engine.compute_similarity(
                                    query_case, case_data
                                )
                                
                                results.append({
                                    "case_id": case_id,
                                    "composite_score": round(result.composite_score, 3),
                                    "domain_scores": {k: round(v, 3) for k, v in result.domain_scores.items()},
                                    "flagged": result.flag_decision.should_flag if result.flag_decision else False,
                                    "source": case_data.get("source", "unknown"),
                                    "diagnosis": case_data.get("primary_diagnosis"),
                                })
                        
                        return json.dumps({
                            "status": "success",
                            "results": results,
                            "disclaimer": "⚠️ Clinical decision SUPPORT only. Not for diagnosis."
                        }, indent=2)
                
            except Exception as e:
                logger.error(f"Similarity query failed: {e}")
                return json.dumps({
                    "status": "error",
                    "message": str(e),
                    "disclaimer": "⚠️ Clinical decision SUPPORT only. Not for diagnosis."
                }, indent=2)
        
        return json.dumps({
            "status": "pending",
            "message": "System initializing - similarity search not yet available",
            "disclaimer": "⚠️ Clinical decision SUPPORT only. Not for diagnosis."
        }, indent=2)
    
    def get_system_status() -> str:
        """Get current system status."""
        mode = HardwareDetector.get_execution_mode(orchestrator.gpu_policy)
        stage = orchestrator.get_current_stage()
        faiss_version = orchestrator.checkpoint_manager.get_faiss_version()
        gpu_usage = orchestrator.gpu_tracker.get_daily_usage()
        
        # Get checkpoint counts
        scraped = len(orchestrator.checkpoint_manager.get_processed_ids("scraped_urls"))
        normalized = len(orchestrator.checkpoint_manager.get_processed_ids("normalized_ids"))
        reviewed = len(orchestrator.checkpoint_manager.get_processed_ids("llm_reviewed_ids"))
        embedded = len(orchestrator.checkpoint_manager.get_processed_ids("embedded_ids"))
        indexed = len(orchestrator.checkpoint_manager.get_processed_ids("indexed_ids"))

        faiss_index_path = orchestrator.data_root / "faiss" / "combined.index"
        faiss_meta_path = orchestrator.data_root / "faiss" / "vector_metadata.jsonl"
        
        return json.dumps({
            "execution_mode": mode.value,
            "current_stage": stage.value,
            "pipeline_progress": {
                "scraped": scraped,
                "normalized": normalized,
                "llm_reviewed": reviewed,
                "embedded": embedded,
                "indexed": indexed,
            },
            "faiss_version": faiss_version,
            "vector_db": {
                "storage_root": str(orchestrator.storage_root),
                "data_root": str(orchestrator.data_root),
                "faiss_index_path": str(faiss_index_path),
                "faiss_metadata_path": str(faiss_meta_path),
                "faiss_index_exists": faiss_index_path.exists(),
                "faiss_metadata_exists": faiss_meta_path.exists(),
            },
            "gpu_usage_today_minutes": round(gpu_usage, 2),
            "gpu_quota_minutes": orchestrator.gpu_policy.get("max_gpu_minutes_per_day", 20),
            "gpu_remaining_minutes": round(
                orchestrator.gpu_policy.get("max_gpu_minutes_per_day", 20) - gpu_usage, 2
            )
        }, indent=2)
    
    # Build Gradio interface
    with gr.Blocks(title="Clinical Case Similarity System") as interface:
        gr.Markdown("""
        # 🏥 Clinical Case Similarity Detection System
        
        **⚠️ IMPORTANT DISCLAIMER:**
        - This is a clinical decision SUPPORT tool only
        - NOT for direct patient diagnosis
        - Results must be reviewed by qualified clinicians
        - For research and educational purposes
        """)
        
        with gr.Tab("Case Query"):
            with gr.Row():
                with gr.Column():
                    symptoms_input = gr.Textbox(
                        label="Symptoms",
                        placeholder="Enter symptom descriptions...",
                        lines=3
                    )
                    ecg_input = gr.Textbox(
                        label="ECG Findings",
                        placeholder="Enter ECG abnormalities...",
                        lines=2
                    )
                    labs_input = gr.Textbox(
                        label="Laboratory Values",
                        placeholder="Enter lab results...",
                        lines=2
                    )
                    demographics_input = gr.Textbox(
                        label="Demographics",
                        placeholder="Age, sex, relevant history...",
                        lines=2
                    )
                    imaging_input = gr.Textbox(
                        label="Imaging Findings",
                        placeholder="Enter imaging results...",
                        lines=2
                    )
                    
                    query_btn = gr.Button("🔍 Find Similar Cases", variant="primary")
                
                with gr.Column():
                    results_output = gr.Textbox(
                        label="Similar Cases",
                        lines=15,
                        interactive=False
                    )
            
            query_btn.click(
                fn=query_similar_cases,
                inputs=[symptoms_input, ecg_input, labs_input, demographics_input, imaging_input],
                outputs=results_output
            )
        
        with gr.Tab("System Status"):
            status_output = gr.Textbox(
                label="System Status",
                lines=10,
                interactive=False
            )
            refresh_btn = gr.Button("🔄 Refresh Status")
            refresh_btn.click(fn=get_system_status, outputs=status_output)
    
    return interface


def main():
    """Main entry point for the application."""
    logger.info("=" * 60)
    logger.info("Clinical Case Similarity System Starting")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    try:
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator()
        
        # Check if running in Hugging Face Spaces
        is_hf_space = os.environ.get("SPACE_ID") is not None
        
        if is_hf_space:
            logger.info("Running in Hugging Face Spaces")
            
            # Create and launch Gradio interface
            interface = create_gradio_interface(orchestrator)
            if interface:
                # Run pipeline in background
                import threading
                pipeline_thread = threading.Thread(
                    target=orchestrator.run_daily_pipeline,
                    daemon=True
                )
                pipeline_thread.start()
                
                # Launch Gradio
                interface.launch(server_name="0.0.0.0", server_port=7860)
            else:
                logger.error("Failed to create Gradio interface")
                sys.exit(1)
        else:
            # Local execution - run pipeline directly
            logger.info("Running in local mode")
            orchestrator.run_daily_pipeline()
    
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
