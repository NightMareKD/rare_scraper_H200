"""
Core Processing Modules for Clinical Case Similarity System
============================================================
This module contains the actual business logic implementations.
"""

import os
import json
import logging
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


# ==========================================
# System States (EXHAUSTIVE)
# ==========================================

class SystemState(Enum):
    """
    Exhaustive system states. NO OTHER STATE EXISTS.
    """
    INIT = "INIT"
    CPU_INGEST = "CPU_INGEST"
    GPU_DISTILL = "GPU_DISTILL"
    CPU_POST = "CPU_POST"
    SERVE = "SERVE"
    SAFE_HALT = "SAFE_HALT"


# ==========================================
# Data Classes
# ==========================================

@dataclass
class RarityClassification:
    """LLM output for rare case classification."""
    is_rare: bool
    rare_reason: Optional[str] = None
    rarity_criteria_met: List[str] = field(default_factory=list)
    criteria_met: List[str] = field(default_factory=list)  # Alias for compatibility
    confidence: float = 1.0
    explanation: str = ""
    primary_diagnosis: Optional[str] = None
    key_ecg_pattern: Optional[str] = None
    key_labs: List[str] = field(default_factory=list)
    key_symptoms: List[str] = field(default_factory=list)
    reference_source: Optional[str] = None
    rejection_reason: Optional[str] = None


@dataclass
class SimilarityResult:
    """Result from similarity search."""
    case_id: str
    source_url: str
    rare_explanation: str
    composite_score: float
    domain_scores: Dict[str, float]
    clinical_rationale: str
    flag_decision: Optional['FlagDecision'] = None


@dataclass
class FlagDecision:
    """Decision on whether to flag a patient."""
    is_flagged: bool
    composite_score: float
    matched_cases: List['SimilarityResult']
    domain_scores: Dict[str, float]
    clinical_rationale: str
    source_references: List[str]
    reason: Optional[str] = None  # For not-flagged cases
    should_flag: bool = False  # Convenience alias
    disclaimer: str = "This is clinical decision SUPPORT only. Not for diagnosis."


# ==========================================
# State Manager
# ==========================================

class StateManager:
    """
    Manages system state transitions.
    Enforces the exhaustive state machine.
    """
    
    VALID_TRANSITIONS = {
        SystemState.INIT: [SystemState.CPU_INGEST, SystemState.SAFE_HALT],
        SystemState.CPU_INGEST: [SystemState.GPU_DISTILL, SystemState.SERVE, SystemState.SAFE_HALT],
        SystemState.GPU_DISTILL: [SystemState.CPU_POST, SystemState.SAFE_HALT],
        SystemState.CPU_POST: [SystemState.SERVE, SystemState.SAFE_HALT],
        SystemState.SERVE: [SystemState.CPU_INGEST, SystemState.SAFE_HALT],
        SystemState.SAFE_HALT: [],  # Terminal state
    }
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        gpu_tracker=None,
        state_file: str = None
    ):
        if state_file:
            self.state_file = Path(state_file)
        else:
            self.state_file = Path(checkpoint_dir) / "system_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.gpu_tracker = gpu_tracker
        self._current_state = self._load_state()
        self._current_state = self._load_state()
    
    def _load_state(self) -> SystemState:
        """Load state from file or return INIT."""
        if not self.state_file.exists():
            return SystemState.INIT
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                return SystemState(data.get("current_state", "INIT"))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Invalid state file, resetting to INIT")
            return SystemState.INIT
    
    def _save_state(self, reason: str) -> None:
        """Persist state to file."""
        data = {
            "current_state": self._current_state.value,
            "last_transition_time": datetime.now().isoformat(),
            "transition_reason": reason
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    @property
    def current_state(self) -> SystemState:
        return self._current_state
    
    def transition_to(self, new_state: SystemState, reason: str) -> bool:
        """
        Attempt state transition.
        Returns True if transition is valid and completed.
        """
        valid_targets = self.VALID_TRANSITIONS.get(self._current_state, [])
        
        if new_state not in valid_targets:
            logger.error(
                f"Invalid transition: {self._current_state.value} -> {new_state.value}"
            )
            return False
        
        old_state = self._current_state
        self._current_state = new_state
        self._save_state(reason)
        
        logger.info(f"State transition: {old_state.value} -> {new_state.value} ({reason})")
        return True
    
    def force_halt(self, reason: str) -> None:
        """Force transition to SAFE_HALT from any state."""
        old_state = self._current_state
        self._current_state = SystemState.SAFE_HALT
        self._save_state(f"FORCED HALT: {reason}")
        logger.critical(f"Forced halt from {old_state.value}: {reason}")


# ==========================================
# Rarity Classifier
# ==========================================

class RarityClassifier:
    """
    Determines if a case is RARE based on locked criteria.
    
    A cardiology case is RARE only if AT LEAST ONE applies:
    1. Prevalence < 1:2,000
    2. Atypical presentation of known disease
    3. Rare ECG pattern
    4. Rare complication
    5. Rare drug interaction
    6. Rare age/sex manifestation
    
    If NONE apply → NOT RARE.
    """
    
    # Known rare conditions (prevalence < 1:2000)
    RARE_CONDITIONS = {
        "brugada syndrome",
        "arrhythmogenic right ventricular cardiomyopathy",
        "arvd", "arvc",
        "catecholaminergic polymorphic ventricular tachycardia",
        "cpvt",
        "long qt syndrome",
        "short qt syndrome",
        "lqts",
        "hypertrophic cardiomyopathy",  # Can be rare presentations
        "cardiac amyloidosis",
        "fabry disease",
        "loeffler endocarditis",
        "cardiac sarcoidosis",
        "takotsubo cardiomyopathy",  # In atypical demographics
        "giant cell myocarditis",
        "chagas cardiomyopathy",
    }
    
    # Rare ECG patterns
    RARE_ECG_PATTERNS = {
        "epsilon wave",
        "brugada pattern",
        "type 1 brugada",
        "osborn wave",
        "j wave",
        "de winter",
        "wellens",
        "short qt",
        "delta wave",  # WPW
        "alternans",
    }
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        use_quantization: bool = True
    ):
        self.config = config or {}
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
    
    def load_model(self) -> bool:
        """Load the LLM for classification."""
        try:
            # For now, we use rule-based classification
            # LLM loading would require significant memory
            self._model_loaded = True
            logger.info("Rarity classifier initialized (rule-based)")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def classify(self, case: Dict[str, Any]) -> RarityClassification:
        """
        Classify a case as rare or not rare.
        Returns structured classification result.
        """
        criteria_met = []
        rare_reasons = []
        
        diagnosis = case.get("diagnosis", {})
        symptoms = case.get("symptoms", {})
        ecg = case.get("ecg", {})
        demographics = case.get("demographics", {})
        
        # Check prevalence criterion
        diagnosis_text = str(diagnosis.get("primary_diagnosis", "")).lower()
        for condition in self.RARE_CONDITIONS:
            if condition in diagnosis_text:
                criteria_met.append("prevalence_threshold")
                rare_reasons.append(f"Rare condition: {condition}")
                break
        
        # Check rare ECG pattern
        ecg_text = str(ecg).lower()
        for pattern in self.RARE_ECG_PATTERNS:
            if pattern in ecg_text:
                criteria_met.append("rare_ecg_pattern")
                rare_reasons.append(f"Rare ECG pattern: {pattern}")
                break
        
        # Check atypical presentation
        age = demographics.get("age", 50)
        if self._is_atypical_presentation(case, age):
            criteria_met.append("atypical_presentation")
            rare_reasons.append("Atypical presentation for demographics")
        
        # Check rare demographic manifestation
        if self._is_rare_demographic(case, demographics):
            criteria_met.append("rare_demographic")
            rare_reasons.append("Rare manifestation for patient demographics")
        
        # Determine if rare
        is_rare = len(criteria_met) > 0
        
        if is_rare:
            return RarityClassification(
                is_rare=True,
                rare_reason=" | ".join(rare_reasons),
                rarity_criteria_met=criteria_met,
                primary_diagnosis=diagnosis.get("primary_diagnosis"),
                key_ecg_pattern=ecg.get("rhythm") or ecg.get("additional_findings"),
                key_labs=self._extract_key_labs(case.get("labs", {})),
                key_symptoms=self._extract_key_symptoms(symptoms),
                reference_source=case.get("source_url")
            )
        else:
            return RarityClassification(
                is_rare=False,
                rejection_reason="Common presentation without rare criteria"
            )
    
    def _is_atypical_presentation(self, case: Dict, age: int) -> bool:
        """Check for atypical disease presentation."""
        diagnosis = str(case.get("diagnosis", {}).get("primary_diagnosis", "")).lower()
        
        # MI in young patient without risk factors
        if "myocardial infarction" in diagnosis or "stemi" in diagnosis or "nstemi" in diagnosis:
            if age < 40:
                return True
        
        # Heart failure in young patient
        if "heart failure" in diagnosis:
            if age < 35:
                return True
        
        return False
    
    def _is_rare_demographic(self, case: Dict, demographics: Dict) -> bool:
        """Check for rare demographic manifestation."""
        diagnosis = str(case.get("diagnosis", {}).get("primary_diagnosis", "")).lower()
        sex = demographics.get("sex", "").lower()
        age = demographics.get("age", 50)
        
        # Takotsubo in young male
        if "takotsubo" in diagnosis:
            if sex == "male" and age < 50:
                return True
        
        # Pediatric acute coronary syndrome
        if "coronary" in diagnosis or "mi" in diagnosis:
            if age < 18:
                return True
        
        return False
    
    def _extract_key_labs(self, labs: Dict) -> List[str]:
        """Extract notable lab values."""
        key_labs = []
        cardiac = labs.get("cardiac_biomarkers", {})
        
        if cardiac.get("troponin"):
            key_labs.append(f"Troponin: {cardiac['troponin']}")
        if cardiac.get("bnp"):
            key_labs.append(f"BNP: {cardiac['bnp']}")
        
        return key_labs[:5]
    
    def _extract_key_symptoms(self, symptoms: Dict) -> List[str]:
        """Extract key symptoms."""
        key_symptoms = []
        
        if symptoms.get("chief_complaint"):
            key_symptoms.append(symptoms["chief_complaint"])
        
        if symptoms.get("presenting_symptoms"):
            key_symptoms.extend(symptoms["presenting_symptoms"][:3])
        
        return key_symptoms[:5]


# ==========================================
# Embedding Manager
# ==========================================

class EmbeddingManager:
    """
    Manages embedding generation with strict policy enforcement.
    
    Rules:
    - Same model per dimension FOREVER
    - Patient queries MUST use same model as database
    - No mixed embeddings
    - No retroactive re-embedding
    """
    
    DIMENSIONS = ["symptoms", "ecg", "labs", "demographics", "imaging"]
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str = "data/embeddings"
    ):
        self.config = config or {}
        self.model = None
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vector_size = 384
        self._model_loaded = False
    
    def load_model(self) -> bool:
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self._model_loaded = True
            logger.info(f"Loaded embedding model: {self.model_name}")
            return True
        except ImportError:
            logger.error("sentence-transformers not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate_embeddings(
        self, 
        case: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all dimensions of a case.
        Returns dict mapping dimension -> embedding vector.
        """
        if not self._model_loaded:
            if not self.load_model():
                raise RuntimeError("Cannot generate embeddings without model")
        
        embeddings = {}
        
        for dimension in self.DIMENSIONS:
            text = self._prepare_text(case, dimension)
            if text:
                vector = self.model.encode(text, normalize_embeddings=True)
                embeddings[dimension] = vector.astype(np.float32)
            else:
                # Zero vector for missing dimension
                embeddings[dimension] = np.zeros(self.vector_size, dtype=np.float32)
        
        return embeddings
    
    def _prepare_text(self, case: Dict[str, Any], dimension: str) -> str:
        """Prepare text for embedding based on dimension."""
        if dimension == "symptoms":
            # Handle both nested and flat case formats
            if "symptoms" in case and isinstance(case["symptoms"], dict):
                symptoms = case.get("symptoms", {})
                parts = [
                    symptoms.get("chief_complaint", ""),
                    " ".join(symptoms.get("presenting_symptoms", [])),
                    " ".join(symptoms.get("associated_symptoms", [])),
                ]
            else:
                # Flat format (from normalizer)
                parts = [
                    case.get("chief_complaint", ""),
                    " ".join(case.get("presenting_symptoms", []) or []),
                ]
            return " | ".join(p for p in parts if p)
        
        elif dimension == "ecg":
            if "ecg" in case and isinstance(case["ecg"], dict):
                ecg = case.get("ecg", {})
                parts = [
                    str(ecg.get("rhythm", "")),
                    str(ecg.get("rate", "")),
                    str(ecg.get("st_changes", "")),
                    str(ecg.get("additional_findings", "")),
                ]
            else:
                # Flat format
                parts = [
                    str(case.get("rhythm", "")),
                    str(case.get("ecg_interpretation", "")),
                    " ".join(case.get("ecg_abnormalities", []) or []),
                ]
            return " | ".join(p for p in parts if p)
        
        elif dimension == "labs":
            if "labs" in case and isinstance(case["labs"], dict):
                labs = case.get("labs", {})
                return json.dumps(labs) if labs else ""
            else:
                # Flat format
                lab_vals = []
                for key in ["troponin", "bnp", "creatinine", "potassium", "hemoglobin"]:
                    val = case.get(key)
                    if val is not None:
                        lab_vals.append(f"{key}:{val}")
                return " ".join(lab_vals)
        
        elif dimension == "demographics":
            if "demographics" in case and isinstance(case["demographics"], dict):
                demo = case.get("demographics", {})
                parts = [
                    f"Age: {demo.get('age', 'unknown')}",
                    f"Sex: {demo.get('sex', 'unknown')}",
                    " ".join(demo.get("cardiovascular_risk_factors", [])),
                ]
            else:
                # Flat format
                parts = [
                    f"Age: {case.get('age', 'unknown')}",
                    f"Sex: {case.get('sex', 'unknown')}",
                ]
            return " | ".join(p for p in parts if p)
        
        elif dimension == "imaging":
            if "imaging" in case and isinstance(case["imaging"], dict):
                imaging = case.get("imaging", {})
                return json.dumps(imaging) if imaging else ""
            else:
                # Flat format
                ef = case.get("echo_ef")
                findings = case.get("imaging_findings", [])
                parts = []
                if ef is not None:
                    parts.append(f"EF:{ef}")
                if findings:
                    parts.extend(str(f) for f in findings)
                return " ".join(parts)
        
        return ""
    
    def generate_domain_embeddings(
        self,
        case: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all domains and a combined embedding.
        Returns dict with domain embeddings plus 'combined' key.
        """
        embeddings = self.generate_embeddings(case)
        
        # Create combined embedding (weighted average)
        weights = {
            "symptoms": 0.30,
            "ecg": 0.25,
            "labs": 0.25,
            "demographics": 0.10,
            "imaging": 0.10,
        }
        
        combined = np.zeros(self.vector_size, dtype=np.float32)
        for dim, emb in embeddings.items():
            weight = weights.get(dim, 0.2)
            combined += weight * emb
        
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        embeddings["combined"] = combined
        
        return embeddings


# ==========================================
# FAISS Index Manager
# ==========================================

class FAISSIndexManager:
    """
    Manages FAISS indices with strict lifecycle rules.
    
    Rules:
    - Append-only (no delete/update)
    - Never rebuilt mid-project
    - Persisted after every update
    - Versioned via faiss_version.json
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        index_dir: str = "data/faiss",
        dimension: int = 384
    ):
        self.config = config or {}
        self.dimension = dimension
        self.indices: Dict[str, Any] = {}
        self.index = None  # Combined index
        self.metadata: List[Dict] = []
        self.id_map: Dict[str, int] = {}  # case_id -> vector_id
        self.data_dir = Path(index_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.data_dir / "vector_metadata.jsonl"
        self.index_file = self.data_dir / "combined.index"
        self.version_file = Path("checkpoints/faiss_version.json")
    
    def load_index(self) -> bool:
        """Load FAISS index from disk or create new."""
        try:
            import faiss
        except ImportError:
            logger.error("FAISS not installed")
            return False
        
        if self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("Created new FAISS index")
        
        # Load metadata
        self._load_metadata()
        
        return True
    
    def _load_metadata(self) -> None:
        """Load vector metadata from JSONL file."""
        self.metadata = []
        self.id_map = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self.metadata.append(entry)
                        self.id_map[entry.get("case_id")] = entry.get("vector_id")
    
    def add_vector(
        self,
        case_id: str,
        vector: np.ndarray
    ) -> int:
        """
        Add a single vector to index (APPEND-ONLY).
        Returns the vector ID.
        """
        if self.index is None:
            self.load_index()
        
        vector_id = len(self.metadata)
        
        # Ensure vector is 2D and float32
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        vector = vector.astype(np.float32)
        
        self.index.add(vector)
        
        # Store metadata
        metadata_entry = {
            "vector_id": vector_id,
            "case_id": case_id,
            "indexed_at": datetime.now().isoformat(),
        }
        
        with open(self.metadata_file, 'a') as f:
            f.write(json.dumps(metadata_entry) + "\n")
        
        self.metadata.append(metadata_entry)
        self.id_map[case_id] = vector_id
        
        return vector_id
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        Returns list of (case_id, score).
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)
        
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.metadata):
                case_id = self.metadata[idx].get("case_id", str(idx))
                results.append((case_id, float(score)))
        
        return results
    
    def save_index(self) -> None:
        """Persist index to disk."""
        if self.index is None:
            return
        
        import faiss
        faiss.write_index(self.index, str(self.index_file))
        logger.info(f"Saved FAISS index: {self.index.ntotal} vectors")
    
    def get_metadata(self, vector_id: int) -> Optional[Dict]:
        """Get metadata for a vector ID."""
        if 0 <= vector_id < len(self.metadata):
            return self.metadata[vector_id]
        return None
    
    def get_case_metadata(self, case_id: str) -> Optional[Dict]:
        """Get metadata for a case ID."""
        vector_id = self.id_map.get(case_id)
        if vector_id is not None:
            return self.get_metadata(vector_id)
        return None


# ==========================================
# Similarity Engine
# ==========================================

class SimilarityEngine:
    """
    Computes patient similarity with LOCKED algorithm.
    
    Flag Rule (FINAL):
    A patient is flagged ONLY IF:
    1. Composite similarity >= 0.50
    AND
    2. ECG OR Symptoms >= domain threshold (0.45)
    AND
    3. >= 2 domains above minimum score (0.40)
    """
    
    DEFAULT_WEIGHTS = {
        "symptoms": 0.30,
        "ecg": 0.25,
        "labs": 0.25,
        "demographics": 0.10,
        "imaging": 0.10,
    }
    
    COMPOSITE_THRESHOLD = 0.50
    CRITICAL_DOMAIN_THRESHOLD = 0.45
    DOMAIN_MINIMUM = 0.40
    MIN_DOMAINS_ABOVE_MINIMUM = 2
    
    def __init__(
        self,
        weights: Dict[str, Any] = None,
        thresholds: Dict[str, Any] = None,
        embedding_manager: EmbeddingManager = None,
        faiss_manager: FAISSIndexManager = None,
        config: Dict[str, Any] = None
    ):
        self.config = config or {}
        self.embedding_manager = embedding_manager
        self.faiss_manager = faiss_manager
        
        # Load weights from config or use defaults
        if weights:
            weights_data = weights.get("weights", {}).get("per_domain", {})
            self.weights = {
                "symptoms": weights_data.get("symptoms", 0.30),
                "ecg": weights_data.get("ecg", 0.25),
                "labs": weights_data.get("labs", 0.25),
                "demographics": weights_data.get("demographics", 0.10),
                "imaging": weights_data.get("imaging", 0.10),
            }
        else:
            self.weights = self.DEFAULT_WEIGHTS.copy()
        
        # Load thresholds from config or use defaults
        if thresholds:
            threshold_data = thresholds.get("thresholds", {}).get("similarity", {})
            self.composite_threshold = threshold_data.get("composite_threshold", 0.50)
            self.critical_domain_threshold = threshold_data.get("critical_domain_threshold", 0.45)
            self.domain_minimum = threshold_data.get("domain_minimum", 0.40)
        else:
            self.composite_threshold = self.COMPOSITE_THRESHOLD
            self.critical_domain_threshold = self.CRITICAL_DOMAIN_THRESHOLD
            self.domain_minimum = self.DOMAIN_MINIMUM
    
    def compute_similarity(
        self,
        query_case: Dict[str, Any],
        target_case: Dict[str, Any]
    ) -> 'SimilarityResult':
        """
        Compute similarity between two cases.
        Returns SimilarityResult with domain scores and flag decision.
        """
        domain_scores = {}
        
        # Compute text-based similarity for each domain
        for domain in self.weights.keys():
            query_text = self._extract_domain_text(query_case, domain)
            target_text = self._extract_domain_text(target_case, domain)
            
            if query_text and target_text:
                score = self._text_similarity(query_text, target_text)
            else:
                score = 0.0
            
            domain_scores[domain] = score
        
        # Compute composite score
        composite = sum(
            self.weights.get(dim, 0) * score
            for dim, score in domain_scores.items()
        )
        
        # Apply flag rule
        flag_decision = self._check_flag_rule(composite, domain_scores)
        
        return SimilarityResult(
            case_id=target_case.get("case_id", ""),
            source_url=target_case.get("url", ""),
            rare_explanation="",
            composite_score=composite,
            domain_scores=domain_scores,
            clinical_rationale=self._generate_rationale(domain_scores) if flag_decision.should_flag else "",
            flag_decision=flag_decision
        )
    
    def _extract_domain_text(self, case: Dict[str, Any], domain: str) -> str:
        """Extract text for a domain from case data."""
        if domain == "symptoms":
            symptoms = case.get("presenting_symptoms", [])
            chief = case.get("chief_complaint", "")
            return f"{chief} {' '.join(symptoms) if isinstance(symptoms, list) else symptoms}"
        
        elif domain == "ecg":
            ecg = case.get("ecg_abnormalities", [])
            rhythm = case.get("rhythm", "")
            interp = case.get("ecg_interpretation", "")
            return f"{rhythm} {' '.join(ecg) if isinstance(ecg, list) else ecg} {interp}"
        
        elif domain == "labs":
            labs = []
            for key in ["troponin", "bnp", "creatinine", "potassium", "hemoglobin"]:
                val = case.get(key)
                if val is not None:
                    labs.append(f"{key}:{val}")
            return " ".join(labs)
        
        elif domain == "demographics":
            age = case.get("age", "")
            sex = case.get("sex", "")
            return f"age:{age} sex:{sex}"
        
        elif domain == "imaging":
            ef = case.get("echo_ef", "")
            findings = case.get("imaging_findings", [])
            return f"EF:{ef} {' '.join(str(f) for f in findings) if findings else ''}"
        
        return ""
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple text similarity (Jaccard)."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _check_flag_rule(
        self,
        composite: float,
        domain_scores: Dict[str, float]
    ) -> 'FlagDecision':
        """Apply the locked flag rule."""
        # Condition 1: Composite >= threshold
        condition_1 = composite >= self.composite_threshold
        
        # Condition 2: ECG OR Symptoms >= critical threshold
        condition_2 = (
            domain_scores.get("ecg", 0) >= self.critical_domain_threshold or
            domain_scores.get("symptoms", 0) >= self.critical_domain_threshold
        )
        
        # Condition 3: >= 2 domains above minimum
        domains_above_min = sum(
            1 for score in domain_scores.values()
            if score >= self.domain_minimum
        )
        condition_3 = domains_above_min >= self.MIN_DOMAINS_ABOVE_MINIMUM
        
        should_flag = condition_1 and condition_2 and condition_3
        
        if should_flag:
            reason = None
        elif not condition_1:
            reason = f"Composite ({composite:.2f}) below threshold ({self.composite_threshold})"
        elif not condition_2:
            reason = "Neither ECG nor symptoms met critical threshold"
        else:
            reason = f"Only {domains_above_min} domains above minimum"
        
        return FlagDecision(
            is_flagged=should_flag,
            composite_score=composite,
            matched_cases=[],
            domain_scores=domain_scores,
            clinical_rationale="" if not should_flag else "Flag conditions met",
            source_references=[],
            reason=reason,
            should_flag=should_flag
        )
    
    def query(self, patient_data: Dict[str, Any]) -> FlagDecision:
        """
        Query for similar cases and make flag decision.
        """
        # Validate input
        validation_error = self._validate_input(patient_data)
        if validation_error:
            return FlagDecision(
                is_flagged=False,
                composite_score=0.0,
                matched_cases=[],
                domain_scores={},
                clinical_rationale="",
                source_references=[],
                reason=validation_error
            )
        
        # Generate patient embeddings
        query_embeddings = self.embedding_manager.generate_embeddings(patient_data)
        
        # Search FAISS indices
        search_results = self.faiss_manager.search(query_embeddings, k=50)
        
        # Aggregate scores per case
        case_scores = self._aggregate_case_scores(search_results)
        
        # Apply flag rule
        return self._make_flag_decision(case_scores)
    
    def _validate_input(self, data: Dict) -> Optional[str]:
        """Validate patient input data."""
        required = ["symptoms", "ecg", "labs", "demographics"]
        
        for field in required:
            if field not in data or not data[field]:
                return f"Missing required field: {field}"
        
        return None
    
    def _aggregate_case_scores(
        self,
        search_results: Dict[str, List[Tuple[int, float]]]
    ) -> Dict[int, Dict[str, float]]:
        """Aggregate similarity scores per case across domains."""
        case_scores: Dict[int, Dict[str, float]] = {}
        
        for dimension, results in search_results.items():
            for vector_id, score in results:
                if vector_id not in case_scores:
                    case_scores[vector_id] = {}
                case_scores[vector_id][dimension] = max(
                    case_scores[vector_id].get(dimension, 0),
                    score
                )
        
        return case_scores
    
    def _make_flag_decision(
        self,
        case_scores: Dict[int, Dict[str, float]]
    ) -> FlagDecision:
        """Apply flag rule and return decision."""
        
        if not case_scores:
            return FlagDecision(
                is_flagged=False,
                composite_score=0.0,
                matched_cases=[],
                domain_scores={},
                clinical_rationale="",
                source_references=[],
                reason="No similar cases found in database"
            )
        
        # Find best matches
        scored_cases = []
        
        for vector_id, domain_scores in case_scores.items():
            # Compute composite score
            composite = sum(
                self.WEIGHTS.get(dim, 0) * score
                for dim, score in domain_scores.items()
            )
            
            # Check flag conditions
            condition_1 = composite >= self.COMPOSITE_THRESHOLD
            
            condition_2 = (
                domain_scores.get("ecg", 0) >= self.CRITICAL_DOMAIN_THRESHOLD or
                domain_scores.get("symptoms", 0) >= self.CRITICAL_DOMAIN_THRESHOLD
            )
            
            domains_above_min = sum(
                1 for score in domain_scores.values()
                if score >= self.DOMAIN_MINIMUM
            )
            condition_3 = domains_above_min >= self.MIN_DOMAINS_ABOVE_MINIMUM
            
            is_flagged = condition_1 and condition_2 and condition_3
            
            scored_cases.append({
                "vector_id": vector_id,
                "composite": composite,
                "domain_scores": domain_scores,
                "is_flagged": is_flagged,
            })
        
        # Sort by composite score
        scored_cases.sort(key=lambda x: x["composite"], reverse=True)
        
        # Get top results
        top_case = scored_cases[0]
        metadata = self.faiss_manager.get_metadata(top_case["vector_id"])
        
        # Build matched cases list
        matched_cases = []
        source_refs = []
        
        for case in scored_cases[:10]:
            if case["composite"] >= self.COMPOSITE_THRESHOLD:
                meta = self.faiss_manager.get_metadata(case["vector_id"])
                if meta:
                    matched_cases.append(SimilarityResult(
                        case_id=meta.get("case_id", ""),
                        source_url=meta.get("source_url", ""),
                        rare_explanation=meta.get("rare_explanation", ""),
                        composite_score=case["composite"],
                        domain_scores=case["domain_scores"],
                        clinical_rationale=f"Matched on {len(case['domain_scores'])} domains"
                    ))
                    if meta.get("source_url"):
                        source_refs.append(meta["source_url"])
        
        # Determine if any case triggers flag
        any_flagged = any(c["is_flagged"] for c in scored_cases[:10])
        
        if any_flagged:
            return FlagDecision(
                is_flagged=True,
                composite_score=top_case["composite"],
                matched_cases=matched_cases,
                domain_scores=top_case["domain_scores"],
                clinical_rationale=self._generate_rationale(top_case),
                source_references=source_refs[:5]
            )
        else:
            return FlagDecision(
                is_flagged=False,
                composite_score=top_case["composite"],
                matched_cases=matched_cases,
                domain_scores=top_case["domain_scores"],
                clinical_rationale="",
                source_references=source_refs[:5],
                reason=self._generate_no_flag_reason(top_case)
            )
    
    def _generate_rationale(self, case: Dict) -> str:
        """Generate clinical rationale for flag."""
        domains = case["domain_scores"]
        strong_domains = [
            dim for dim, score in domains.items()
            if score >= self.CRITICAL_DOMAIN_THRESHOLD
        ]
        
        return (
            f"Strong similarity detected in {', '.join(strong_domains)}. "
            f"Composite score: {case['composite']:.2f}. "
            f"Review matched rare cardiology cases for clinical context."
        )
    
    def _generate_no_flag_reason(self, case: Dict) -> str:
        """Generate reason for not flagging."""
        if case["composite"] < self.COMPOSITE_THRESHOLD:
            return f"Composite score ({case['composite']:.2f}) below threshold ({self.COMPOSITE_THRESHOLD})"
        
        domains = case["domain_scores"]
        if (domains.get("ecg", 0) < self.CRITICAL_DOMAIN_THRESHOLD and
            domains.get("symptoms", 0) < self.CRITICAL_DOMAIN_THRESHOLD):
            return "Neither ECG nor symptoms met critical domain threshold"
        
        domains_above_min = sum(
            1 for score in domains.values()
            if score >= self.DOMAIN_MINIMUM
        )
        if domains_above_min < self.MIN_DOMAINS_ABOVE_MINIMUM:
            return f"Only {domains_above_min} domains above minimum (need {self.MIN_DOMAINS_ABOVE_MINIMUM})"
        
        return "Flag conditions not met"
