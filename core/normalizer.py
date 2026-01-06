"""
Normalizer Module for Clinical Case Processing
===============================================
Converts raw scraped content into structured clinical schema.
"""

import re
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


@dataclass
class NormalizedCase:
    """Represents a normalized clinical case."""
    case_id: str
    source: str
    url: str
    
    # Demographics
    age: Optional[int] = None
    age_unit: str = "years"  # years, months, days
    sex: Optional[str] = None  # M, F, Unknown
    ethnicity: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    bmi: Optional[float] = None
    
    # Symptoms
    presenting_symptoms: List[str] = field(default_factory=list)
    symptom_duration: Optional[str] = None
    chief_complaint: Optional[str] = None
    
    # Vital Signs
    heart_rate: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    respiratory_rate: Optional[int] = None
    temperature_c: Optional[float] = None
    oxygen_saturation: Optional[float] = None
    
    # ECG
    rhythm: Optional[str] = None
    ecg_intervals: Dict[str, Any] = field(default_factory=dict)
    ecg_abnormalities: List[str] = field(default_factory=list)
    ecg_interpretation: Optional[str] = None
    
    # Labs
    troponin: Optional[float] = None
    troponin_unit: str = "ng/mL"
    bnp: Optional[float] = None
    bnp_unit: str = "pg/mL"
    creatinine: Optional[float] = None
    hemoglobin: Optional[float] = None
    potassium: Optional[float] = None
    sodium: Optional[float] = None
    cholesterol_total: Optional[float] = None
    ldl: Optional[float] = None
    hdl: Optional[float] = None
    glucose: Optional[float] = None
    hba1c: Optional[float] = None
    
    # Imaging
    imaging_findings: List[Dict[str, str]] = field(default_factory=list)
    echo_ef: Optional[float] = None
    echo_findings: List[str] = field(default_factory=list)
    
    # Diagnosis
    primary_diagnosis: Optional[str] = None
    secondary_diagnoses: List[str] = field(default_factory=list)
    icd_codes: List[str] = field(default_factory=list)
    
    # Treatment
    medications: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    interventions: List[str] = field(default_factory=list)
    
    # Outcome
    outcome: Optional[str] = None
    follow_up_duration: Optional[str] = None
    complications: List[str] = field(default_factory=list)
    
    # Metadata
    publication_date: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    raw_text: Optional[str] = None
    normalized_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class TextExtractor:
    """Extracts clinical information from unstructured text."""
    
    # Age patterns
    AGE_PATTERNS = [
        r'(\d{1,3})[\s-]*(year|yr|y)[\s-]*(old)?',
        r'aged?\s*(\d{1,3})',
        r'(\d{1,3})\s*(months?|mo)\s*(old)?',
    ]
    
    # Sex patterns
    SEX_PATTERNS = [
        r'\b(male|female|man|woman|boy|girl)\b',
        r'\b([MF])\s*/?\s*(\d{1,3})',
    ]
    
    # Symptom patterns
    SYMPTOM_KEYWORDS = [
        "chest pain", "dyspnea", "shortness of breath", "palpitations",
        "syncope", "dizziness", "fatigue", "edema", "orthopnea",
        "paroxysmal nocturnal dyspnea", "claudication", "angina",
        "nausea", "vomiting", "diaphoresis", "presyncope",
    ]
    
    # ECG rhythm patterns
    RHYTHM_PATTERNS = [
        r'(sinus rhythm|atrial fibrillation|atrial flutter|'
        r'ventricular tachycardia|ventricular fibrillation|'
        r'supraventricular tachycardia|heart block|bradycardia|'
        r'tachycardia|asystole|pulseless electrical activity)',
    ]
    
    # Lab value patterns
    LAB_PATTERNS = {
        "troponin": r'troponin[:\s]+(\d+\.?\d*)\s*(ng/mL|ng/L|µg/L)?',
        "bnp": r'(bnp|nt-?probnp)[:\s]+(\d+\.?\d*)\s*(pg/mL)?',
        "creatinine": r'creatinine[:\s]+(\d+\.?\d*)\s*(mg/dL|µmol/L)?',
        "potassium": r'potassium[:\s]+(\d+\.?\d*)\s*(mEq/L|mmol/L)?',
        "hemoglobin": r'(hemoglobin|hb|hgb)[:\s]+(\d+\.?\d*)\s*(g/dL)?',
    }
    
    # Vital sign patterns
    VITAL_PATTERNS = {
        "heart_rate": r'(heart rate|hr|pulse)[:\s]+(\d{2,3})\s*(bpm)?',
        "bp": r'(blood pressure|bp)[:\s]+(\d{2,3})\s*/\s*(\d{2,3})',
        "respiratory_rate": r'(respiratory rate|rr)[:\s]+(\d{1,2})',
        "temperature": r'(temperature|temp)[:\s]+(\d{2}\.?\d?)\s*(°?[CF])?',
        "oxygen_saturation": r'(spo2|oxygen saturation|o2 sat)[:\s]+(\d{2,3})\s*%?',
    }
    
    # Echo patterns
    ECHO_PATTERNS = {
        "ef": r'(ejection fraction|ef|lvef)[:\s]+(\d{1,2})\s*%?',
    }
    
    def extract_age(self, text: str) -> Tuple[Optional[int], str]:
        """Extract age from text."""
        text_lower = text.lower()
        
        for pattern in self.AGE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                age = int(match.group(1))
                unit = "years"
                if "month" in match.group(0) or "mo" in match.group(0):
                    unit = "months"
                return age, unit
        
        return None, "years"
    
    def extract_sex(self, text: str) -> Optional[str]:
        """Extract sex from text."""
        text_lower = text.lower()
        
        if re.search(r'\b(female|woman|girl)\b', text_lower):
            return "F"
        if re.search(r'\b(male|man|boy)\b', text_lower):
            return "M"
        
        match = re.search(r'\b([MF])\s*/\s*\d', text)
        if match:
            return match.group(1)
        
        return None
    
    def extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text."""
        text_lower = text.lower()
        found_symptoms = []
        
        for symptom in self.SYMPTOM_KEYWORDS:
            if symptom in text_lower:
                found_symptoms.append(symptom)
        
        return found_symptoms
    
    def extract_rhythm(self, text: str) -> Optional[str]:
        """Extract ECG rhythm from text."""
        text_lower = text.lower()
        
        for pattern in self.RHYTHM_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)
        
        return None
    
    def extract_ecg_abnormalities(self, text: str) -> List[str]:
        """Extract ECG abnormalities from text."""
        abnormalities = []
        text_lower = text.lower()
        
        patterns = [
            r'st[\s-]*(elevation|depression)',
            r't[\s-]*(wave|inversion)',
            r'q[\s-]*(wave)',
            r'(left|right)\s*(bundle branch block|bbb)',
            r'(prolonged|short)\s*(qt|pr|qrs)',
            r'lvh|rvh|lae|rae',
            r'axis deviation',
        ]
        
        for pattern in patterns:
            if re.search(pattern, text_lower):
                match = re.search(pattern, text_lower)
                abnormalities.append(match.group(0))
        
        return abnormalities
    
    def extract_vitals(self, text: str) -> Dict[str, Any]:
        """Extract vital signs from text."""
        vitals = {}
        text_lower = text.lower()
        
        # Heart rate
        match = re.search(self.VITAL_PATTERNS["heart_rate"], text_lower)
        if match:
            vitals["heart_rate"] = int(match.group(2))
        
        # Blood pressure
        match = re.search(self.VITAL_PATTERNS["bp"], text_lower)
        if match:
            vitals["bp_systolic"] = int(match.group(2))
            vitals["bp_diastolic"] = int(match.group(3))
        
        # Respiratory rate
        match = re.search(self.VITAL_PATTERNS["respiratory_rate"], text_lower)
        if match:
            vitals["respiratory_rate"] = int(match.group(2))
        
        # Temperature
        match = re.search(self.VITAL_PATTERNS["temperature"], text_lower)
        if match:
            vitals["temperature"] = float(match.group(2))
        
        # Oxygen saturation
        match = re.search(self.VITAL_PATTERNS["oxygen_saturation"], text_lower)
        if match:
            vitals["oxygen_saturation"] = float(match.group(2))
        
        return vitals
    
    def extract_labs(self, text: str) -> Dict[str, Any]:
        """Extract lab values from text."""
        labs = {}
        text_lower = text.lower()
        
        for lab_name, pattern in self.LAB_PATTERNS.items():
            match = re.search(pattern, text_lower)
            if match:
                # Handle different capture groups
                if lab_name == "bnp":
                    labs[lab_name] = float(match.group(2))
                elif lab_name == "hemoglobin":
                    labs[lab_name] = float(match.group(2))
                else:
                    labs[lab_name] = float(match.group(1))
        
        return labs
    
    def extract_echo_ef(self, text: str) -> Optional[float]:
        """Extract ejection fraction from text."""
        text_lower = text.lower()
        
        match = re.search(self.ECHO_PATTERNS["ef"], text_lower)
        if match:
            return float(match.group(2))
        
        return None


class CaseNormalizer:
    """
    Normalizes raw scraped cases into structured clinical schema.
    """
    
    def __init__(self, schema_config: Dict[str, Any]):
        self.schema_config = schema_config
        self.extractor = TextExtractor()
        self.output_dir = Path("data/structured")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def normalize(self, raw_case: Dict[str, Any]) -> NormalizedCase:
        """Normalize a raw case into structured format."""
        text = raw_case.get("content", "") or ""
        title = raw_case.get("title", "") or ""
        combined_text = f"{title} {text}"
        
        # Extract demographics
        age, age_unit = self.extractor.extract_age(combined_text)
        sex = self.extractor.extract_sex(combined_text)
        
        # Extract symptoms
        symptoms = self.extractor.extract_symptoms(combined_text)
        
        # Extract vitals
        vitals = self.extractor.extract_vitals(combined_text)
        
        # Extract ECG info
        rhythm = self.extractor.extract_rhythm(combined_text)
        ecg_abnormalities = self.extractor.extract_ecg_abnormalities(combined_text)
        
        # Extract labs
        labs = self.extractor.extract_labs(combined_text)
        
        # Extract echo
        echo_ef = self.extractor.extract_echo_ef(combined_text)
        
        # Create normalized case
        normalized = NormalizedCase(
            case_id=raw_case.get("case_id", ""),
            source=raw_case.get("source", ""),
            url=raw_case.get("url", ""),
            
            # Demographics
            age=age,
            age_unit=age_unit,
            sex=sex,
            
            # Symptoms
            presenting_symptoms=symptoms,
            chief_complaint=self._extract_chief_complaint(combined_text),
            
            # Vitals
            heart_rate=vitals.get("heart_rate"),
            blood_pressure_systolic=vitals.get("bp_systolic"),
            blood_pressure_diastolic=vitals.get("bp_diastolic"),
            respiratory_rate=vitals.get("respiratory_rate"),
            temperature_c=vitals.get("temperature"),
            oxygen_saturation=vitals.get("oxygen_saturation"),
            
            # ECG
            rhythm=rhythm,
            ecg_abnormalities=ecg_abnormalities,
            
            # Labs
            troponin=labs.get("troponin"),
            bnp=labs.get("bnp"),
            creatinine=labs.get("creatinine"),
            potassium=labs.get("potassium"),
            hemoglobin=labs.get("hemoglobin"),
            
            # Echo
            echo_ef=echo_ef,
            
            # Metadata
            publication_date=raw_case.get("publication_date"),
            journal=raw_case.get("journal"),
            doi=raw_case.get("doi"),
            raw_text=combined_text[:10000],  # Truncate for storage
            normalized_at=datetime.now().isoformat(),
        )
        
        return normalized
    
    def _extract_chief_complaint(self, text: str) -> Optional[str]:
        """Extract chief complaint from text."""
        patterns = [
            r'(chief complaint|presenting with|presented with)[:\s]+([^.]+)',
            r'(presented to|admitted for)[:\s]+([^.]+)',
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(2).strip()[:200]
        
        return None
    
    def save_normalized(self, case: NormalizedCase) -> None:
        """Save normalized case to disk."""
        output_file = self.output_dir / f"{case.case_id}_normalized.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(case.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved normalized case: {case.case_id}")
    
    def load_normalized(self, case_id: str) -> Optional[NormalizedCase]:
        """Load a normalized case from disk."""
        file_path = self.output_dir / f"{case_id}_normalized.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return NormalizedCase(**data)


class NormalizationPipeline:
    """
    Orchestrates the normalization of all raw cases.
    """
    
    def __init__(self, schema_config: Dict[str, Any], checkpoint_manager):
        self.normalizer = CaseNormalizer(schema_config)
        self.checkpoint_manager = checkpoint_manager
        self.raw_dir = Path("data/raw")
    
    def run(self) -> int:
        """
        Normalize all raw cases.
        Returns count of normalized cases.
        """
        normalized_count = 0
        processed_ids = self.checkpoint_manager.get_processed_ids("normalized_cases")
        
        # Find all raw case files
        raw_files = []
        for source_dir in self.raw_dir.iterdir():
            if source_dir.is_dir():
                raw_files.extend(source_dir.glob("*.json"))
        
        logger.info(f"Found {len(raw_files)} raw case files")
        
        for raw_file in raw_files:
            case_id = raw_file.stem
            
            if case_id in processed_ids:
                continue
            
            try:
                with open(raw_file, 'r', encoding='utf-8') as f:
                    raw_case = json.load(f)
                
                normalized = self.normalizer.normalize(raw_case)
                self.normalizer.save_normalized(normalized)
                
                self.checkpoint_manager.mark_processed("normalized_cases", case_id)
                normalized_count += 1
                
            except Exception as e:
                logger.error(f"Failed to normalize {case_id}: {e}")
                continue
        
        logger.info(f"Normalized {normalized_count} cases")
        return normalized_count
