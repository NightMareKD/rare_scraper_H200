# Regulatory Notes

## Overview

This document provides guidance on the regulatory considerations for the Clinical Case Similarity System. While this system is designed for research and educational purposes, this documentation ensures alignment with ethical AI principles and medical decision support norms.

---

## Classification

### What This System IS

- **Clinical Decision Support Tool**: Provides information to assist clinician decision-making
- **Research Tool**: Facilitates pattern identification in medical literature
- **Educational Resource**: Helps learners understand similar case presentations

### What This System IS NOT

- **Medical Device**: Not intended to diagnose, treat, or prevent disease
- **Autonomous Decision Maker**: Does not make clinical decisions independently
- **Patient Data Processor**: Does not process individual patient health records

---

## Regulatory Framework Considerations

### United States (FDA)

The FDA's Clinical Decision Support (CDS) guidance provides exemptions for certain software functions. This system is designed to fall within exempted categories:

**Exemption Criteria (21st Century Cures Act, Section 3060)**:

✅ **Criterion 1**: Not intended to acquire, process, or analyze a medical image or signal  
*This system analyzes text from published literature, not patient-generated data*

✅ **Criterion 2**: Intended for displaying, analyzing, or printing medical information  
*This system displays similarity matches from published case reports*

✅ **Criterion 3**: Intended for supporting or providing recommendations  
*This system supports clinical decisions, does not make autonomous recommendations*

✅ **Criterion 4**: Intended for the user to independently review the basis for recommendations  
*All matches include source citations and similarity breakdowns for review*

### European Union (EU MDR)

Under EU Medical Device Regulation (MDR 2017/745):

**Non-Device Arguments**:
- Software that provides general clinical information is not a medical device
- Decision support that requires independent clinician judgment is exempted
- Literature-based reference tools are typically not regulated as devices

**Documentation Maintained**:
- Intended purpose statement (this document)
- Risk analysis (see below)
- Clinical evaluation (via validation strategy)

### Other Jurisdictions

The same principles apply:
- Clear intended purpose as decision SUPPORT
- Clinician in the loop for all decisions
- Transparency about limitations
- No patient-specific diagnostic claims

---

## Risk Analysis

### Hazard Identification

| Hazard | Likelihood | Severity | Risk Level | Mitigation |
|--------|------------|----------|------------|------------|
| False positive match leads to incorrect consideration | Medium | Low | Low | Clinical review requirement, conservative thresholds |
| False negative misses important similar case | Low | Medium | Low | High recall design, multiple domain matching |
| Outdated case information | Medium | Low | Low | Source date tracking, disclaimers |
| System unavailability | Low | Low | Very Low | Not critical for patient care |
| Incorrect similarity score calculation | Low | Low | Very Low | Validation testing, audit logs |

### Risk Controls

1. **Clinical Disclaimers**: Prominently displayed, required acknowledgment
2. **Human-in-the-Loop**: All output requires clinician review
3. **Audit Trail**: Complete logging of all matches
4. **Validation Program**: Ongoing accuracy monitoring
5. **Threshold Calibration**: Conservative settings to minimize false positives

### Residual Risk Assessment

After controls, residual risk is **ACCEPTABLE** because:
- System does not make autonomous clinical decisions
- All output is reviewed by qualified clinicians
- Worst-case scenario (incorrect match) has low clinical impact
- Benefits (rare disease identification) outweigh residual risks

---

## Ethical AI Principles

### Transparency

- **Explainability**: All similarity scores include breakdowns and justifications
- **Source Attribution**: Original sources cited for all cases
- **Limitation Disclosure**: Clear documentation of system limitations

### Fairness

- **Demographic Awareness**: Demographics weighted appropriately (10%)
- **Bias Monitoring**: Regular review for demographic bias in matching
- **Inclusive Data**: Sources from multiple geographic regions

### Privacy

- **No Patient Data**: System uses only published, de-identified case reports
- **Open Access Only**: Only publicly available literature is scraped
- **No Re-identification**: System does not attempt to identify patients

### Accountability

- **Audit Logs**: Complete record of all operations
- **Version Control**: All configurations tracked and versioned
- **Clear Ownership**: Defined responsibility for system operation

### Safety

- **Conservative Thresholds**: Designed to minimize false positives
- **Clinical Review**: Mandatory human oversight
- **Continuous Monitoring**: Ongoing validation and improvement

---

## Data Governance

### Data Sources

| Source | Type | License | Compliance |
|--------|------|---------|------------|
| PubMed Central | API | Open Access | NIH Terms of Service |
| ESC Cases | Web | Open Access | Research/Education Use |
| AHA Journals | Web | Open Access | Publisher Terms |
| Open Access Journals | Web | CC/Open | Per-journal license |

### Data Processing

1. **Collection**: Only open access, publicly available content
2. **Storage**: Secure, access-controlled persistent storage
3. **Retention**: Indefinite for traceability
4. **Deletion**: Upon request or license change

### GDPR Considerations

While the system primarily uses published literature (not personal data):

- No individual patient data is processed
- Published case reports are not considered personal data
- Author information (if stored) follows legitimate interest basis
- Right to be forgotten: Can remove specific sources if required

---

## Quality Management

### Applicable Standards

While not required for research tools, the system aligns with:

- **ISO 13485**: Quality management for medical devices (principles only)
- **IEC 62304**: Software lifecycle for medical devices (principles only)
- **ISO 14971**: Risk management for medical devices (principles only)

### Documentation Maintained

- System requirements specification
- Software architecture documentation
- Validation test records
- Risk analysis records
- Change control logs
- Incident reports (if any)

### Change Control

All changes to:
- Thresholds
- Weights
- Clinical schema
- Data sources

Require:
1. Documentation of change
2. Impact assessment
3. Validation testing
4. Review and approval

---

## Post-Market Considerations

### Surveillance Activities

- Monitor user feedback
- Track false positive/negative reports
- Review clinical literature for relevant updates
- Annual validation review

### Incident Handling

If a potential harm is identified:

1. **Immediate**: Assess severity
2. **Investigation**: Root cause analysis
3. **Corrective Action**: Implement fix
4. **Notification**: Report to stakeholders
5. **Documentation**: Record in incident log

### Continuous Improvement

- Quarterly threshold review
- Monthly validation testing
- Continuous clinician feedback integration
- Annual comprehensive review

---

## Compliance Checklist

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Intended purpose documented | ✅ | This document |
| Clinical disclaimers in place | ✅ | UI, README |
| Risk analysis completed | ✅ | This document |
| Validation strategy defined | ✅ | validation_strategy.md |
| Explainability documented | ✅ | explainability.md |
| Audit logging implemented | ✅ | app.py, logs/audit.log |
| Open access compliance | ✅ | sources.yaml |
| Human-in-the-loop required | ✅ | UI design, disclaimers |

---

## Contact Information

For regulatory questions:
- Review system documentation
- Consult institutional compliance office
- Open issue in project repository

---

*This document is for informational purposes and does not constitute legal or regulatory advice. Consult qualified regulatory professionals for specific compliance questions.*

*Last updated: January 5, 2026*
