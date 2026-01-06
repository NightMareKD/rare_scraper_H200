# Mode Detection System

## Overview

This document explains how the Clinical Case Similarity System detects available hardware and switches between CPU-only and GPU-enabled execution modes.

---

## Hardware Detection Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    STARTUP                                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Check for CUDA-capable GPU                      │
│              torch.cuda.is_available()                       │
└─────────────────────────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
        ┌─────────┐             ┌─────────┐
        │   YES   │             │   NO    │
        └─────────┘             └─────────┘
              │                       │
              ▼                       ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Check GPU Quota    │     │  CPU_ONLY Mode      │
│  (gpu_usage.log)    │     │                     │
└─────────────────────┘     └─────────────────────┘
              │
    ┌─────────┴─────────┐
    │                   │
    ▼                   ▼
┌─────────┐       ┌─────────────┐
│ Quota   │       │   Quota     │
│ OK      │       │   Exceeded  │
└─────────┘       └─────────────┘
    │                   │
    ▼                   ▼
┌─────────────┐   ┌─────────────┐
│ GPU_AVAIL-  │   │ GPU_LIMITED │
│ ABLE Mode   │   │ Mode        │
└─────────────┘   └─────────────┘
```

---

## Execution Modes

### 1. CPU_ONLY Mode

**Triggered when:**
- No CUDA-capable GPU is detected
- PyTorch is not installed with CUDA support
- Running on CPU-only infrastructure

**Allowed operations:**
- Data scraping and ingestion
- Text normalization
- Data validation
- FAISS index querying (CPU implementation)
- Checkpoint management
- Logging and auditing

**Deferred operations:**
- Embedding generation
- LLM distillation
- FAISS index building (if GPU-optimized)

### 2. GPU_AVAILABLE Mode

**Triggered when:**
- CUDA GPU is detected
- Daily GPU quota has not been exceeded
- GPU memory is sufficient

**Allowed operations:**
- All CPU operations
- Embedding generation with GPU acceleration
- LLM inference for case distillation
- GPU-accelerated FAISS indexing

**Constraints:**
- Subject to `gpu_policy.yaml` limits
- Session duration limits apply
- Batch sizes from configuration

### 3. GPU_LIMITED Mode

**Triggered when:**
- GPU is physically available
- Daily quota has been exceeded (>= max_gpu_minutes_per_day)

**Behavior:**
- Operates like CPU_ONLY mode
- GPU operations are deferred to next day
- Warning logged for visibility

---

## Detection Code

The mode detection is handled by the `HardwareDetector` class in `app.py`:

```python
class HardwareDetector:
    @staticmethod
    def detect_gpu() -> Tuple[bool, Optional[str]]:
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                return True, device_name
        except ImportError:
            pass
        return False, None
    
    @staticmethod
    def get_execution_mode(config: Dict[str, Any]) -> ExecutionMode:
        gpu_available, _ = HardwareDetector.detect_gpu()
        
        if not gpu_available:
            return ExecutionMode.CPU_ONLY
        
        # Check quota from logs
        daily_usage = get_daily_gpu_usage()
        max_minutes = config.get('max_gpu_minutes_per_day', 60)
        
        if daily_usage >= max_minutes:
            return ExecutionMode.GPU_LIMITED
        
        return ExecutionMode.GPU_AVAILABLE
```

---

## Hugging Face Spaces Detection

The system also detects if it's running in a Hugging Face Space:

```python
is_hf_space = os.environ.get("SPACE_ID") is not None
```

When running in HF Spaces:
1. Gradio interface is launched
2. Pipeline runs in background thread
3. Zero GPU mode is respected
4. Persistent storage paths are configured

---

## Mode Switching During Execution

The system can switch modes during execution:

| Event | Current Mode | New Mode | Action |
|-------|-------------|----------|--------|
| Quota exceeded | GPU_AVAILABLE | GPU_LIMITED | Save checkpoint, defer GPU ops |
| Memory pressure | GPU_AVAILABLE | CPU_ONLY | Clear cache, fallback to CPU |
| Temperature high | GPU_AVAILABLE | GPU_LIMITED | Cooling period |
| New day | GPU_LIMITED | GPU_AVAILABLE | Resume GPU operations |

---

## Logging

Mode changes are logged to `logs/ingestion.log`:

```
2026-01-05 02:00:00 - INFO - GPU detected: NVIDIA T4 (15.0GB)
2026-01-05 02:00:00 - INFO - Execution mode: gpu
2026-01-05 04:30:00 - WARNING - GPU quota exhausted: 60.5/60 minutes
2026-01-05 04:30:00 - INFO - Switching to GPU_LIMITED mode
```

---

## Troubleshooting

### GPU not detected when it should be

1. Check PyTorch CUDA installation:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.version.cuda)
   ```

2. Verify CUDA drivers:
   ```bash
   nvidia-smi
   ```

3. Check environment variables:
   ```bash
   echo $CUDA_VISIBLE_DEVICES
   ```

### Quota seems wrong

1. Check `logs/gpu_usage.log` format
2. Verify timezone settings (uses UTC)
3. Check for date parsing errors

### Mode not switching correctly

1. Review `get_execution_mode()` logic
2. Check config loading
3. Verify checkpoint manager state

---

## Future Improvements

1. **Multi-GPU support**: Detect and use multiple GPUs
2. **Dynamic batch sizing**: Adjust based on available memory
3. **Predictive quota management**: Forecast usage patterns
4. **Graceful degradation**: Partial GPU operations when limited
