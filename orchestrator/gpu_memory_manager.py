"""
GPU Memory Manager Module

Provides utilities for managing GPU memory across the Synapse pipeline.
Ensures clean GPU state between model loads and provides diagnostic logging.
"""

import gc
import platform
from typing import Optional, Dict


# Try to import torch - optional for systems without GPU
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class GPUMemoryError(Exception):
    """Raised when insufficient GPU memory is available."""
    pass


def is_cuda_available() -> bool:
    """Check if CUDA GPU is available."""
    return TORCH_AVAILABLE and torch.cuda.is_available()


def get_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory state.

    Returns:
        Dictionary with memory info in GB:
        - allocated: Currently allocated memory
        - reserved: Total reserved by CUDA
        - free: Estimated free memory
        - total: Total GPU memory
    """
    if not is_cuda_available():
        return {
            "allocated": 0.0,
            "reserved": 0.0,
            "free": 0.0,
            "total": 0.0,
        }

    device = torch.cuda.current_device()
    allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3

    # Estimated free = reserved is what we've asked for, but can still allocate more
    # Use a more accurate estimate:
    try:
        free_gb = total_gb - reserved_gb
    except:
        free_gb = 0.0

    return {
        "allocated": round(allocated_gb, 2),
        "reserved": round(reserved_gb, 2),
        "free": round(free_gb, 2),
        "total": round(total_gb, 2),
    }


def clear_gpu_memory() -> None:
    """
    Force GPU memory cleanup AND RAM cleanup.

    This does:
    1. Multiple Python garbage collection passes (aggressive RAM cleanup)
    2. Empty CUDA cache (release unused GPU memory blocks)
    3. Reset peak memory stats (for accurate tracking)
    4. Synchronize CUDA to ensure operations complete
    5. Clear module caches that might hold large objects
    6. Log both GPU and RAM usage

    Call this before loading a new heavy model.
    """
    if not is_cuda_available():
        print("[Memory Manager] CUDA not available - no GPU memory to clear")
        # Still run RAM cleanup even without GPU
        _force_ram_cleanup()
        return

    print("[Memory Manager] Clearing GPU and RAM memory...")
    print("[Memory Manager] RAM usage before cleanup:")

    try:
        # Get RAM usage before cleanup
        ram_before = _get_ram_usage()
        print(f"[Memory Manager]  RAM used: {ram_before:.1f} GB")

        # STEP 1: Force Python GC multiple times (aggressive)
        # This releases Python objects, cycles, and weak references
        print("[Memory Manager] Step 1: Aggressive Python GC...")
        for _ in range(3):
            gc.collect()

        # STEP 2: Clear common module caches that might hold large objects
        print("[Memory Manager] Step 2: Clearing module caches...")
        _clear_module_caches()

        # STEP 3: Empty CUDA cache - this is the key operation
        print("[Memory Manager] Step 3: Clearing CUDA cache...")
        torch.cuda.empty_cache()

        # STEP 4: Reset peak memory stats for accurate tracking
        torch.cuda.reset_peak_memory_stats()

        # STEP 5: Synchronize to ensure all operations complete
        torch.cuda.synchronize()

        # STEP 6: One final GC pass
        print("[Memory Manager] Step 4: Final GC pass...")
        gc.collect()

        # Log memory state after cleanup
        ram_after = _get_ram_usage()
        ram_freed = ram_before - ram_after

        print(f"[Memory Manager] Memory cleanup complete:")
        print(f"  RAM freed: {ram_freed:.2f} GB")
        print(f"  RAM used after: {ram_after:.1f} GB")

        if is_cuda_available():
            mem_info = get_memory_info()
            print(f"  GPU memory after cleanup:")
            print(f"    Allocated: {mem_info['allocated']} GB")
            print(f"    Reserved: {mem_info['reserved']} GB")
            print(f"    Free: {mem_info['free']} GB")
            print(f"    Total: {mem_info['total']} GB")

    except Exception as e:
        print(f"[Memory Manager] Warning during memory cleanup: {e}")


def _get_ram_usage() -> float:
    """
    Get current RAM usage in GB.

    Returns:
        RAM usage in GB
    """
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)  # Resident Set Size
    except ImportError:
        # Fallback: estimate using os-specific methods
        try:
            import os
            if platform.system() == 'Windows':
                # Windows
                import ctypes
                kernel32 = ctypes.windll.kernel32
                GetCurrentProcess = kernel32.GetCurrentProcess
                process_handle = GetCurrentProcess()
                counters = {}
                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ('WorkingSetSize', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t),
                    ]
                # This is a rough estimate
                return 0.0
            else:
                # Linux/Mac
                import resource
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)
        except:
            return 0.0
    except:
        return 0.0


def _clear_module_caches() -> None:
    """
    Clear caches in common modules that might hold large objects.

    This frees RAM from:
    - PIL images (image processing)
    - Numpy arrays
    - Transformes cache
    - HTTP connection pools
    """
    try:
        # Clear PIL image cache if available
        try:
            from PIL import Image
            Image.Image._PIL__ = None
        except:
            pass

        # Clear numpy cache if possible
        try:
            import numpy
            if hasattr(numpy, 'clear_cache'):
                numpy.clear_cache()
        except:
            pass

        # Clear transformers cache
        try:
            from transformers import cache_utils
            cache_utils.clear_transformers_cache()
        except:
            pass

    except Exception as e:
        print(f"[Memory Manager] Warning while clearing module caches: {e}")


def _force_ram_cleanup() -> None:
    """
    Force RAM cleanup without GPU (for CPU-only systems).
    """
    print("[Memory Manager] RAM-only cleanup mode...")
    print("[Memory Manager] RAM usage before cleanup:")

    ram_before = _get_ram_usage()
    print(f"[Memory Manager]  RAM used: {ram_before:.1f} GB")

    try:
        # Multiple GC passes
        for _ in range(3):
            gc.collect()

        # Clear module caches
        _clear_module_caches()

        ram_after = _get_ram_usage()
        ram_freed = ram_before - ram_after

        print(f"[Memory Manager] RAM cleanup complete:")
        print(f"  RAM freed: {ram_freed:.2f} GB")
        print(f"  RAM used after: {ram_after:.1f} GB")

    except Exception as e:
        print(f"[Memory Manager] Warning during RAM cleanup: {e}")


def ensure_gpu_free(min_required_gb: float) -> bool:
    """
    Check if enough GPU memory is available for a model.

    Args:
        min_required_gb: Minimum GPU memory required in GB

    Returns:
        True if enough memory is available

    Raises:
        GPUMemoryError: If insufficient memory
    """
    if not is_cuda_available():
        print(f"[Memory Manager] CUDA not available - cannot ensure {min_required_gb} GB")
        return False

    mem_info = get_memory_info()

    if mem_info["free"] < min_required_gb:
        print(f"[Memory Manager] Insufficient GPU memory!")
        print(f"  Required: {min_required_gb} GB")
        print(f"  Available: {mem_info['free']} GB")
        print(f"  Shortage: {min_required_gb - mem_info['free']:.2f} GB")

        # Try clearing cache and recheck
        print("[Memory Manager] Attempting to clear cache and recheck...")
        clear_gpu_memory()
        mem_info = get_memory_info()

        if mem_info["free"] < min_required_gb:
            raise GPUMemoryError(
                f"Insufficient GPU memory: need {min_required_gb} GB, "
                f"have {mem_info['free']} GB free"
            )
        else:
            print(f"[Memory Manager] After cleanup: {mem_info['free']} GB free - OK")

    print(f"[Memory Manager] GPU memory check: {mem_info['free']} GB available >= {min_required_gb} GB required - OK")
    return True


def log_memory_state(stage_name: str) -> None:
    """
    Log GPU memory state for a pipeline stage.

    Useful for debugging and tracking memory usage.

    Args:
        stage_name: Name of the current stage (e.g., "Before VLM Load")
    """
    if not is_cuda_available():
        print(f"[Memory Manager] Stage: {stage_name} - CUDA not available")
        return

    mem_info = get_memory_info()
    device_name = torch.cuda.get_device_name(0)

    print(f"\n{'='*60}")
    print(f"[Memory Manager] Stage: {stage_name}")
    print(f"  Device: {device_name}")
    print(f"  Allocated: {mem_info['allocated']} GB / {mem_info['total']} GB ({mem_info['allocated']/mem_info['total']*100:.1f}%)")
    print(f"  Reserved: {mem_info['reserved']} GB")
    print(f"  Free: {mem_info['free']} GB")
    print(f"{'='*60}\n")


def get_system_info() -> Dict[str, str]:
    """
    Get system information for diagnostics.

    Returns:
        Dictionary with system info:
        - platform: OS name
        - cuda_available: Whether CUDA is available
        - cuda_version: CUDA version if available
        - torch_version: PyTorch version
        - gpu_name: GPU device name if available
    """
    info = {
        "platform": platform.system(),
        "cuda_available": str(is_cuda_available()),
    }

    if TORCH_AVAILABLE:
        info["torch_version"] = torch.__version__

        if is_cuda_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory"] = f"{props.total_memory / 1024**3:.2f} GB"
        else:
            info["gpu_name"] = "N/A (CUDA not available)"
            info["gpu_memory"] = "N/A"
    else:
        info["torch_version"] = "Not installed"

    return info


def print_system_info() -> None:
    """Print system information for diagnostics."""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)

    info = get_system_info()
    for key, value in info.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

    print("="*60 + "\n")


# Convenience functions for common pipeline stages

def before_whisper():
    """Setup before Whisper transcription."""
    log_memory_state("Before Whisper Load")


def after_whisper():
    """Cleanup after Whisper transcription."""
    log_memory_state("After Whisper Complete")
    print("[Memory Manager] Whisper completed - clearing GPU memory...")
    clear_gpu_memory()


def before_vlm_load():
    """Setup before VLM model loads - critical step."""
    log_memory_state("Before VLM Load")
    print("[Memory Manager] CRITICAL: Clearing GPU before VLM load...")
    clear_gpu_memory()


def after_vlm_load():
    """Log after VLM loaded."""
    log_memory_state("After VLM Loaded")


def before_tts():
    """Setup before TTS generation."""
    log_memory_state("Before TTS Generation")


def after_tts():
    """Cleanup after TTS completes."""
    log_memory_state("After TTS Complete")


if __name__ == "__main__":
    # Test the module
    print_system_info()

    print("\nTesting memory management functions...\n")

    # Test memory info
    print("1. Getting memory info:")
    mem_info = get_memory_info()
    for k, v in mem_info.items():
        print(f"   {k}: {v}")

    # Test clearing
    print("\n2. Testing GPU cleanup:")
    clear_gpu_memory()

    # Test logging
    print("\n3. Testing stage logging:")
    log_memory_state("Test Stage")

    # Test availability check
    print("\n4. Testing memory availability check (1 GB):")
    try:
        ensure_gpu_free(1.0)
        print("   Check passed!")
    except GPUMemoryError as e:
        print(f"   Check failed: {e}")
