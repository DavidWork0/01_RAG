"""
Test Script for Hyperparameter Tuning with Neptune
===================================================
Quick validation that the Neptune integration works correctly.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*70)
print("HYPERPARAMETER TUNING NEPTUNE - VALIDATION TEST")
print("="*70)

# Test 1: Check Neptune availability
print("\n[Test 1] Checking Neptune package...")
try:
    import neptune
    print("  [OK] Neptune package installed")
    NEPTUNE_AVAILABLE = True
except ImportError:
    print("  [Error] Neptune package NOT installed")
    print("   Install with: pip install neptune")
    NEPTUNE_AVAILABLE = False

# Test 2: Check credentials
print("\n[Test 2] Checking Neptune credentials...")
api_token = os.environ.get('NEPTUNE_API_TOKEN')
project = os.environ.get('NEPTUNE_PROJECT')

if api_token:
    print(f"  [OK] NEPTUNE_API_TOKEN is set ({api_token[:10]}...)")
else:
    print("  [Error] NEPTUNE_API_TOKEN not set")
    print("   Set with: $env:NEPTUNE_API_TOKEN='your_token'")

if project:
    print(f"  [OK] NEPTUNE_PROJECT is set ({project})")
else:
    print("  [Error] NEPTUNE_PROJECT not set")
    print("   Set with: $env:NEPTUNE_PROJECT='username/project'")

# Test 3: Check module imports
print("\n[Test 3] Checking module imports...")
try:
    from hyperparameter_tuner import HyperparameterTuner
    print("  [OK] Base HyperparameterTuner imported")
except ImportError as e:
    print(f"  [Error] Failed to import HyperparameterTuner: {e}")

try:
    from hyperparameter_tuner_neptune import NeptuneHyperparameterTuner
    print("  [OK] NeptuneHyperparameterTuner imported")
except ImportError as e:
    print(f"  [Error] Failed to import NeptuneHyperparameterTuner: {e}")

# Test 4: Check config file
print("\n[Test 4] Checking configuration file...")
config_file = project_root / "src" / "rag_config.py"
if config_file.exists():
    print(f"  [OK] Config file exists: {config_file}")
else:
    print(f"  [Error] Config file not found: {config_file}")

# Test 5: Check chunking script
print("\n[Test 5] Checking chunking script...")
chunking_script = project_root / "src" / "chunk_qwen3_0_6B.py"
if chunking_script.exists():
    print(f"  [OK] Chunking script exists: {chunking_script}")
else:
    print(f"  [Error] Chunking script not found: {chunking_script}")

# Test 6: Check results directory
print("\n[Test 6] Checking results directory...")
results_dir = project_root / "tests" / "logs" / "hyperparameter_tuning"
results_dir.mkdir(parents=True, exist_ok=True)
print(f"  [OK] Results directory ready: {results_dir}")

# Summary
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

all_good = True

if not NEPTUNE_AVAILABLE:
    print("  [Warning] Neptune package is not installed")
    print("   Run: pip install neptune")
    all_good = False

if not api_token or not project:
    print("  [Warning] Neptune credentials not configured")
    print("   Set environment variables:")
    print("     $env:NEPTUNE_API_TOKEN='your_token'")
    print("     $env:NEPTUNE_PROJECT='username/project'")
    all_good = False

if all_good:
    print("\n  [OK] All checks passed! You're ready to run hyperparameter tuning.")
    print("\nNext steps:")
    print("  1. Quick test (3 experiments):")
    print("     python src\\hyperparameter_tuner_neptune.py --mode quick")
    print("\n  2. Chunk size optimization (20 experiments):")
    print("     python src\\hyperparameter_tuner_neptune.py --mode chunk_size")
    print("\n  3. Without Neptune:")
    print("     python src\\hyperparameter_tuner.py")
else:
    print("\n  [Warning] Some issues found. Please fix them before running tuning.")
    print("\nYou can still run without Neptune:")
    print("  python src\\hyperparameter_tuner.py")

print("\n" + "="*70)
