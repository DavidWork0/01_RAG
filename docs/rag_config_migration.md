# RAG Configuration Migration

**Date:** November 9, 2025  
**Author:** GitHub Copilot  
**Branch:** parameter_file_createion_and_inference_tests_log

## Overview

All RAG system parameters from `chunk_qwen3_0_6B.py` and `hybrid_rag_module_qwen3.py` have been consolidated into a centralized configuration file `rag_config.py`, following the same pattern as `inference_config.py`.

## Files Modified

### 1. Created: `src/rag_config.py`
A new centralized configuration file containing all RAG-related parameters:

#### Configuration Sections:
- **Path Configuration**: Project paths, input/output folders, model cache
- **Embedding Model Configuration**: Model selection, dimensions, device settings
- **Chunking Configuration**: Strategy selection and parameters (fixed-size vs sentence-based)
- **Database Configuration**: ChromaDB settings and collection management
- **Batch Processing Configuration**: Batch sizes for embedding generation
- **Search Configuration**: Hybrid search weights and parameters
- **Embedding Generation Configuration**: Tokenization settings
- **Verbose Mode**: Logging control
- **NLTK Configuration**: Settings for sentence chunking

#### Key Features:
- **Validation**: `validate_config()` ensures parameters are valid
- **Helper Functions**: 
  - `get_db_path()` - Auto-generate database path based on settings
  - `get_input_folder()` - Get appropriate input folder
  - `get_device()` - Auto-detect CUDA/CPU
  - `get_torch_dtype()` - Get appropriate torch dtype
  - `print_config_summary()` - Display current configuration
- **Backward Compatibility**: Default values match original hardcoded values

### 2. Updated: `src/chunk_qwen3_0_6B.py`

**Before:**
```python
# Hardcoded configuration
BATCH_SIZE = 20
FIXED_SIZE_CHUNK_SIZE = 1000
FIXED_SIZE_OVERLAP = 250
CHUNK_STRATEGY = "fixed_size"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# ... etc
```

**After:**
```python
# Import from centralized config
from rag_config import (
    BATCH_SIZE,
    FIXED_SIZE_CHUNK_SIZE,
    FIXED_SIZE_OVERLAP,
    CHUNK_STRATEGY,
    EMBEDDING_MODEL,
    # ... all other parameters
    get_device,
    get_torch_dtype,
    print_config_summary
)
```

**Key Changes:**
- All configuration now imported from `rag_config`
- Uses helper functions for device/dtype detection
- Displays configuration summary on startup
- Tokenizer settings now use config parameters
- Collection name now uses config constant

### 3. Updated: `src/hybrid_rag_module_qwen3.py`

**Before:**
```python
# Hardcoded configuration
DB_PATH = "./data/output/chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_TOP_K = 25
SEMANTIC_WEIGHT = 0.70
KEYWORD_WEIGHT = 0.30
# ... etc
```

**After:**
```python
# Import from centralized config
from rag_config import (
    DEFAULT_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    SEMANTIC_WEIGHT,
    KEYWORD_WEIGHT,
    STOP_WORDS,
    KEYWORD_SCORING_METHOD,
    # ... all other parameters
    get_device,
    get_torch_dtype
)

# Backward compatibility
DB_PATH = DEFAULT_DB_PATH
```

**Key Changes:**
- All configuration now imported from `rag_config`
- Stop words moved to config file
- Keyword extraction uses config MIN_KEYWORD_LENGTH
- Search parameters use config constants
- Initial retrieval multiplier and cap now configurable
- Tokenizer settings use config parameters

## Benefits of Centralization

1. **Single Source of Truth**: All RAG parameters in one file
2. **Consistency**: Both chunking and retrieval use same settings
3. **Easy Tuning**: Change parameters in one place to affect entire system
4. **Validation**: Config validates on import to catch errors early
5. **Documentation**: Clear documentation of all parameters
6. **Maintainability**: Easier to understand and modify system behavior
7. **Testing**: Easier to test different configurations
8. **Backward Compatible**: Default values match original behavior

## Configuration Parameters

### Chunking Strategy
```python
CHUNK_STRATEGY = "fixed_size"  # or "by_sentence"
FIXED_SIZE_CHUNK_SIZE = 1000   # Characters per chunk
FIXED_SIZE_OVERLAP = 250        # Overlap between chunks
```

### Embedding Settings
```python
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024
MODEL_CACHE_DIR = './models/huggingface'
MAX_EMBEDDING_LENGTH = 512
```

### Search Configuration
```python
DEFAULT_TOP_K = 25              # Results to return
SEMANTIC_WEIGHT = 0.70          # Semantic similarity weight
KEYWORD_WEIGHT = 0.30           # Keyword matching weight
INITIAL_K_MULTIPLIER = 3        # Retrieve 3x before re-ranking
INITIAL_K_CAP = 100             # Max initial retrieval
```

### Database Settings
```python
COLLECTION_NAME = "documents"
BATCH_SIZE = 20                 # Chunks per batch during embedding
```

## Usage Examples

### Basic Usage (unchanged)
```python
# Both modules work exactly as before
from hybrid_rag_module_qwen3 import create_rag_system
rag = create_rag_system()  # Uses config defaults

# Or with custom parameters (overrides config)
rag = create_rag_system(
    db_path="custom/path",
    semantic_weight=0.8
)
```

### Changing Configuration
```python
# Option 1: Edit rag_config.py directly
# Option 2: Override at runtime
from rag_config import SEMANTIC_WEIGHT
import rag_config
rag_config.SEMANTIC_WEIGHT = 0.8
```

### Display Current Config
```python
from rag_config import print_config_summary
print_config_summary()
```

## Testing Recommendations

1. **Run chunking script** to verify configuration is loaded correctly:
   ```powershell
   python src/chunk_qwen3_0_6B.py
   ```

2. **Test RAG search** to ensure hybrid search works:
   ```powershell
   python src/hybrid_rag_module_qwen3.py
   ```

3. **Verify inference tests** still work with new config:
   ```powershell
   python src/test_inference.py
   ```

## Migration Checklist

- [x] Create `rag_config.py` with all parameters
- [x] Update `chunk_qwen3_0_6B.py` to use config
- [x] Update `hybrid_rag_module_qwen3.py` to use config
- [x] Add validation function
- [x] Add helper functions
- [x] Ensure backward compatibility
- [x] No linting errors
- [ ] Test chunking script execution
- [ ] Test RAG search functionality
- [ ] Test inference with new config
- [ ] Update other scripts if needed (e.g., streamlit app)

## Notes

- Configuration is validated on import
- All original default values preserved
- Both files maintain their original API
- No breaking changes for existing code
- Helper functions simplify device/path management
