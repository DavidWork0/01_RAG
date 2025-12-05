"""
RAG System Configuration
========================
Shared configuration for RAG chunking, embedding, and retrieval operations.

This file contains all parameters for:
- Text chunking strategies
- Embedding model settings
- Vector database configuration
- Hybrid search parameters

Author: Generated for 01_RAG project
Date: November 9, 2025
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Get absolute paths relative to this config file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Input folder for text files (cleaned versions)
DEFAULT_INPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output", "final_merged", "cleaned")

# Alternative input folder (uncleaned)
INPUT_FOLDER_UNCLEANED = os.path.join(PROJECT_ROOT, "data", "output", "final_merged")

# Model cache directory can be overridden via environment to support
# Jenkins/docker setups where the repo checkout does not contain the
# heavyweight model artifacts.
_DEFAULT_MODEL_CACHE_DIR = Path(PROJECT_ROOT) / 'models' / 'huggingface'

def _normalize_hf_cache_path(path: Path) -> Path:
    """Ensure the cache path points to the huggingface cache directory."""
    path = path.expanduser()
    if path.name == 'huggingface':
        return path
    candidate = path / 'huggingface'
    if candidate.exists():
        return candidate
    return path

def _resolve_model_cache_dir() -> Path:
    env_candidates = [
        os.environ.get('RAG_MODEL_CACHE_DIR'),
        os.environ.get('HF_HOME'),
        os.environ.get('HUGGINGFACE_HUB_CACHE'),
        os.environ.get('TRANSFORMERS_CACHE'),
    ]
    for candidate in env_candidates:
        if not candidate:
            continue
        normalized = _normalize_hf_cache_path(Path(candidate))
        if normalized.exists():
            return normalized
    return _DEFAULT_MODEL_CACHE_DIR

MODEL_CACHE_DIR = str(_resolve_model_cache_dir())

# Ensure downstream HuggingFace clients default to the same cache directory.
for _env_var in (
    'HF_HOME',
    'HUGGINGFACE_HUB_CACHE',
    'TRANSFORMERS_CACHE',
    'RAG_MODEL_CACHE_DIR'
):
    os.environ[_env_var] = MODEL_CACHE_DIR
del _env_var

# =============================================================================
# EMBEDDING MODEL CONFIGURATION
# =============================================================================

# Embedding model identifier from HuggingFace
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Embedding vector dimension
EMBEDDING_DIMENSION = 1024

# Device configuration
# Options: 'cuda', 'cpu', or None (auto-detect)
DEVICE = None  # None = auto-detect

# Torch dtype for model
# 'float16' for CUDA, 'float32' for CPU
TORCH_DTYPE = 'auto'  # 'auto' = float16 for CUDA, float32 for CPU

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

# Chunking strategy
# Options: "fixed_size", "by_sentence"
CHUNK_STRATEGY = "fixed_size"

# Fixed-size chunking parameters
FIXED_SIZE_CHUNK_SIZE = 1000  # Characters per chunk
FIXED_SIZE_OVERLAP = 250      # Character overlap between chunks

# Sentence-based chunking parameters
CHUNK_SIZE_MAX_BY_SENTENCE = 1000  # Max characters per chunk for sentence-based chunking

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Auto-generate collection name based on chunking settings
def get_collection_name(chunk_strategy=None, chunk_size=None, overlap=None):
    """
    Generate collection name based on chunking configuration.
    
    Args:
        chunk_strategy: Chunking strategy (defaults to CHUNK_STRATEGY)
        chunk_size: Chunk size (defaults to FIXED_SIZE_CHUNK_SIZE or CHUNK_SIZE_MAX_BY_SENTENCE)
        overlap: Overlap size (defaults to FIXED_SIZE_OVERLAP, only for fixed_size)
    
    Returns:
        Collection name string
    """
    chunk_strategy = chunk_strategy or CHUNK_STRATEGY
    
    if chunk_strategy == "fixed_size":
        chunk_size = chunk_size or FIXED_SIZE_CHUNK_SIZE
        overlap = overlap or FIXED_SIZE_OVERLAP
        return f"docs_{chunk_strategy}_{chunk_size}_{overlap}"
    elif chunk_strategy == "by_sentence":
        chunk_size = chunk_size or CHUNK_SIZE_MAX_BY_SENTENCE
        return f"docs_{chunk_strategy}_{chunk_size}"
    else:
        return "documents"

# Collection name in ChromaDB (dynamically generated)
COLLECTION_NAME = get_collection_name()

# Auto-generate database path based on settings
def get_db_path(chunk_strategy=None, embedding_model=None, embedding_dim=None, chunk_size=None, overlap=None, cleaned=True):
    """
    Generate database path based on configuration.
    
    Args:
        chunk_strategy: Chunking strategy (defaults to CHUNK_STRATEGY)
        embedding_model: Model name (defaults to EMBEDDING_MODEL)
        embedding_dim: Embedding dimension (defaults to EMBEDDING_DIMENSION)
        chunk_size: Chunk size (defaults based on strategy)
        overlap: Overlap size (defaults to FIXED_SIZE_OVERLAP, only for fixed_size)
        cleaned: Whether using cleaned data (defaults to True)
    
    Returns:
        Full path to database directory
    """
    chunk_strategy = chunk_strategy or CHUNK_STRATEGY
    embedding_model = embedding_model or EMBEDDING_MODEL
    embedding_dim = embedding_dim or EMBEDDING_DIMENSION
    
    # Build the db_type with chunking parameters
    if chunk_strategy == "fixed_size":
        chunk_size = chunk_size or FIXED_SIZE_CHUNK_SIZE
        overlap = overlap or FIXED_SIZE_OVERLAP
        db_type = f"chroma_db_{chunk_strategy}_{embedding_model.replace('/', '_')}_{embedding_dim}_{chunk_size}_{overlap}"
    elif chunk_strategy == "by_sentence":
        chunk_size = chunk_size or CHUNK_SIZE_MAX_BY_SENTENCE
        db_type = f"chroma_db_{chunk_strategy}_{embedding_model.replace('/', '_')}_{embedding_dim}_{chunk_size}"
    else:
        db_type = f"chroma_db_{chunk_strategy}_{embedding_model.replace('/', '_')}_{embedding_dim}"
    
    if cleaned:
        db_type += "_cleaned"
    
    return os.path.join(PROJECT_ROOT, "data", "output", db_type)


def _resolve_env_db_path():
    """Allow overriding DB path via env (absolute or project-root relative)."""
    env_path = os.environ.get("RAG_DB_PATH") or os.environ.get("RAG_OVERRIDE_DB_PATH")
    if not env_path:
        return None
    if os.path.isabs(env_path):
        return env_path
    return os.path.join(PROJECT_ROOT, env_path)


def get_db_path_with_env(chunk_strategy=None, embedding_model=None, embedding_dim=None, chunk_size=None, overlap=None, cleaned=True):
    override = _resolve_env_db_path()
    if override:
        return override
    return get_db_path(chunk_strategy, embedding_model, embedding_dim, chunk_size, overlap, cleaned)


# Default database path (env override takes precedence when set)
DEFAULT_DB_PATH = get_db_path_with_env()

# =============================================================================
# BATCH PROCESSING CONFIGURATION
# =============================================================================

# Number of chunks to process in each batch during embedding creation
# Recommended values:
# - 20-25 for Qwen3-0.6B with 6-8GB VRAM
# - 100 for Qwen3-0.6B with 12GB+ VRAM
BATCH_SIZE = 50

# =============================================================================
# SEARCH CONFIGURATION - HYBRID RAG
# =============================================================================

# Default number of results to retrieve
DEFAULT_TOP_K = 20

# Minimum similarity score threshold (0-100)
# Results with similarity scores below this threshold will be filtered out
MIN_SIMILARITY_THRESHOLD = 40.0

# Hybrid search weights (must sum to ~1.0)
SEMANTIC_WEIGHT = 0.70  # Weight for semantic similarity (0-1)
KEYWORD_WEIGHT = 0.30   # Weight for keyword matching (0-1)

# Initial retrieval multiplier for re-ranking
# Retrieves (TOP_K * INITIAL_K_MULTIPLIER) results before keyword re-ranking
INITIAL_K_MULTIPLIER = 3

# Cap for initial retrieval
INITIAL_K_CAP = 100

# Stop words for keyword extraction
STOP_WORDS = {
    'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 
    'was', 'were', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for',
    'and', 'or', 'but', 'with', 'from', 'of', 'by', 'as', 'that',
    'this', 'these', 'those', 'be', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
    'can', 'may', 'might', 'must', 'shall'
}

# Minimum keyword length
MIN_KEYWORD_LENGTH = 2

# Keyword scoring method
# Options: "simple", "tfidf"
KEYWORD_SCORING_METHOD = "simple"

# =============================================================================
# EMBEDDING GENERATION CONFIGURATION
# =============================================================================

# Maximum INPUT sequence length (in tokens) for the embedding model
# This limits how many tokens the tokenizer will process from input text
# 
# What this affects:
#   - Individual text chunks during database creation (each chunk is embedded separately)
#   - User search queries during RAG retrieval (query text is embedded for similarity search)
# 
# What this does NOT affect:
#   - Total context length passed to LLM (limited by LLM's n_ctx, e.g., 40,960 tokens)
#   - Number of retrieved chunks (limited by top_k parameter)
#   - Length of concatenated context (retrieved chunks are NOT re-embedded, just concatenated as text)
# 
# Note: This is different from EMBEDDING_DIMENSION (output vector size = 1024)
# Qwen3-Embedding-0.6B: accepts up to 512 input tokens → outputs 1024-dimensional vectors
MAX_EMBEDDING_LENGTH = 512

# Padding and truncation settings
PADDING = True
TRUNCATION = True

# =============================================================================
# VERBOSE MODE
# =============================================================================

# Whether to print detailed logs during processing
VERBOSE_MODE = False

# Whether to print initialization messages in HybridRAG
VERBOSE_RAG = True

# =============================================================================
# NLTK CONFIGURATION (for sentence chunking)
# =============================================================================

# Whether to download NLTK data quietly
NLTK_QUIET = True

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """
    Validate configuration parameters.
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Check weights sum to approximately 1.0
    weight_sum = SEMANTIC_WEIGHT + KEYWORD_WEIGHT
    if not (0.99 <= weight_sum <= 1.01):
        raise ValueError(
            f"SEMANTIC_WEIGHT + KEYWORD_WEIGHT must sum to ~1.0, got {weight_sum}"
        )
    
    # Check weights are in valid range
    if not (0.0 <= SEMANTIC_WEIGHT <= 1.0):
        raise ValueError(f"SEMANTIC_WEIGHT must be between 0 and 1, got {SEMANTIC_WEIGHT}")
    
    if not (0.0 <= KEYWORD_WEIGHT <= 1.0):
        raise ValueError(f"KEYWORD_WEIGHT must be between 0 and 1, got {KEYWORD_WEIGHT}")
    
    # Check minimum similarity threshold is valid
    if not (0.0 <= MIN_SIMILARITY_THRESHOLD <= 100.0):
        raise ValueError(
            f"MIN_SIMILARITY_THRESHOLD must be between 0 and 100, got {MIN_SIMILARITY_THRESHOLD}"
        )
    
    # Check chunking strategy is valid
    valid_strategies = ["fixed_size", "by_sentence"]
    if CHUNK_STRATEGY not in valid_strategies:
        raise ValueError(
            f"CHUNK_STRATEGY must be one of {valid_strategies}, got '{CHUNK_STRATEGY}'"
        )
    
    # Check overlap is less than chunk size
    if CHUNK_STRATEGY == "fixed_size" and FIXED_SIZE_OVERLAP >= FIXED_SIZE_CHUNK_SIZE:
        raise ValueError(
            f"FIXED_SIZE_OVERLAP ({FIXED_SIZE_OVERLAP}) must be less than "
            f"FIXED_SIZE_CHUNK_SIZE ({FIXED_SIZE_CHUNK_SIZE})"
        )
    
    # Check keyword scoring method
    valid_methods = ["simple", "tfidf"]
    if KEYWORD_SCORING_METHOD not in valid_methods:
        raise ValueError(
            f"KEYWORD_SCORING_METHOD must be one of {valid_methods}, "
            f"got '{KEYWORD_SCORING_METHOD}'"
        )

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_input_folder(cleaned=True):
    """
    Get input folder path.
    
    Args:
        cleaned: Whether to use cleaned data folder
    
    Returns:
        Path to input folder
    """
    return DEFAULT_INPUT_FOLDER if cleaned else INPUT_FOLDER_UNCLEANED


def get_device():
    """
    Get compute device.
    
    Returns:
        'cuda' if available, else 'cpu'
    """
    if DEVICE is not None:
        return DEVICE
    
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_torch_dtype():
    """
    Get torch dtype based on device.
    
    Returns:
        torch dtype
    """
    import torch
    
    if TORCH_DTYPE == 'auto':
        return torch.float16 if get_device() == 'cuda' else torch.float32
    elif TORCH_DTYPE == 'float16':
        return torch.float16
    elif TORCH_DTYPE == 'float32':
        return torch.float32
    else:
        raise ValueError(f"Invalid TORCH_DTYPE: {TORCH_DTYPE}")


def print_config_summary():
    """Print a summary of the current configuration."""
    print("\n" + "="*70)
    print("RAG SYSTEM CONFIGURATION SUMMARY")
    print("="*70)
    
    print("\n[PATHS]")
    print(f"  Project Root:     {PROJECT_ROOT}")
    print(f"  Input Folder:     {DEFAULT_INPUT_FOLDER}")
    print(f"  Model Cache:      {MODEL_CACHE_DIR}")
    print(f"  Database Path:    {DEFAULT_DB_PATH}")
    
    print("\n[EMBEDDING MODEL]")
    print(f"  Model:            {EMBEDDING_MODEL}")
    print(f"  Dimension:        {EMBEDDING_DIMENSION}")
    print(f"  Device:           {get_device()}")
    print(f"  Torch dtype:      {get_torch_dtype()}")
    
    print("\n[CHUNKING]")
    print(f"  Strategy:         {CHUNK_STRATEGY}")
    if CHUNK_STRATEGY == "fixed_size":
        print(f"  Chunk Size:       {FIXED_SIZE_CHUNK_SIZE} chars")
        print(f"  Overlap:          {FIXED_SIZE_OVERLAP} chars")
    else:
        print(f"  Max Size:         {CHUNK_SIZE_MAX_BY_SENTENCE} chars")
    
    print("\n[DATABASE]")
    print(f"  Collection Name:  {COLLECTION_NAME}")
    print(f"  Batch Size:       {BATCH_SIZE}")
    
    print("\n[SEARCH]")
    print(f"  Default Top-K:    {DEFAULT_TOP_K}")
    print(f"  Min Similarity:   {MIN_SIMILARITY_THRESHOLD}%")
    print(f"  Semantic Weight:  {SEMANTIC_WEIGHT}")
    print(f"  Keyword Weight:   {KEYWORD_WEIGHT}")
    print(f"  Scoring Method:   {KEYWORD_SCORING_METHOD}")
    
    print("\n" + "="*70 + "\n")


# Validate on import
validate_config()
