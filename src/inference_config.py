"""
Inference Configuration
=======================
Shared configuration for LLM inference settings used across the RAG system.

This ensures consistent settings between the Streamlit app and test scripts.

Author: Generated for 01_RAG project
Date: November 6, 2025
"""

# =============================================================================
# RAG SYSTEM CONFIGURATION
# =============================================================================

# Database path (relative to project root)
DEFAULT_DB_PATH = "data/output/chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_cleaned"

# Embedding model for RAG
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Number of chunks to retrieve
TOP_K_RESULTS = 25

# =============================================================================
# LLM MODEL CONFIGURATION
# =============================================================================

# Available LLM models with their settings
MODEL_CONFIG = {
    "InternVL3_5-2B-Q6_K": {
        "path": "models/llamacpp/InternVL3_5-2B-Q6_K.gguf",
        "n_ctx": 40960,
        "temperature": 0.7,
        "top_p": 0.9,
        "n_gpu_layers": -1,  # -1 means use all GPU layers
        "verbose": False
    },
    "InternVL3_5-8B-Q4_K_M": {
        "path": "models/llamacpp/InternVL3_5-8B-Q4_K_M.gguf",
        "n_ctx": 40960,
        "temperature": 0.7,
        "top_p": 0.9,
        "n_gpu_layers": -1,
        "verbose": False
    }
}


# Default model to use
DEFAULT_MODEL = "InternVL3_5-2B-Q6_K"
#DEFAULT_MODEL = "InternVL3_5-8B-Q4_K_M"

# =============================================================================
# INFERENCE SETTINGS
# =============================================================================

# Default maximum tokens for responses
DEFAULT_MAX_TOKENS = 4096

# Maximum tokens options for UI
MAX_TOKENS_OPTIONS = [512, 1024, 2048, 4096, 6144, 8192]

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# System message for InternVL models (with thinking)
SYSTEM_MESSAGE_INTERNVL = """You are a helpful AI assistant. Answer the user's question based on the provided context from the knowledge base.

Before providing your final answer, show your reasoning process inside <think></think> tags. Then provide your clear, accurate, and concise answer outside the tags.

Example format:
<think>
Let me analyze the context... The key points are... Therefore...
</think>
Based on the analysis, the answer is..."""

# System message for other models (without thinking)
SYSTEM_MESSAGE_STANDARD = """You are a helpful AI assistant. Answer the user's question based on the provided context from the knowledge base."""

# Prompt template
PROMPT_TEMPLATE = """<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
Context from knowledge base:
{context}

Question: {query}<|im_end|>
<|im_start|>assistant
"""

# Prompt template with conversation history
PROMPT_TEMPLATE_WITH_HISTORY = """<|im_start|>system
{system_message}<|im_end|>
{history}<|im_start|>user
Context from knowledge base:
{context}

Question: {query}<|im_end|>
<|im_start|>assistant
"""

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Path to test questions (relative to project root)
TEST_QUESTIONS_PATH = "data/test/inference_test_questions.json"

# Log directory for inference tests (relative to project root)
TEST_LOG_DIR = "data/test/logs"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_system_message(model_name: str) -> str:
    """
    Get the appropriate system message based on model name.
    
    Args:
        model_name: Name of the model
    
    Returns:
        System message string
    """
    if "InternVL" in model_name:
        return SYSTEM_MESSAGE_INTERNVL
    else:
        return SYSTEM_MESSAGE_STANDARD


def get_model_config(model_name: str) -> dict:
    """
    Get model configuration by name.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Model configuration dictionary
    
    Raises:
        KeyError: If model name not found
    """
    if model_name not in MODEL_CONFIG:
        raise KeyError(f"Model '{model_name}' not found in configuration. Available models: {list(MODEL_CONFIG.keys())}")
    return MODEL_CONFIG[model_name]


def get_available_models() -> list:
    """
    Get list of available model names.
    
    Returns:
        List of model names
    """
    return list(MODEL_CONFIG.keys())
