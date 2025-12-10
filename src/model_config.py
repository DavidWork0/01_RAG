"""
Inference Configuration
=======================
Shared configuration for LLM inference settings used across the RAG system.

This ensures consistent settings between the Streamlit app and test scripts.

Author: Generated for 01_RAG project
Date: November 6, 2025
"""

import os

# =============================================================================
# RAG SYSTEM CONFIGURATION
# =============================================================================

# Database path (relative to project root)
#DEFAULT_DB_PATH = "data/output/chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1000_250_cleaned"
DEFAULT_DB_PATH = "data/output/chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1000_250_cleaned"

"""
def databases = [
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_500_150_cleaned',  OK
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_500_250_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_750_150_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_750_250_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1000_150_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1000_200_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1000_250_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1000_350_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1000_450_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1250_150_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1250_250_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1250_350_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1250_450_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1500_150_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1500_200_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1500_250_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1500_350_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1500_450_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_2000_150_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_2000_250_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_2000_350_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_2000_450_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_3000_150_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_3000_250_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_3000_350_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_3000_450_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_4000_150_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_4000_250_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_4000_350_cleaned',
                        'chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_4000_450_cleaned',
                        'chroma_db_by_sentence_Qwen_Qwen3-Embedding-0.6B_1024_500_cleaned',
                        'chroma_db_by_sentence_Qwen_Qwen3-Embedding-0.6B_1024_1000_cleaned',
                        'chroma_db_by_sentence_Qwen_Qwen3-Embedding-0.6B_1024_1500_cleaned',
                        'chroma_db_by_sentence_Qwen_Qwen3-Embedding-0.6B_1024_2000_cleaned',
                        'chroma_db_by_sentence_Qwen_Qwen3-Embedding-0.6B_1024_3000_cleaned'
                    ]
"""

# Embedding model for RAG
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Number of chunks to retrieve
TOP_K_RESULTS = 20

# Similarity threshold for filtering chunks (percentage)
SIMILARITY_THRESHOLD = 50.0

# =============================================================================
# LLM MODEL CONFIGURATION
# =============================================================================

# Available LLM models with their settings
MODEL_CONFIG = {
    "InternVL3_5-2B-Q6_K": {
        "path": "models/llamacpp/InternVL3_5-2B-Q6_K.gguf",
        "n_ctx": 32768,
        "temperature": 0.7,
        "top_p": 0.9,
        "n_gpu_layers": -1,  # -1 means use all GPU layers
        "verbose": False
    },
    "InternVL3_5-8B-Q4_K_M": {
        "path": "models/llamacpp/internvl3_5-8b-q4_k_m.gguf",
        "n_ctx": 32768,
        "temperature": 0.7,
        "top_p": 0.9,
        "n_gpu_layers": -1,
        "verbose": False
    },
    "InternVL3-2B-Instruct-Q5_K_M": {
        "path": "models/llamacpp/internvl3-2b-instruct-q5_k_m.gguf",
        "n_ctx": 32768,
        "temperature": 0.7,
        "top_p": 0.9,
        "n_gpu_layers": -1,
        "verbose": False
    },
    "Qwen3-4B-Instruct-2507-UD-Q6_K_XL": {
        "path": "models/llamacpp/Qwen3-4B-Instruct-2507-UD-Q6_K_XL.gguf",
        "n_ctx": 32768,
        "temperature": 0.7,
        "top_p": 0.8,  # Adjusted from 0.9 to match official recommendations
        "top_k": 20,  # Added - recommended by Qwen team
        "min_p": 0.0,  # Added - recommended for consistency
        "repeat_penalty": 1.05,  # Added - helps reduce repetition
        "seed": 42,  # Fixed seed for deterministic responses across environments
        "n_gpu_layers": -1,
        "verbose": False
    },
    "Qwen3-8B-Q6_K": {
        "path": "models/llamacpp/Qwen3-8B-Q6_K.gguf",
        "n_ctx": 32768,
        "temperature": 0.7,
        "top_p": 0.8,  # Adjusted from 0.9 to match official recommendations
        "top_k": 20,  # Added - recommended by Qwen team
        "min_p": 0.0,  # Added - recommended for consistency
        "repeat_penalty": 1.05,  # Added - helps reduce repetition
        "seed": 42,  # Fixed seed for deterministic responses across environments
        "n_gpu_layers": -1,
        "verbose": False
    },
    "OpenGVLab_InternVL3_5-8B-Q6_K": {
        "path": "models/llamacpp/OpenGVLab_InternVL3_5-8B-Q6_K.gguf",
        "n_ctx": 32768,
        "temperature": 0.7,
        "top_p": 0.9,
        "n_gpu_layers": -1,
        "verbose": False
    },
    "InternVL3_5-2B-Q8_0": {
        "path": "models/llamacpp/InternVL3_5-2B-Q8_0.gguf",
        "n_ctx": 32768,
        "temperature": 0.7,
        "top_p": 0.9,
        "n_gpu_layers": -1,
        "verbose": False
    },
    "Qwen3-4B-Instruct-2507-Q8_0": {
        "path": "models/llamacpp/Qwen3-4B-Instruct-2507-Q8_0.gguf",
        "n_ctx": 32768,
        "temperature": 0.7,
        "top_p": 0.8,  # Adjusted from 0.9 to match official recommendations
        "top_k": 20,  # Added - recommended by Qwen team
        "min_p": 0.0,  # Added - recommended for consistency
        "repeat_penalty": 1.05,  # Added - helps reduce repetition
        "seed": 42,  # Fixed seed for deterministic responses across environments
        "n_gpu_layers": -1,
        "verbose": False
    },
    "Qwen3-8B-Q4_K_M": {
        "path": "models/llamacpp/Qwen3-8B-Q4_K_M.gguf",
        "n_ctx": 32768,
        "temperature": 0.7,
        "top_p": 0.8,  # Adjusted from 0.9 to match official recommendations
        "top_k": 20,  # Added - recommended by Qwen team
        "min_p": 0.0,  # Added - recommended for consistency
        "repeat_penalty": 1.05,  # Added - helps reduce repetition
        "seed": 42,  # Fixed seed for deterministic responses across environments
        "n_gpu_layers": -1,
        "verbose": False
    },

    "Qwen3-8B-Q5_K_M": {
        "path": "models/llamacpp/Qwen3-8B-Q5_K_M.gguf",
        "n_ctx": 32768,  # Reduced from 16384, 32768 for better VRAM efficiency
        "temperature": 0.7,
        "top_p": 0.8,  # Adjusted from 0.9 to match official recommendations
        "top_k": 20,  # Added - recommended by Qwen team
        "min_p": 0.0,  # Added - recommended for consistency
        "repeat_penalty": 1.05,  # Added - helps reduce repetition
        "seed": 42,  # Fixed seed for deterministic responses across environments
        "n_gpu_layers": -1,
        "verbose": False
    }
}


# Default model to use this is used by the Streamlit app on startup
#DEFAULT_MODEL = "InternVL3_5-2B-Q6_K"
#DEFAULT_MODEL = "InternVL3_5-2B-Q8_0"
#DEFAULT_MODEL = "Qwen3-8B-Q4_K_M"
DEFAULT_MODEL = "Qwen3-8B-Q5_K_M" #activated only for thinking parse testing
#DEFAULT_MODEL = "Qwen3-4B-Instruct-2507-Q8_0" #BEST MODEL


# =============================================================================
# INFERENCE SETTINGS
# =============================================================================

# Default maximum tokens for responses
DEFAULT_MAX_TOKENS = 8192

# Maximum tokens options for UI
MAX_TOKENS_OPTIONS = [512, 1024, 2048, 4096, 6144, 8192]

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# System message for other models (without thinking)
SYSTEM_MESSAGE_STANDARD = """You are an AI assistant for a Retrieval-Augmented Generation (RAG) system focused on technical documentation.

Key principles:
- Prioritize retrieved context over general knowledge
- Extract and synthesize ALL relevant information from the context
- Structure responses clearly with proper formatting (bullet points, sections, etc.)
- If context is insufficient, state this clearly before providing general knowledge
- Cite specific sections when referencing context (e.g., "According to the manual...")
- Be precise, accurate, and comprehensive in your answers"""


# System message for InternVL models (with thinking)
SYSTEM_MESSAGE_INTERNVL =  """You are a helpful AI assistant specialized in hybrid Retrieval-Augmented Generation (RAG) tasks. Your role is to answer the user's question using both retrieved context from the knowledge base and reasoning based on prior conversation history.

Always:
- Analyze the retrieved context carefully before forming an answer.
- Outside the tags, write your final answer clearly, accurately, and concisely in English.
- If you recommend actions, ensure they are directly supported by the context.

Example structure:
<think>
Step-by-step reasoning and evidence analysis...
</think>

Your clear and concise answer here."""

SYSTEM_MESSAGE_STANDARD_OLD = """You are an AI assistant designed for a Retrieval-Augmented Generation (RAG) system. Your primary goal is to answer the user's question using the retrieved context from the knowledge base.
-If the context contains the answer, use it directly and clearly attribute or reference it when appropriate.
-If the context does not contain relevant information, respond concisely using general knowledge without inventing details or assuming missing information.
-Always provide structured, accurate, and easy-to-read responses.
-Collect every important detail from the context and present it in a clear manner."""

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
#TEST_QUESTIONS_PATH= "data/test/refined_inference_test_questions.json"

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
        return SYSTEM_MESSAGE_STANDARD
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


def parse_thinking_response(response_text: str) -> dict:
    """
    Parse LLM response to separate thinking process from final answer.
    
    This function extracts thinking tags (<think>, <thinking>, etc.) from the
    response and returns both the thinking process and the final answer.
    
    Args:
        response_text: Raw LLM response text
    
    Returns:
        Dictionary with keys:
            - has_thinking (bool): Whether thinking tags were found
            - thinking (str or None): Extracted thinking content
            - answer (str): Final answer with thinking tags removed
    """
    import re
    
    patterns = [
        (r'<think>(.*?)</think>', 'think'),
        (r'<thinking>(.*?)</thinking>', 'thinking'),
        (r'<thoughts>(.*?)</thoughts>', 'thoughts'),
        (r'\[THINKING\](.*?)\[/THINKING\]', 'bracket'),
    ]
    
    for pattern, tag_type in patterns:
        thinking_matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        if thinking_matches:
            thinking = '\n\n'.join([t.strip() for t in thinking_matches])
            final_answer = re.sub(pattern, '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()
            final_answer = re.sub(r'\n{3,}', '\n\n', final_answer).strip()
            
            return {
                'has_thinking': True,
                'thinking': thinking,
                'answer': final_answer if final_answer else "Answer extracted from thinking process."
            }
    
    return {
        'has_thinking': False,
        'thinking': None,
        'answer': response_text.strip()
    }


def load_llm_model(model_name: str, project_root):
    """
    Load and instantiate a Llama model with configuration from MODEL_CONFIG.
    
    This ensures consistent model loading across test_inference and streamlit_dashboard.
    
    Args:
        model_name: Name of the model (must exist in MODEL_CONFIG)
        project_root: Path object pointing to the project root directory
    
    Returns:
        Llama model instance, or None if loading fails
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        KeyError: If model_name not found in MODEL_CONFIG
    """
    from pathlib import Path
    
    # Get model configuration
    config = get_model_config(model_name)
    
    # Resolve model path using project root first, then fallbacks (useful for Jenkins/docker)
    rel_model_path = Path(config["path"])
    search_roots = [Path(project_root)]
    env_root_candidates = [
        os.environ.get('RAG_PROJECT_ROOT_OVERRIDE'),
        os.environ.get('RAG_MODEL_ROOT'),
        os.environ.get('RAG_PROJECT_ROOT_FALLBACK')
    ]
    for root in env_root_candidates:
        if root:
            search_roots.append(Path(root))
    candidate_paths = [(root / rel_model_path) for root in search_roots]
    llama_models_dir = os.environ.get('RAG_LLM_MODELS_DIR')
    if llama_models_dir:
        llama_base = Path(llama_models_dir)
        try:
            relative_to_llama = rel_model_path.relative_to(Path('models') / 'llamacpp')
            candidate_paths.append(llama_base / relative_to_llama)
        except ValueError:
            candidate_paths.append(llama_base / rel_model_path.name)
    direct_override = os.environ.get('RAG_LLM_MODEL_PATH')
    if direct_override:
        candidate_paths.append(Path(direct_override))
    model_path = next((path for path in candidate_paths if path.exists()), None)
    if model_path is None:
        searched_locations = '\n  - '.join(str(path) for path in candidate_paths)
        raise FileNotFoundError(
            f"Model file '{rel_model_path.name}' not found in any of the searched locations:\n  - {searched_locations}"
        )
    
    # Import llama_cpp here to avoid import errors if not installed
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(f"llama-cpp-python not installed: {e}")
    
    # Instantiate model with config parameters
    # Note: For multi-user scenarios, each call creates a separate model instance
    # to avoid KV cache contamination between users
    llm_params = {
        "model_path": str(model_path),
        "n_ctx": config["n_ctx"],
        "n_gpu_layers": config["n_gpu_layers"],
        "temperature": config["temperature"],
        "verbose": config["verbose"]
    }
    
    # Add seed if specified in config for deterministic responses
    if "seed" in config:
        llm_params["seed"] = config["seed"]
    
    llm = Llama(**llm_params)
    # KV cache is managed per-instance; in streamlit with @st.cache_resource,
    # one instance is shared, so we rely on thread locking to prevent
    # concurrent access that could mix KV cache states
    
    return llm


def generate_llm_response(
    llm_model,
    query: str,
    context: str,
    model_name: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    chat_history: list = None,
    temperature: float = None
) -> str:
    """
    Generate LLM response using retrieved context and optional chat history.
    
    This is the shared core function for LLM inference used by both test_inference
    and streamlit_dashboard. It handles prompt building with or without conversation
    history and calls the LLM with appropriate parameters.
    
    Args:
        llm_model: Loaded Llama model instance
        query: User's question/query
        context: Retrieved context from RAG system
        model_name: Name of the model (for getting system message and config)
        max_tokens: Maximum tokens for response (default: DEFAULT_MAX_TOKENS)
        chat_history: Optional list of previous messages [{'role': 'user/assistant', 'content': '...'}]
        temperature: Optional temperature override (uses model config if None)
    
    Returns:
        Generated response text from the LLM
    """
    # Get system message from config
    system_message = get_system_message(model_name)
    
    # Get model config for inference settings
    model_config = get_model_config(model_name)
    
    # Use provided temperature or fall back to model config
    temp = temperature if temperature is not None else model_config['temperature']
    
    # Build prompt based on whether we have chat history
    if chat_history and len(chat_history) > 0:
        # Build conversation history string
        history_str = ""
        recent_history = chat_history[-10:]  # Keep last 10 messages for context
        for msg in recent_history:
            if msg['role'] == 'user':
                history_str += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg['role'] == 'assistant':
                history_str += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        
        # Use template with history
        if context and context.strip():
            prompt = PROMPT_TEMPLATE_WITH_HISTORY.format(
                system_message=system_message,
                history=history_str,
                context=context,
                query=query
            )
        else:
            # No context but has history
            prompt = f"""<|im_start|>system
{system_message}<|im_end|>
{history_str}<|im_start|>user
Knowledge base search: No relevant documents found.

Question: {query}

Provide an answer based on general knowledge. Begin your response by stating: "I couldn't find this information in the provided documents."<|im_end|>
<|im_start|>assistant
"""
    else:
        # No history - use simple template
        if context and context.strip():
            prompt = PROMPT_TEMPLATE.format(
                system_message=system_message,
                context=context,
                query=query
            )
        else:
            # No context and no history
            prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
Knowledge base search: No relevant documents found.

Question: {query}

Provide an answer based on general knowledge. Begin your response by stating: "I couldn't find this information in the provided documents."<|im_end|>
<|im_start|>assistant
"""
    
    # Call LLM with configured parameters
    try:
        output = llm_model(
            prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=model_config['top_p'],
            echo=False
        )
        return output["choices"][0]["text"]
    except Exception as e:
        return f"Error generating response: {str(e)}"
