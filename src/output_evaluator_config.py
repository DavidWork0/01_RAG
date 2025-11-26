"""
Answer Evaluator Configuration
================================
Configuration parameters for the answer quality evaluation system.

This file contains all configurable parameters for comparing generated answers
against reference answers using various text similarity metrics.

Author: Generated for 01_RAG project
Date: November 26, 2025
"""

# =============================================================================
# INPUT/OUTPUT PATHS
# =============================================================================

# Default path to gold standard dataset (relative to project root)
DEFAULT_GOLD_STANDARD_PATH = "data/test/sonar-reasoning-pro_gold_standard_dataset.json"

# Default path to inference log file (relative to project root)
DEFAULT_INFERENCE_LOG_PATH = "tests/logs/inference_log.jsonl"

# Default output path for evaluation results JSON (relative to project root)
DEFAULT_OUTPUT_JSON_PATH = "tests/sessions/answer_compare/evaluation_results.json"

# =============================================================================
# INFERENCE LOG SELECTION
# =============================================================================

# Which inference log entry to use for each question when multiple exist
# Options:
#   -1: Use the LAST (most recent) inference log entry
#    0: Use the FIRST inference log entry
#    N: Use the Nth inference log entry (0-indexed)
INFERENCE_LOG_INDEX = -1

# Number of inference sessions to evaluate (starting from last backwards)
# Set to 1 to evaluate only one session (specified by INFERENCE_LOG_INDEX)
# Set to N to evaluate last N sessions
# Set to None to evaluate all available sessions
NUM_INFERENCE_EVAL = 1

# Maximum number of questions to evaluate
# Set to None to evaluate all questions
# Set to a positive integer to limit evaluation to first N questions
MAX_QUESTIONS_TO_EVALUATE = None

# Skip questions with missing reference answers
SKIP_MISSING_REFERENCE_ANSWERS = True

# Skip questions with missing generated answers
SKIP_MISSING_GENERATED_ANSWERS = True

# =============================================================================
# EMBEDDING MODEL CONFIGURATION
# =============================================================================

# Sentence transformer model for semantic similarity calculation
# Should match the embedding model used in your RAG system for consistency
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Alternative embedding models (uncomment to use):
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, lightweight
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Better quality
# EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # High quality, slower

# =============================================================================
# METRICS CONFIGURATION
# =============================================================================

# Enable/disable specific metrics
ENABLE_SEMANTIC_SIMILARITY = True  # Embedding-based cosine similarity
ENABLE_ROUGE_SCORES = True         # ROUGE-1, ROUGE-2, ROUGE-L
ENABLE_BLEU_SCORE = True           # BLEU score
ENABLE_TFIDF_SIMILARITY = True     # TF-IDF cosine similarity

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Print verbose evaluation progress
VERBOSE_OUTPUT = True

# Print individual question results (not just aggregates)
PRINT_INDIVIDUAL_RESULTS = False

# Number of decimal places for metric display
METRIC_DECIMAL_PLACES = 4

# Include raw reference and generated answers in JSON output
INCLUDE_FULL_ANSWERS_IN_JSON = False

# =============================================================================
# STATISTICAL AGGREGATION
# =============================================================================

# Statistical measures to compute for aggregate metrics
COMPUTE_MEAN = True
COMPUTE_MEDIAN = True
COMPUTE_STD = True
COMPUTE_MIN = True
COMPUTE_MAX = True

# =============================================================================
# FILTERING OPTIONS
# =============================================================================

# Minimum answer length (characters) to include in evaluation
# Set to 0 to include all answers
MIN_ANSWER_LENGTH = 0

# Maximum answer length ratio (generated/reference) to flag as outlier
# Set to None to disable
MAX_LENGTH_RATIO_THRESHOLD = None

# Questions to exclude from evaluation (list of question IDs)
EXCLUDE_QUESTION_IDS = []

# Questions to include in evaluation (list of question IDs)
# If empty, all questions are included (except those in EXCLUDE_QUESTION_IDS)
INCLUDE_QUESTION_IDS = []

# =============================================================================
# ROUGE CONFIGURATION
# =============================================================================

# Use stemming in ROUGE calculation
ROUGE_USE_STEMMER = True

# =============================================================================
# BLEU CONFIGURATION
# =============================================================================

# BLEU smoothing method (1-7, or None for no smoothing)
# Method 1 (default): Add epsilon counts to precision
BLEU_SMOOTHING_METHOD = 1

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_inference_log_index():
    """
    Get the inference log index to use.
    
    Returns:
        int: Index for selecting inference log entry
    """
    return INFERENCE_LOG_INDEX


def get_max_questions():
    """
    Get the maximum number of questions to evaluate.
    
    Returns:
        int or None: Maximum number of questions, or None for all
    """
    return MAX_QUESTIONS_TO_EVALUATE


def get_embedding_model():
    """
    Get the embedding model name.
    
    Returns:
        str: Embedding model name
    """
    return EMBEDDING_MODEL


def should_skip_missing_reference():
    """
    Check if questions with missing reference answers should be skipped.
    
    Returns:
        bool: True to skip, False to include
    """
    return SKIP_MISSING_REFERENCE_ANSWERS


def should_skip_missing_generated():
    """
    Check if questions with missing generated answers should be skipped.
    
    Returns:
        bool: True to skip, False to include
    """
    return SKIP_MISSING_GENERATED_ANSWERS


def is_metric_enabled(metric_name):
    """
    Check if a specific metric is enabled.
    
    Args:
        metric_name: Name of the metric (semantic, rouge, bleu, tfidf)
    
    Returns:
        bool: True if enabled, False otherwise
    """
    metric_flags = {
        'semantic': ENABLE_SEMANTIC_SIMILARITY,
        'rouge': ENABLE_ROUGE_SCORES,
        'bleu': ENABLE_BLEU_SCORE,
        'tfidf': ENABLE_TFIDF_SIMILARITY,
    }
    return metric_flags.get(metric_name.lower(), True)


def get_output_config():
    """
    Get output configuration as a dictionary.
    
    Returns:
        dict: Output configuration
    """
    return {
        'verbose': VERBOSE_OUTPUT,
        'print_individual': PRINT_INDIVIDUAL_RESULTS,
        'decimal_places': METRIC_DECIMAL_PLACES,
        'include_full_answers': INCLUDE_FULL_ANSWERS_IN_JSON,
    }


def should_include_question(question_id):
    """
    Check if a question should be included in evaluation.
    
    Args:
        question_id: Question ID to check
    
    Returns:
        bool: True if should be included, False otherwise
    """
    # Check exclusion list
    if question_id in EXCLUDE_QUESTION_IDS:
        return False
    
    # Check inclusion list (if specified)
    if INCLUDE_QUESTION_IDS and question_id not in INCLUDE_QUESTION_IDS:
        return False
    
    return True
