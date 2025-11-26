"""
Inference Testing Script
========================
Standalone script for running automated inference tests on the RAG system.
Tests performance, logs results, and generates reports.

Usage:
    python test_inference.py --model InternVL3_5-2B-Q6_K --mode all
    python test_inference.py --model Qwen3-8B-Q4_K_M --mode quick
    python test_inference.py --model Qwen3-8B-Q5_K_M --mode all
    python3 test_inference.py --model Qwen3-8B-Q5_K_M --mode all
    python test_inference.py --model Qwen3-8B-Q5_K_M --mode quick
    python3 test_inference.py --model Qwen3-8B-Q5_K_M --mode all --include-environment
    python test_inference.py --model Qwen3-8B-Q5_K_M --mode all --include-environment
    python test_inference.py --model InternVL3_5-2B-Q6_K --mode single --question-id 1
    python test_inference.py --show-stats
    python test_inference.py --export-report

Author: Generated for 01_RAG project
Date: November 6, 2025
"""

import sys
from pathlib import Path
import argparse
import json
import time
import re
import hashlib
from typing import Dict, List, Optional

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import required modules
from src.hybrid_rag_module_qwen3 import HybridRAGQwen3_Module
from src.inference_logger import InferenceLogger

# Import hardware information module
try:
    from hardware_info import get_all_hardware_info, format_hardware_info
    HARDWARE_INFO_AVAILABLE = True
except ImportError:
    HARDWARE_INFO_AVAILABLE = False
    print("⚠️  Warning: hardware_info module not available. Hardware information will not be logged.")

# Import environment collector module
try:
    from environment_collector import (
        get_filtered_environment_variables,
        get_python_environment,
        get_installed_packages,
        write_environment_report
    )
    ENVIRONMENT_COLLECTOR_AVAILABLE = True
except ImportError:
    ENVIRONMENT_COLLECTOR_AVAILABLE = False
    print("⚠️  Warning: environment_collector module not available. Environment information will not be logged.")

# Import shared configuration
from src.model_config import (
    DEFAULT_DB_PATH,
    TEST_QUESTIONS_PATH,
    TOP_K_RESULTS,
    MODEL_CONFIG,
    DEFAULT_MAX_TOKENS,
    EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    get_system_message,
    get_model_config,
    get_available_models,
    parse_thinking_response,
    load_llm_model,
    generate_llm_response,
    PROMPT_TEMPLATE,
    MAX_TOKENS_OPTIONS,
    SYSTEM_MESSAGE_INTERNVL,
    SYSTEM_MESSAGE_STANDARD,
    PROMPT_TEMPLATE_WITH_HISTORY,
    DEFAULT_MODEL
)

# Import RAG configuration
from src.rag_config import (
    CHUNK_STRATEGY,
    FIXED_SIZE_CHUNK_SIZE,
    FIXED_SIZE_OVERLAP,
    CHUNK_SIZE_MAX_BY_SENTENCE,
    EMBEDDING_DIMENSION,
    COLLECTION_NAME,
    BATCH_SIZE,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_THRESHOLD,
    SEMANTIC_WEIGHT,
    KEYWORD_WEIGHT,
    INITIAL_K_MULTIPLIER,
    INITIAL_K_CAP,
    KEYWORD_SCORING_METHOD,
    MAX_EMBEDDING_LENGTH,
    PADDING,
    TRUNCATION,
    get_db_path,
    get_device,
    get_torch_dtype
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file with normalized content.
    Removes all whitespace and newlines to ensure cross-platform compatibility
    (Windows uses CRLF, Linux uses LF).
    
    Args:
        file_path: Path to the file
    
    Returns:
        Hexadecimal hash string, or "FILE_NOT_FOUND" if file doesn't exist
    """
    if not file_path.exists():
        return "FILE_NOT_FOUND"
    
    try:
        # Read file content as text
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Remove all whitespace characters (spaces, tabs, newlines, carriage returns)
        normalized_content = ''.join(content.split())
        
        # Compute hash of normalized content
        sha256_hash = hashlib.sha256()
        sha256_hash.update(normalized_content.encode('utf-8'))
        return sha256_hash.hexdigest()
    except Exception as e:
        return f"ERROR: {str(e)}"


def get_python_file_hashes() -> Dict[str, str]:
    """
    Get hashes of all critical Python files in the project.
    
    Returns:
        Dictionary mapping file paths to their SHA256 hashes
    """
    critical_files = [
        # Core modules
        project_root / "src" / "model_config.py",
        project_root / "src" / "rag_config.py",
        project_root / "src" / "hybrid_rag_module_qwen3.py",
        project_root / "src" / "inference_logger.py",
        project_root / "src" / "data_pipeline_pdf.py",
        project_root / "src" / "chunk_qwen3_0_6B.py",
        project_root / "src" / "pre_chunking.py",
        project_root / "src" / "dashboard.py",
        # InternVL modules
        project_root / "src" / "intevl3_5" / "InternVL35_2B_reducedv2_single.py",
        project_root / "src" / "intevl3_5" / "InternVL35_4B_reducedv2_single.py",
        # Test files
        project_root / "tests" / "test_inference.py",
        project_root / "tests" / "test_similarity_filter.py",
        project_root / "tests" / "test_full.py",
        project_root / "tests" / "test_basics.py",
        project_root / "tests" / "test_reranking.py",
        project_root / "tests" / "hardware_info.py",
        project_root / "tests" / "environment_collector.py",
    ]
    
    file_hashes = {}
    for file_path in critical_files:
        relative_path = file_path.relative_to(project_root)
        file_hashes[str(relative_path)] = compute_file_hash(file_path)
    
    return file_hashes


def load_test_questions(questions_path: str) -> List[Dict]:
    """Load test questions from JSON file."""
    try:
        with open(questions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('test_questions', [])
    except Exception as e:
        print(f"❌ Error loading test questions: {e}")
        return []


def create_session_log_file(model_name: str, db_path: str = None, top_k: int = None, similarity_threshold: float = None, test_questions_path: str = None) -> tuple[Path, str]:
    """
    Create a detailed log file for the current test session.
    
    Args:
        model_name: Name of the model being tested
        db_path: Path to database (for tagging)
        top_k: Top K results parameter (for tagging)
        similarity_threshold: Similarity threshold (for tagging)
        test_questions_path: Path to test questions (for tagging)
    
    Returns:
        Tuple of (Path to the created log file, session name with tags)
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Use defaults if not provided
    db_path = db_path or DEFAULT_DB_PATH
    top_k = top_k or TOP_K_RESULTS
    similarity_threshold = similarity_threshold or SIMILARITY_THRESHOLD
    test_questions_path = test_questions_path or TEST_QUESTIONS_PATH
    
    # Create session name with configuration tags
    session_name = f"test_session_{model_name}_{timestamp}"
    
    # Store tags in metadata (will be used by Neptune uploader)
    session_tags = {
        'model': model_name,
        'db_path': db_path,
        'top_k': top_k,
        'similarity_threshold': similarity_threshold,
        'test_questions_path': test_questions_path
    }
    
    log_dir = project_root / "tests" / "logs" / "sessions"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{session_name}.log"
    
    # Write tags to a separate JSON file for Neptune uploader to read
    tags_file = log_dir / f"{session_name}_tags.json"
    with open(tags_file, 'w', encoding='utf-8') as f:
        json.dump(session_tags, f, indent=2)
    
    return log_file, session_name


def write_log_header(log_file: Path, model_name: str, args, include_environment: bool = False):
    """
    Write comprehensive header information to the log file.
    
    Args:
        log_file: Path to the log file
        model_name: Name of the model being tested
        args: Command line arguments
        include_environment: Include environment variables and Python environment info
    """
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("INFERENCE TEST SESSION LOG\n")
        f.write("="*100 + "\n\n")
        
        # =====================================================================
        # PROJECT INFORMATION
        # =====================================================================
        f.write("╔" + "═"*98 + "╗\n")
        f.write("║" + " "*40 + "PROJECT INFORMATION" + " "*39 + "║\n")
        f.write("╚" + "═"*98 + "╝\n\n")
        
        f.write(f"Project Name:           01_RAG (Hybrid RAG System)\n")
        f.write(f"Project Root:           {project_root}\n")
        f.write(f"Session Start Time:     {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Session ID:             {log_file.stem}\n")
        f.write(f"Test Mode:              {args.mode}\n")
        if args.mode == 'single':
            f.write(f"Question ID:            {args.question_id}\n")
        f.write(f"Selected Model:         {model_name}\n")
        f.write(f"\n")
        
        # =====================================================================
        # HARDWARE INFORMATION
        # =====================================================================
        if HARDWARE_INFO_AVAILABLE:
            f.write("╔" + "═"*98 + "╗\n")
            f.write("║" + " "*39 + "HARDWARE INFORMATION" + " "*39 + "║\n")
            f.write("╚" + "═"*98 + "╝\n\n")
            
            try:
                hw_info = get_all_hardware_info()
                
                # OS Information
                os_info = hw_info['os']
                f.write(f"Operating System:\n")
                f.write("-"*100 + "\n")
                f.write(f"  System:                 {os_info['system']}\n")
                f.write(f"  Release:                {os_info['release']}\n")
                f.write(f"  Version:                {os_info['version']}\n")
                f.write(f"  Platform:               {os_info['platform']}\n")
                f.write(f"  Python Version:         {os_info['python_version']}\n")
                f.write(f"  Python Implementation:  {os_info['python_implementation']}\n")
                f.write(f"\n")
                
                # CPU Information
                cpu_info = hw_info['cpu']
                f.write(f"CPU:\n")
                f.write("-"*100 + "\n")
                f.write(f"  Processor:              {cpu_info['processor']}\n")
                f.write(f"  Architecture:           {cpu_info['architecture']}\n")
                f.write(f"  Physical Cores:         {cpu_info['physical_cores']}\n")
                f.write(f"  Logical Cores:          {cpu_info['logical_cores']}\n")
                f.write(f"  Max Frequency:          {cpu_info['max_frequency']}\n")
                f.write(f"  Current Frequency:      {cpu_info['current_frequency']}\n")
                f.write(f"\n")
                
                # RAM Information
                ram_info = hw_info['ram']
                f.write(f"RAM:\n")
                f.write("-"*100 + "\n")
                f.write(f"  Total:                  {ram_info['total']}\n")
                f.write(f"  Available:              {ram_info['available']}\n")
                f.write(f"  Used:                   {ram_info['used']}\n")
                f.write(f"  Percent Used:           {ram_info['percent_used']}\n")
                f.write(f"\n")
                
                # GPU Information
                f.write(f"GPU(s):\n")
                f.write("-"*100 + "\n")
                for gpu in hw_info['gpus']:
                    f.write(f"  GPU {gpu['index']}:\n")
                    f.write(f"    Name:                 {gpu['name']}\n")
                    f.write(f"    Vendor:               {gpu.get('vendor', 'Unknown')}\n")
                    if gpu.get('driver_version') != 'N/A':
                        f.write(f"    Driver Version:       {gpu['driver_version']}\n")
                    f.write(f"    Memory Total:         {gpu['memory_total']}\n")
                    if gpu['memory_used'] != 'N/A':
                        f.write(f"    Memory Used:          {gpu['memory_used']}\n")
                        f.write(f"    Memory Free:          {gpu['memory_free']}\n")
                    if gpu.get('temperature') and gpu['temperature'] != 'N/A':
                        f.write(f"    Temperature:          {gpu['temperature']}\n")
                    if gpu.get('utilization') and gpu['utilization'] != 'N/A':
                        f.write(f"    Utilization:          {gpu['utilization']}\n")
                    if gpu.get('compute_capability'):
                        f.write(f"    Compute Capability:   {gpu['compute_capability']}\n")
                f.write(f"\n")
                
                # CUDA Information
                cuda_info = hw_info['cuda']
                f.write(f"CUDA:\n")
                f.write("-"*100 + "\n")
                f.write(f"  Available:              {cuda_info['available']}\n")
                f.write(f"  Version:                {cuda_info['version']}\n")
                f.write(f"  Device Count:           {cuda_info['device_count']}\n")
                if cuda_info['current_device'] != 'N/A':
                    f.write(f"  Current Device:         {cuda_info['current_device']}\n")
                f.write(f"\n")
                
                # Disk Information
                disk_info = hw_info['disk']
                f.write(f"Disk (Working Directory):\n")
                f.write("-"*100 + "\n")
                f.write(f"  Total:                  {disk_info['total']}\n")
                f.write(f"  Used:                   {disk_info['used']}\n")
                f.write(f"  Free:                   {disk_info['free']}\n")
                f.write(f"  Percent Used:           {disk_info['percent_used']}\n")
                f.write(f"\n")
                
            except Exception as e:
                f.write(f"⚠️  Error collecting hardware information: {str(e)}\n\n")
        else:
            f.write("⚠️  Hardware information module not available.\n\n")
        
        # =====================================================================
        # FILE INTEGRITY (SHA256 HASHES)
        # =====================================================================
        f.write("╔" + "═"*98 + "╗\n")
        f.write("║" + " "*36 + "FILE INTEGRITY HASHES" + " "*41 + "║\n")
        f.write("╚" + "═"*98 + "╝\n\n")
        
        f.write("Python Source File Hashes (SHA256) with python content normalization:\n")
        f.write("-"*100 + "\n")
        
        file_hashes = get_python_file_hashes()
        for file_path, file_hash in sorted(file_hashes.items()):
            f.write(f"  {file_path:<60} {file_hash}\n")
        f.write(f"\n")
        
        # =====================================================================
        # LLM MODEL CONFIGURATION (model_config.py)
        # =====================================================================
        f.write("╔" + "═"*98 + "╗\n")
        f.write("║" + " "*37 + "LLM MODEL CONFIGURATION" + " "*38 + "║\n")
        f.write("╚" + "═"*98 + "╝\n\n")
        
        model_config = get_model_config(model_name)
        f.write(f"Current Model Settings [{model_name}]:\n")
        f.write("-"*100 + "\n")
        f.write(f"  Model Path:             {model_config['path']}\n")
        f.write(f"  Context Size (n_ctx):   {model_config['n_ctx']}\n")
        f.write(f"  Temperature:            {model_config['temperature']}\n")
        f.write(f"  Top P:                  {model_config['top_p']}\n")
        if 'top_k' in model_config:
            f.write(f"  Top K:                  {model_config['top_k']}\n")
        if 'min_p' in model_config:
            f.write(f"  Min P:                  {model_config['min_p']}\n")
        if 'repeat_penalty' in model_config:
            f.write(f"  Repeat Penalty:         {model_config['repeat_penalty']}\n")
        if 'seed' in model_config:
            f.write(f"  Seed:                   {model_config['seed']} (for deterministic responses)\n")
        f.write(f"  GPU Layers:             {model_config['n_gpu_layers']}\n")
        f.write(f"  Verbose:                {model_config['verbose']}\n")
        f.write(f"\n")
        
        f.write(f"All Available Models:\n")
        f.write("-"*100 + "\n")
        for idx, available_model in enumerate(get_available_models(), 1):
            marker = " ← SELECTED" if available_model == model_name else ""
            f.write(f"  {idx:2}. {available_model}{marker}\n")
        f.write(f"\n")
        
        f.write(f"Default Model:          {DEFAULT_MODEL}\n")
        f.write(f"\n")
        
        # =====================================================================
        # INFERENCE CONFIGURATION (model_config.py)
        # =====================================================================
        f.write("╔" + "═"*98 + "╗\n")
        f.write("║" + " "*38 + "INFERENCE CONFIGURATION" + " "*37 + "║\n")
        f.write("╚" + "═"*98 + "╝\n\n")
        
        f.write(f"Token Settings:\n")
        f.write("-"*100 + "\n")
        f.write(f"  Max Tokens (Current):   {args.max_tokens}\n")
        f.write(f"  Default Max Tokens:     {DEFAULT_MAX_TOKENS}\n")
        f.write(f"  Available Options:      {MAX_TOKENS_OPTIONS}\n")
        f.write(f"\n")
        
        f.write(f"Test Configuration:\n")
        f.write("-"*100 + "\n")
        f.write(f"  Test Questions Path:    {TEST_QUESTIONS_PATH}\n")
        f.write(f"\n")
        
        # =====================================================================
        # RAG SYSTEM CONFIGURATION (rag_config.py)
        # =====================================================================
        f.write("╔" + "═"*98 + "╗\n")
        f.write("║" + " "*38 + "RAG SYSTEM CONFIGURATION" + " "*36 + "║\n")
        f.write("╚" + "═"*98 + "╝\n\n")
        
        f.write(f"Database Configuration:\n")
        f.write("-"*100 + "\n")
        f.write(f"  Database Path:          {args.db_path}\n")
        f.write(f"  Default DB Path:        {DEFAULT_DB_PATH}\n")
        f.write(f"  Collection Name:        {COLLECTION_NAME}\n")
        f.write(f"\n")
        
        f.write(f"Embedding Configuration:\n")
        f.write("-"*100 + "\n")
        f.write(f"  Embedding Model:        {EMBEDDING_MODEL}\n")
        f.write(f"  Embedding Dimension:    {EMBEDDING_DIMENSION}\n")
        f.write(f"  Max Embedding Length:   {MAX_EMBEDDING_LENGTH} tokens\n")
        f.write(f"  Device:                 {get_device()}\n")
        f.write(f"  Torch Dtype:            {get_torch_dtype()}\n")
        f.write(f"  Padding:                {PADDING}\n")
        f.write(f"  Truncation:             {TRUNCATION}\n")
        f.write(f"  Batch Size:             {BATCH_SIZE}\n")
        f.write(f"\n")
        
        f.write(f"Chunking Configuration:\n")
        f.write("-"*100 + "\n")
        f.write(f"  Chunk Strategy:         {CHUNK_STRATEGY}\n")
        if CHUNK_STRATEGY == "fixed_size":
            f.write(f"  Fixed Chunk Size:       {FIXED_SIZE_CHUNK_SIZE} chars\n")
            f.write(f"  Fixed Overlap:          {FIXED_SIZE_OVERLAP} chars\n")
        f.write(f"  Sentence Max Size:      {CHUNK_SIZE_MAX_BY_SENTENCE} chars\n")
        f.write(f"\n")
        
        f.write(f"Retrieval Configuration:\n")
        f.write("-"*100 + "\n")
        f.write(f"  Top K Results:          {TOP_K_RESULTS}\n")
        f.write(f"  Default Top K:          {DEFAULT_TOP_K}\n")
        f.write(f"  Min Similarity:         {MIN_SIMILARITY_THRESHOLD}%\n")
        f.write(f"\n")
        
        f.write(f"Hybrid Search Configuration:\n")
        f.write("-"*100 + "\n")
        f.write(f"  Semantic Weight:        {SEMANTIC_WEIGHT}\n")
        f.write(f"  Keyword Weight:         {KEYWORD_WEIGHT}\n")
        f.write(f"  Keyword Scoring:        {KEYWORD_SCORING_METHOD}\n")
        f.write(f"  Initial K Multiplier:   {INITIAL_K_MULTIPLIER}\n")
        f.write(f"  Initial K Cap:          {INITIAL_K_CAP}\n")
        f.write(f"\n")
        
        # =====================================================================
        # PROMPT TEMPLATES
        # =====================================================================
        f.write("╔" + "═"*98 + "╗\n")
        f.write("║" + " "*41 + "PROMPT TEMPLATES" + " "*41 + "║\n")
        f.write("╚" + "═"*98 + "╝\n\n")
        
        f.write(f"System Message (InternVL Models):\n")
        f.write("-"*100 + "\n")
        for line in SYSTEM_MESSAGE_INTERNVL.split('\n'):
            f.write(f"  {line}\n")
        f.write(f"\n")
        
        f.write(f"System Message (Standard Models):\n")
        f.write("-"*100 + "\n")
        for line in SYSTEM_MESSAGE_STANDARD.split('\n'):
            f.write(f"  {line}\n")
        f.write(f"\n")
        
        f.write(f"Current Model System Message:\n")
        f.write("-"*100 + "\n")
        system_msg = get_system_message(model_name)
        for line in system_msg.split('\n'):
            f.write(f"  {line}\n")
        f.write(f"\n")
        
        f.write(f"Base Prompt Template:\n")
        f.write("-"*100 + "\n")
        for line in PROMPT_TEMPLATE.split('\n'):
            f.write(f"  {line}\n")
        f.write(f"\n")
        
        # =====================================================================
        # ENVIRONMENT INFORMATION (Optional)
        # =====================================================================
        if include_environment and ENVIRONMENT_COLLECTOR_AVAILABLE:
            f.write("╔" + "═"*98 + "╗\n")
            f.write("║" + " "*35 + "ENVIRONMENT INFORMATION" + " "*40 + "║\n")
            f.write("╚" + "═"*98 + "╝\n\n")
            
            # Environment Variables
            f.write("Environment Variables (Sensitive values redacted):\n")
            f.write("-"*100 + "\n")
            env_vars = get_filtered_environment_variables()
            for key, value in sorted(env_vars.items()):
                # Truncate very long values
                if len(value) > 150:
                    value = value[:147] + "..."
                f.write(f"  {key:<45} = {value}\n")
            f.write(f"\n")
            
            # Python Environment
            py_env = get_python_environment()
            f.write("Python Environment Details:\n")
            f.write("-"*100 + "\n")
            f.write(f"  Python Version:         {py_env['python_version'].split()[0]}\n")
            f.write(f"  Python Implementation:  {py_env['python_implementation']}\n")
            f.write(f"  Python Compiler:        {py_env['python_compiler']}\n")
            f.write(f"  Platform:               {py_env['platform']}\n")
            f.write(f"  Executable:             {py_env['executable']}\n")
            f.write(f"  Is Virtual Env:         {py_env['is_virtual_env']}\n")
            if py_env.get('virtual_env_type'):
                f.write(f"  Virtual Env Type:       {py_env['virtual_env_type']}\n")
            if py_env.get('virtual_env_path'):
                f.write(f"  Virtual Env Path:       {py_env['virtual_env_path']}\n")
            f.write(f"  Default Encoding:       {py_env['default_encoding']}\n")
            f.write(f"\n")
            
            # Python Path
            f.write("Python Path (sys.path):\n")
            f.write("-"*100 + "\n")
            for i, path in enumerate(py_env['path'], 1):
                f.write(f"  {i:2}. {path}\n")
            f.write(f"\n")
            
            # Installed Packages Summary
            packages = get_installed_packages()
            f.write(f"Installed Packages: {len(packages)} total\n")
            f.write("-"*100 + "\n")
            f.write("Key packages relevant to this project:\n")
            
            # List important packages
            important_packages = [
                'torch', 'transformers', 'chromadb', 'streamlit', 'llama-cpp-python',
                'numpy', 'pandas', 'pillow', 'pytest', 'openpyxl'
            ]
            
            for pkg in packages:
                if pkg['name'].lower() in important_packages:
                    f.write(f"  {pkg['name']:<30} {pkg['version']}\n")
            f.write(f"\n")
            
            f.write("Note: Full package list and pip freeze output saved to separate environment report file.\n")
            f.write(f"\n")
        
        f.write("="*100 + "\n")
        f.write("END OF HEADER - TEST RESULTS BEGIN BELOW\n")
        f.write("="*100 + "\n\n")


def append_test_result(log_file: Path, question: Dict, result: Dict, raw_response: str, chunks: List[Dict]):
    """
    Append a test result to the log file.
    
    Args:
        log_file: Path to the log file
        question: Question dictionary
        result: Test result dictionary from logger
        raw_response: Raw LLM response (including thinking tags)
        chunks: Retrieved chunks from RAG
    """
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"QUESTION {question['id']}\n")
        f.write("="*80 + "\n\n")
        
        # Question details
        f.write(f"Category: {question.get('category', 'N/A')}\n")
        f.write(f"Tags: {', '.join(question.get('tags', []))}\n")
        f.write(f"\n")
        
        f.write(f"Question:\n")
        f.write(f"{question['question']}\n")
        f.write(f"\n")
        
        f.write("-"*80 + "\n")
        f.write("RAW LLM OUTPUT (Including Thinking Process)\n")
        f.write("-"*80 + "\n\n")
        f.write(f"{raw_response}\n")
        f.write(f"\n")
        
        # Parsed components
        f.write("-"*80 + "\n")
        f.write("PARSED COMPONENTS\n")
        f.write("-"*80 + "\n\n")
        
        if result.get('has_thinking') and result.get('thinking'):
            f.write(f"Thinking Process:\n")
            f.write(f"{result['thinking']}\n")
            f.write(f"\n")
        
        f.write(f"Final Answer:\n")
        f.write(f"{result['answer']}\n")
        f.write(f"\n")
        
        # Performance metrics
        f.write("-"*80 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Response Time: {result['response_time_seconds']:.2f} seconds\n")
        f.write(f"Chunks Retrieved: {result['num_chunks_retrieved']}\n")
        f.write(f"Answer Length: {result['answer_length']} characters\n")
        if result.get('has_thinking'):
            f.write(f"Thinking Length: {result.get('thinking_length', 0)} characters\n")
        f.write(f"Success: {result['success']}\n")
        if not result['success']:
            f.write(f"Error: {result.get('error_message', 'Unknown')}\n")
        f.write(f"\n")
        
        # Retrieved chunks
        f.write("-"*80 + "\n")
        f.write(f"RETRIEVED CHUNKS (Top {len(chunks)})\n")
        f.write("-"*80 + "\n\n")
        
        for idx, chunk in enumerate(chunks, 1):
            f.write(f"Chunk {idx}:\n")
            f.write(f"  Source: {chunk['metadata'].get('source', 'Unknown')}\n")
            f.write(f"  Chunk Index: {chunk['metadata'].get('chunk_index', 'N/A')}\n")
            f.write(f"  Similarity Score: {chunk['similarity_score']:.1f}%\n")
            f.write(f"  Content Preview: {chunk['content']}...\n")
            f.write(f"\n")
        
        f.write("\n")


def write_log_footer(log_file: Path, stats: Dict, total_time: float):
    """
    Write session summary to the log file.
    
    Args:
        log_file: Path to the log file
        stats: Statistics dictionary
        total_time: Total session time in seconds
    """
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SESSION SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Session End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Session Duration: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        f.write(f"\n")
        
        f.write("-"*80 + "\n")
        f.write("TEST STATISTICS\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"Total Tests Run: {stats.get('total_tests', 0)}\n")
        f.write(f"Successful: {stats.get('successful', 0)}\n")
        f.write(f"Failed: {stats.get('failed', 0)}\n")
        f.write(f"Success Rate: {stats.get('success_rate', 0):.1f}%\n")
        f.write(f"\n")
        
        if stats.get('successful', 0) > 0:
            f.write(f"Response Time Statistics:\n")
            f.write(f"  Average: {stats.get('avg_time', 0):.2f} seconds\n")
            f.write(f"  Minimum: {stats.get('min_time', 0):.2f} seconds\n")
            f.write(f"  Maximum: {stats.get('max_time', 0):.2f} seconds\n")
            f.write(f"\n")
            
            f.write(f"Chunk Statistics:\n")
            f.write(f"  Average Chunks Retrieved: {stats.get('avg_chunks', 0):.1f}\n")
            f.write(f"\n")
            
            f.write(f"Answer Statistics:\n")
            f.write(f"  Average Answer Length: {stats.get('avg_answer_length', 0):.0f} characters\n")
        
        f.write(f"\n")
        f.write("="*80 + "\n")
        f.write("END OF SESSION LOG\n")
        f.write("="*80 + "\n")


# =============================================================================
# TEST EXECUTION FUNCTIONS
# =============================================================================

def run_single_test(
    rag_system,
    llm_model,
    model_name: str,
    question: Dict,
    logger: InferenceLogger,
    max_tokens: int = 2048,
    top_k: int = TOP_K_RESULTS,
    verbose: bool = True,
    log_file: Optional[Path] = None,
    session_name: Optional[str] = None
) -> Dict:
    """
    Run a single inference test.
    
    Args:
        rag_system: The RAG system instance
        llm_model: The LLM model instance
        model_name: Name of the model being used
        question: Question dictionary with 'id', 'question', etc.
        logger: InferenceLogger instance
        max_tokens: Maximum tokens for response
        verbose: Print progress messages
        log_file: Optional path to session log file
        session_name: Optional session name for grouping tests
    
    Returns:
        Dictionary with test results including raw_response and chunks
    """
    q_id = question['id']
    q_text = question['question']
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing Q{q_id}: {q_text}")
        print(f"{'='*80}")
    
    start_time = time.time()
    error_msg = None
    answer = None
    thinking = None
    results = None
    raw_response = None
    
    try:
        # Search for relevant chunks
        if verbose:
            print(f"🔍 Searching knowledge base (top_k={top_k})...")
        results = rag_system.search(query=q_text, top_k=top_k)
        
        if verbose:
            print(f"✓ Found {len(results)} relevant chunks")
        
        # Format context for LLM
        context = rag_system.format_for_llm(results, max_chunks=None)
        
        # Generate response
        if verbose:
            print(f"🤖 Generating response with {model_name}...")
        raw_response = generate_llm_response(llm_model, q_text, context, model_name, max_tokens)
        
        # Parse response
        parsed_response = parse_thinking_response(raw_response)
        answer = parsed_response['answer']
        thinking = parsed_response.get('thinking')
        
        if verbose:
            if parsed_response['has_thinking']:
                print(f"✓ Response generated with thinking process")
            else:
                print(f"✓ Response generated")
        
    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"❌ Error: {error_msg}")
    
    response_time = time.time() - start_time
    
    # Log the inference
    log_entry = logger.log_inference(
        question_id=q_id,
        question=q_text,
        answer=answer or "",
        model_name=model_name,
        response_time=response_time,
        num_chunks_retrieved=len(results) if results else 0,
        thinking=thinking,
        sources=results,
        error=error_msg,
        session_name=session_name
    )
    
    # Add other data to log entry for session logging
    log_entry['raw_response'] = raw_response or ""
    log_entry['chunks'] = results or []
    
    # Append to session log file if provided
    if log_file and log_file.exists():
        append_test_result(log_file, question, log_entry, raw_response or "", results or [])
    
    if verbose:
        print(f"\n📊 Results:")
        print(f"   ⏱️  Response time: {response_time:.2f}s")
        print(f"   📚 Chunks used: {len(results) if results else 0}")
        print(f"   ✅ Success: {error_msg is None}")
        
        if answer and not error_msg:
            print(f"\n💬 Answer ({len(answer)} chars):")
            print(f"   {answer}")
    
    return log_entry


def run_all_tests(
    rag_system,
    llm_model,
    model_name: str,
    questions: List[Dict],
    logger: InferenceLogger,
    max_tokens: int = 2048,
    top_k: int = TOP_K_RESULTS,
    log_file: Optional[Path] = None,
    session_name: Optional[str] = None
) -> List[Dict]:
    """
    Run all inference tests.
    
    Args:
        rag_system: The RAG system instance
        llm_model: The LLM model instance
        model_name: Name of the model being used
        questions: List of question dictionaries
        logger: InferenceLogger instance
        max_tokens: Maximum tokens for response
        log_file: Optional path to session log file
        session_name: Optional session name for grouping tests
    
    Returns:
        List of test result dictionaries
    """
    print(f"\n{'='*80}")
    print(f"Running {len(questions)} inference tests with {model_name}")
    print(f"{'='*80}\n")
    
    results = []
    start_time = time.time()
    
    for idx, question in enumerate(questions, 1):
        print(f"\n[{idx}/{len(questions)}] ", end="")
        result = run_single_test(
            rag_system=rag_system,
            llm_model=llm_model,
            model_name=model_name,
            question=question,
            logger=logger,
            max_tokens=max_tokens,
            top_k=top_k,
            verbose=True,
            log_file=log_file,
            session_name=session_name
        )
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"TEST BATCH COMPLETE")
    print(f"{'='*80}")
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    avg_time = sum(r['response_time_seconds'] for r in results if r['success']) / successful if successful > 0 else 0
    
    # Calculate statistics for log footer
    stats = {
        'total_tests': len(results),
        'successful': successful,
        'failed': failed,
        'success_rate': (successful / len(results) * 100) if results else 0,
        'avg_time': avg_time,
        'min_time': min(r['response_time_seconds'] for r in results if r['success']) if successful > 0 else 0,
        'max_time': max(r['response_time_seconds'] for r in results if r['success']) if successful > 0 else 0,
        'avg_chunks': sum(r['num_chunks_retrieved'] for r in results if r['success']) / successful if successful > 0 else 0,
        'avg_answer_length': sum(r['answer_length'] for r in results if r['success']) / successful if successful > 0 else 0
    }
    
    # Write log footer if log file provided
    if log_file and log_file.exists():
        write_log_footer(log_file, stats, total_time)
        print(f"\n📄 Detailed session log saved to: {log_file}")
    
    print(f"\n📊 Summary:")
    print(f"   Total tests: {len(results)}")
    print(f"   ✅ Successful: {successful} ({successful/len(results)*100:.1f}%)")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⏱️  Avg response time: {avg_time:.2f}s")
    print(f"   🕐 Total time: {total_time:.2f}s")
    
    return results


def show_statistics(logger: InferenceLogger, model_name: Optional[str] = None):
    """Display statistics from logged tests."""
    print(f"\n{'='*80}")
    print(f"INFERENCE PERFORMANCE STATISTICS")
    if model_name:
        print(f"Model: {model_name}")
    print(f"{'='*80}\n")
    
    stats = logger.get_statistics(model_name=model_name)
    
    if stats['total_inferences'] == 0:
        print("No inference tests logged yet.")
        return
    
    print(f"📊 Overall Statistics:")
    print(f"   Total inferences: {stats['total_inferences']}")
    print(f"   Successful: {stats['successful_inferences']}")
    print(f"   Failed: {stats['failed_inferences']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"\n⏱️  Response Time:")
    print(f"   Average: {stats['avg_response_time']:.2f}s")
    print(f"   Min: {stats['min_response_time']:.2f}s")
    print(f"   Max: {stats['max_response_time']:.2f}s")
    print(f"\n📚 Chunks:")
    print(f"   Average retrieved: {stats['avg_chunks_retrieved']:.1f}")
    print(f"\n💬 Answers:")
    print(f"   Average length: {stats['avg_answer_length']:.0f} chars")
    
    # Model comparison
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON")
    print(f"{'='*80}\n")
    
    df = logger.get_model_comparison()
    if not df.empty:
        print(f"{'Model':<40} {'Tests':<8} {'Success%':<10} {'Avg Time':<10}")
        print(f"{'-'*80}")
        for _, row in df.iterrows():
            print(f"{row['model_name']:<40} {row['total_inferences']:<8} {row['success_rate']:<9.1f}% {row['avg_response_time']:<9.2f}s")
    else:
        print("No model comparison data available.")


def export_report(logger: InferenceLogger, output_path: Optional[str] = None):
    """Export test results to Excel report."""
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = project_root / "tests" / "logs" / f"inference_report_{timestamp}.xlsx"
    else:
        output_path = Path(output_path)
    
    try:
        logger.export_to_excel(str(output_path))
        print(f"\n✅ Report exported to: {output_path}")
    except Exception as e:
        print(f"\n❌ Export failed: {e}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point for the inference testing script."""
    parser = argparse.ArgumentParser(
        description="Run inference tests on the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with a specific model
  python test_inference.py --model InternVL3_5-2B-Q6_K --mode all
  
  # Run all tests with environment information included
  python test_inference.py --model InternVL3_5-2B-Q6_K --mode all --include-environment
  
  # Run quick test (5 selected questions)
  python test_inference.py --model InternVL3_5-2B-Q6_K --mode quick
  
  # Run a single test
  python test_inference.py --model InternVL3_5-2B-Q6_K --mode single --question-id 1
  
  # Show statistics
  python test_inference.py --show-stats
  
  # Show statistics for specific model
  python test_inference.py --show-stats --model InternVL3_5-2B-Q6_K
  
  # Export report
  python test_inference.py --export-report
        """
    )
    
    parser.add_argument(
        '--model',
        choices=get_available_models(),
        default=None,
        help='LLM model to use for testing'
    )
    
    parser.add_argument(
        '--mode',
        choices=['single', 'all', 'quick'],
        help='Test mode: single question, all questions, or quick test (5 questions)'
    )
    
    parser.add_argument(
        '--question-id',
        type=int,
        help='Question ID to test (for single mode)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f'Maximum tokens for LLM response (default: {DEFAULT_MAX_TOKENS})'
    )
    
    parser.add_argument(
        '--db-path',
        default=DEFAULT_DB_PATH,
        help='Path to ChromaDB database'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=TOP_K_RESULTS,
        help=f'Number of chunks to retrieve from RAG (default: {TOP_K_RESULTS})'
    )
    
    parser.add_argument(
        '--show-stats',
        action='store_true',
        help='Show statistics from logged tests'
    )
    
    parser.add_argument(
        '--export-report',
        action='store_true',
        help='Export test results to Excel report'
    )
    
    parser.add_argument(
        '--include-environment',
        action='store_true',
        help='Include environment variables and Python environment in session log'
    )
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = InferenceLogger()
    
    # Handle stats display
    if args.show_stats:
        show_statistics(logger, args.model)
        return
    
    # Handle report export
    if args.export_report:
        export_report(logger)
        return
    
    # Validate arguments for running tests
    if not args.model or not args.mode:
        parser.error("--model and --mode are required for running tests")
    
    if args.mode == 'single' and args.question_id is None:
        parser.error("--question-id is required for single mode")
    
    # Load test questions
    questions_path = project_root / TEST_QUESTIONS_PATH
    questions = load_test_questions(str(questions_path))
    
    if not questions:
        print("❌ No test questions loaded. Exiting.")
        sys.exit(1)
    
    print(f"✓ Loaded {len(questions)} test questions")
    
    # Initialize RAG system
    print(f"\n🔧 Initializing RAG system...")
    db_path = project_root / args.db_path
    
    try:
        rag_system = HybridRAGQwen3_Module(
            embedding_model=EMBEDDING_MODEL,
            db_path=str(db_path),
            device='cuda',
            verbose=False
        )
        print(f"✅ RAG system initialized")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        sys.exit(1)
    
    # Load LLM model using shared function
    print(f"\n🤖 Loading LLM model: {args.model}...")
    try:
        llm_model = load_llm_model(args.model, project_root)
        print(f"✅ Model loaded: {args.model}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Create session log file
    log_file, session_name = create_session_log_file(
        model_name=args.model,
        db_path=args.db_path,
        top_k=args.top_k,
        similarity_threshold=SIMILARITY_THRESHOLD,
        test_questions_path=TEST_QUESTIONS_PATH
    )
    write_log_header(log_file, args.model, args, include_environment=args.include_environment)
    print(f"📄 Session log created: {log_file}")
    print(f"📋 Session name: {session_name}")
    
    # Create separate environment report if requested
    if args.include_environment and ENVIRONMENT_COLLECTOR_AVAILABLE:
        env_report_file = log_file.parent / f"{log_file.stem}_environment.txt"
        print(f"📝 Generating detailed environment report...")
        write_environment_report(
            output_path=env_report_file,
            include_env_vars=True,
            include_python_env=True,
            include_packages=True,
            include_pip_freeze=True,
            filter_sensitive=True
        )
        print(f"✅ Environment report saved: {env_report_file}")
    
    # Run tests
    if args.mode == 'single':
        # Find the question
        question = next((q for q in questions if q['id'] == args.question_id), None)
        if not question:
            print(f"❌ Question ID {args.question_id} not found")
            sys.exit(1)
        
        session_start = time.time()
        result = run_single_test(
            rag_system=rag_system,
            llm_model=llm_model,
            model_name=args.model,
            question=question,
            logger=logger,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            verbose=True,
            log_file=log_file,
            session_name=session_name
        )
        session_time = time.time() - session_start
        
        # Write footer for single test
        stats = {
            'total_tests': 1,
            'successful': 1 if result['success'] else 0,
            'failed': 0 if result['success'] else 1,
            'success_rate': 100.0 if result['success'] else 0.0,
            'avg_time': result['response_time_seconds'] if result['success'] else 0,
            'min_time': result['response_time_seconds'] if result['success'] else 0,
            'max_time': result['response_time_seconds'] if result['success'] else 0,
            'avg_chunks': result['num_chunks_retrieved'] if result['success'] else 0,
            'avg_answer_length': result['answer_length'] if result['success'] else 0
        }
        write_log_footer(log_file, stats, session_time)
        
    elif args.mode == 'quick':
        # Quick test mode: run only selected questions (1, 5, 10, 11, 14)
        quick_test_ids = [1, 5, 10, 11, 14]
        quick_questions = [q for q in questions if q['id'] in quick_test_ids]
        
        if len(quick_questions) != len(quick_test_ids):
            found_ids = [q['id'] for q in quick_questions]
            missing_ids = [qid for qid in quick_test_ids if qid not in found_ids]
            print(f"⚠️  Warning: Could not find questions with IDs: {missing_ids}")
        
        print(f"\n🚀 Quick test mode: Running {len(quick_questions)} selected questions")
        print(f"   Question IDs: {[q['id'] for q in quick_questions]}")
        
        results = run_all_tests(
            rag_system=rag_system,
            llm_model=llm_model,
            model_name=args.model,
            questions=quick_questions,
            logger=logger,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            log_file=log_file,
            session_name=session_name
        )
        
    elif args.mode == 'all':
        results = run_all_tests(
            rag_system=rag_system,
            llm_model=llm_model,
            model_name=args.model,
            questions=questions,
            logger=logger,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            log_file=log_file,
            session_name=session_name
        )
    
    print(f"\n✅ Tests complete. Results logged to: {logger.log_dir}")
    print(f"📄 Detailed session log: {log_file}")
    print(f"\nTo view statistics, run:")
    print(f"  python test_inference.py --show-stats --model {args.model}")
    print(f"\nTo export report, run:")
    print(f"  python test_inference.py --export-report")


if __name__ == "__main__":
    main()
