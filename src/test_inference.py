"""
Inference Testing Script
========================
Standalone script for running automated inference tests on the RAG system.
Tests performance, logs results, and generates reports.

Usage:
    python test_inference.py --model InternVL3_5-2B-Q6_K --mode all
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
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import required modules
from hybrid_rag_module_qwen3 import HybridRAGQwen3_Module
from inference_logger import InferenceLogger
from llama_cpp import Llama

# Import shared configuration
from inference_config import (
    DEFAULT_DB_PATH,
    TEST_QUESTIONS_PATH,
    TOP_K_RESULTS,
    MODEL_CONFIG,
    DEFAULT_MAX_TOKENS,
    EMBEDDING_MODEL,
    get_system_message,
    get_model_config,
    get_available_models,
    PROMPT_TEMPLATE
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_test_questions(questions_path: str) -> List[Dict]:
    """Load test questions from JSON file."""
    try:
        with open(questions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('test_questions', [])
    except Exception as e:
        print(f"❌ Error loading test questions: {e}")
        return []


def create_session_log_file(model_name: str) -> Path:
    """
    Create a detailed log file for the current test session.
    
    Args:
        model_name: Name of the model being tested
    
    Returns:
        Path to the created log file
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = project_root / "data" / "test" / "logs" / "sessions"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"test_session_{model_name}_{timestamp}.log"
    return log_file


def write_log_header(log_file: Path, model_name: str, args):
    """
    Write header information to the log file.
    
    Args:
        log_file: Path to the log file
        model_name: Name of the model being tested
        args: Command line arguments
    """
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("INFERENCE TEST SESSION LOG\n")
        f.write("="*80 + "\n\n")
        
        # Session information
        f.write(f"Session Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Mode: {args.mode}\n")
        if args.mode == 'single':
            f.write(f"Question ID: {args.question_id}\n")
        f.write(f"\n")
        
        # Configuration from inference_config
        f.write("-"*80 + "\n")
        f.write("CONFIGURATION PARAMETERS\n")
        f.write("-"*80 + "\n\n")
        
        model_config = get_model_config(model_name)
        f.write(f"LLM Model Configuration:\n")
        f.write(f"  Model Path: {model_config['path']}\n")
        f.write(f"  Context Size (n_ctx): {model_config['n_ctx']}\n")
        f.write(f"  Temperature: {model_config['temperature']}\n")
        f.write(f"  Top P: {model_config['top_p']}\n")
        f.write(f"  GPU Layers: {model_config['n_gpu_layers']}\n")
        f.write(f"\n")
        
        f.write(f"RAG Configuration:\n")
        f.write(f"  Database Path: {DEFAULT_DB_PATH}\n")
        f.write(f"  Embedding Model: {EMBEDDING_MODEL}\n")
        f.write(f"  Top K Results: {TOP_K_RESULTS}\n")
        f.write(f"\n")
        
        f.write(f"Inference Settings:\n")
        f.write(f"  Max Tokens: {args.max_tokens}\n")
        f.write(f"  Default Max Tokens: {DEFAULT_MAX_TOKENS}\n")
        f.write(f"\n")
        
        f.write(f"System Message:\n")
        system_msg = get_system_message(model_name)
        for line in system_msg.split('\n'):
            f.write(f"  {line}\n")
        f.write(f"\n")
        
        f.write("="*80 + "\n\n")


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


def parse_thinking_response(response_text: str) -> Dict:
    """Parse LLM response to separate thinking process from final answer."""
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


def generate_llm_response(
    llm_model,
    query: str,
    context: str,
    model_name: str,
    max_tokens: int = 2048
) -> str:
    """Generate LLM response using retrieved context."""
    
    # Get system message from config
    system_message = get_system_message(model_name)
    
    # Build prompt using config template
    prompt = PROMPT_TEMPLATE.format(
        system_message=system_message,
        context=context,
        query=query
    )
    
    # Get model config for inference settings
    model_config = get_model_config(model_name)
    
    try:
        output = llm_model(
            prompt,
            max_tokens=max_tokens,
            temperature=model_config['temperature'],
            top_p=model_config['top_p'],
            echo=False
        )
        return output["choices"][0]["text"]
    except Exception as e:
        return f"Error generating response: {str(e)}"


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
    verbose: bool = True,
    log_file: Optional[Path] = None
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
    
    Returns:
        Dictionary with test results including raw_response and chunks
    """
    q_id = question['id']
    q_text = question['question']
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing Q{q_id}: {q_text}...")
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
            print(f"🔍 Searching knowledge base...")
        results = rag_system.search(query=q_text, top_k=TOP_K_RESULTS)
        
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
        error=error_msg
    )
    
    # Add raw response and chunks to log entry for session logging
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
    log_file: Optional[Path] = None
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
            verbose=True,
            log_file=log_file
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
        output_path = project_root / "data" / "test" / "logs" / f"inference_report_{timestamp}.xlsx"
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
        choices=['single', 'all'],
        help='Test mode: single question or all questions'
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
        '--show-stats',
        action='store_true',
        help='Show statistics from logged tests'
    )
    
    parser.add_argument(
        '--export-report',
        action='store_true',
        help='Export test results to Excel report'
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
    
    # Load LLM model
    print(f"\n🤖 Loading LLM model: {args.model}...")
    model_config = get_model_config(args.model)
    model_path = project_root / model_config['path']
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    try:
        llm_model = Llama(
            model_path=str(model_path),
            n_ctx=model_config['n_ctx'],
            n_gpu_layers=model_config['n_gpu_layers'],
            temperature=model_config['temperature'],
            verbose=model_config['verbose']
        )
        print(f"✅ Model loaded: {args.model}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Create session log file
    log_file = create_session_log_file(args.model)
    write_log_header(log_file, args.model, args)
    print(f"📄 Session log created: {log_file}")
    
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
            verbose=True,
            log_file=log_file
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
        
    elif args.mode == 'all':
        results = run_all_tests(
            rag_system=rag_system,
            llm_model=llm_model,
            model_name=args.model,
            questions=questions,
            logger=logger,
            max_tokens=args.max_tokens,
            log_file=log_file
        )
    
    print(f"\n✅ Tests complete. Results logged to: {logger.log_dir}")
    print(f"📄 Detailed session log: {log_file}")
    print(f"\nTo view statistics, run:")
    print(f"  python test_inference.py --show-stats --model {args.model}")
    print(f"\nTo export report, run:")
    print(f"  python test_inference.py --export-report")


if __name__ == "__main__":
    main()
