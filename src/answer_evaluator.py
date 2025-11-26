"""
Answer Quality Evaluation Module

Evaluates the quality of RAG-generated answers by comparing them against
reference answers using multiple text similarity metrics.

This module focuses on answer quality rather than retrieval quality.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

# Import configuration
try:
    from output_evaluator_config import (
        get_inference_log_index,
        get_num_inference_eval,
        get_max_questions,
        get_embedding_model,
        should_skip_missing_reference,
        should_skip_missing_generated,
        is_metric_enabled,
        get_output_config,
        should_include_question,
        ROUGE_USE_STEMMER,
        BLEU_SMOOTHING_METHOD,
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("[WARNING] Configuration file not found, using defaults")

# Text similarity metrics
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class AnswerMetrics:
    """Container for answer quality metrics."""
    question_id: int
    question: str
    reference_answer: str
    generated_answer: str
    model_name: str
    
    # Semantic similarity (0-1, higher is better)
    semantic_similarity: Optional[float] = None
    
    # ROUGE scores (0-1, higher is better)
    rouge_1_f: Optional[float] = None
    rouge_1_p: Optional[float] = None
    rouge_1_r: Optional[float] = None
    rouge_2_f: Optional[float] = None
    rouge_2_p: Optional[float] = None
    rouge_2_r: Optional[float] = None
    rouge_l_f: Optional[float] = None
    rouge_l_p: Optional[float] = None
    rouge_l_r: Optional[float] = None
    
    # BLEU score (0-1, higher is better)
    bleu_score: Optional[float] = None
    
    # TF-IDF cosine similarity (0-1, higher is better)
    tfidf_similarity: Optional[float] = None
    
    # Answer length metrics
    reference_length: int = 0
    generated_length: int = 0
    length_ratio: float = 0.0
    
    # Response time
    response_time_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'question_id': self.question_id,
            'question': self.question,
            'reference_answer': self.reference_answer,
            'generated_answer': self.generated_answer,
            'model_name': self.model_name,
            'reference_length': self.reference_length,
            'generated_length': self.generated_length,
            'length_ratio': self.length_ratio,
            'response_time_seconds': self.response_time_seconds,
            'semantic_similarity': self.semantic_similarity,
            'rouge_1_f': self.rouge_1_f,
            'rouge_1_p': self.rouge_1_p,
            'rouge_1_r': self.rouge_1_r,
            'rouge_2_f': self.rouge_2_f,
            'rouge_2_p': self.rouge_2_p,
            'rouge_2_r': self.rouge_2_r,
            'rouge_l_f': self.rouge_l_f,
            'rouge_l_p': self.rouge_l_p,
            'rouge_l_r': self.rouge_l_r,
            'bleu_score': self.bleu_score,
            'tfidf_similarity': self.tfidf_similarity,
        }


class AnswerEvaluator:
    """
    Evaluates RAG answer quality using multiple text similarity metrics.
    """
    
    def __init__(
        self,
        gold_standard_path: str,
        inference_log_path: str,
        embedding_model: str = None,
        inference_index: int = None,
        max_questions: int = None,
        num_inference_eval: int = None
    ):
        """
        Initialize the answer evaluator.
        
        Args:
            gold_standard_path: Path to gold standard JSON file
            inference_log_path: Path to inference log JSONL file
            embedding_model: Sentence transformer model for semantic similarity (default from config or Qwen3-Embedding-0.6B)
            inference_index: Which inference log to use (-1 for last, 0 for first, default from config)
            max_questions: Maximum number of questions to evaluate (None for all, default from config)
            num_inference_eval: Number of inference sessions to evaluate starting from last backwards (default from config or 1)
        """
        # Apply defaults from config if available
        if embedding_model is None:
            embedding_model = get_embedding_model() if CONFIG_AVAILABLE else "Qwen/Qwen3-Embedding-0.6B"
        if inference_index is None:
            inference_index = get_inference_log_index() if CONFIG_AVAILABLE else -1
        if max_questions is None:
            max_questions = get_max_questions() if CONFIG_AVAILABLE else None
        if num_inference_eval is None:
            num_inference_eval = get_num_inference_eval() if CONFIG_AVAILABLE else 1
        
        self.inference_index = inference_index
        self.max_questions = max_questions
        self.num_inference_eval = num_inference_eval
        self.gold_standard_path = Path(gold_standard_path)
        self.inference_log_path = Path(inference_log_path)
        self.embedding_model_name = embedding_model
        
        # Load data
        self.gold_standard = self._load_gold_standard()
        self.inference_logs = self._load_inference_logs()
        
        # Initialize models
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading embedding model: {embedding_model}")
                self.embedding_model = SentenceTransformer(embedding_model)
                print("[OK] Embedding model loaded")
            except Exception as e:
                print(f"[WARNING] Failed to load embedding model: {e}")
        
        self.rouge_scorer = None
        if ROUGE_AVAILABLE and (not CONFIG_AVAILABLE or is_metric_enabled('rouge')):
            use_stemmer = ROUGE_USE_STEMMER if CONFIG_AVAILABLE else True
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=use_stemmer
            )
            print("[OK] ROUGE scorer initialized")
        
        if NLTK_AVAILABLE:
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
            print("[OK] NLTK initialized")
        
        if SKLEARN_AVAILABLE:
            print("[OK] Scikit-learn available for TF-IDF")
    
    def _load_gold_standard(self) -> Dict[int, Dict]:
        """Load gold standard dataset."""
        with open(self.gold_standard_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Index by question_id
        gold_standard = {}
        for entry in data.get('validation_entries', []):
            qid = entry.get('question_id')
            if qid:
                gold_standard[qid] = entry
        
        print(f"Loaded {len(gold_standard)} reference answers from gold standard")
        return gold_standard
    
    def _load_inference_logs(self) -> Dict[int, List[Dict]]:
        """Load inference logs."""
        inference_logs = defaultdict(list)
        
        with open(self.inference_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    log_entry = json.loads(line)
                    qid = log_entry.get('question_id')
                    if qid:
                        inference_logs[qid].append(log_entry)
        
        total_logs = sum(len(logs) for logs in inference_logs.values())
        print(f"Loaded {total_logs} inference logs for {len(inference_logs)} questions")
        return inference_logs
    
    def calculate_semantic_similarity(
        self,
        reference: str,
        generated: str
    ) -> Optional[float]:
        """
        Calculate semantic similarity using sentence embeddings.
        
        Returns:
            Cosine similarity score (0-1), or None if model unavailable
        """
        if not self.embedding_model or (CONFIG_AVAILABLE and not is_metric_enabled('semantic')):
            return None
        
        try:
            # Encode both texts
            ref_embedding = self.embedding_model.encode(reference, convert_to_tensor=True)
            gen_embedding = self.embedding_model.encode(generated, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = util.cos_sim(ref_embedding, gen_embedding).item()
            return float(similarity)
        except Exception as e:
            print(f"[WARNING] Error calculating semantic similarity: {e}")
            return None
    
    def calculate_rouge_scores(
        self,
        reference: str,
        generated: str
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
        """
        if not self.rouge_scorer or (CONFIG_AVAILABLE and not is_metric_enabled('rouge')):
            return {}
        
        try:
            scores = self.rouge_scorer.score(reference, generated)
            return {
                'rouge_1_f': scores['rouge1'].fmeasure,
                'rouge_1_p': scores['rouge1'].precision,
                'rouge_1_r': scores['rouge1'].recall,
                'rouge_2_f': scores['rouge2'].fmeasure,
                'rouge_2_p': scores['rouge2'].precision,
                'rouge_2_r': scores['rouge2'].recall,
                'rouge_l_f': scores['rougeL'].fmeasure,
                'rouge_l_p': scores['rougeL'].precision,
                'rouge_l_r': scores['rougeL'].recall,
            }
        except Exception as e:
            print(f"[WARNING] Error calculating ROUGE scores: {e}")
            return {}
    
    def calculate_bleu_score(
        self,
        reference: str,
        generated: str
    ) -> Optional[float]:
        """
        Calculate BLEU score.
        
        Returns:
            BLEU score (0-1), or None if NLTK unavailable
        """
        if not NLTK_AVAILABLE or (CONFIG_AVAILABLE and not is_metric_enabled('bleu')):
            return None
        
        try:
            # Tokenize
            reference_tokens = word_tokenize(reference.lower())
            generated_tokens = word_tokenize(generated.lower())
            
            # Calculate BLEU with smoothing
            smoothing_method = BLEU_SMOOTHING_METHOD if CONFIG_AVAILABLE else 1
            if smoothing_method:
                smoothing = getattr(SmoothingFunction(), f'method{smoothing_method}')
            else:
                smoothing = None
            
            score = sentence_bleu(
                [reference_tokens],
                generated_tokens,
                smoothing_function=smoothing
            )
            return float(score)
        except Exception as e:
            print(f"[WARNING] Error calculating BLEU score: {e}")
            return None
    
    def calculate_tfidf_similarity(
        self,
        reference: str,
        generated: str
    ) -> Optional[float]:
        """
        Calculate TF-IDF cosine similarity.
        
        Returns:
            Cosine similarity score (0-1), or None if sklearn unavailable
        """
        if not SKLEARN_AVAILABLE or (CONFIG_AVAILABLE and not is_metric_enabled('tfidf')):
            return None
        
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([reference, generated])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"[WARNING] Error calculating TF-IDF similarity: {e}")
            return None
    
    def evaluate_answer(
        self,
        question_id: int,
        inference_index: int = None
    ) -> Optional[AnswerMetrics]:
        """
        Evaluate a single answer.
        
        Args:
            question_id: Question ID to evaluate
            inference_index: Which inference to use (None uses instance default)
        
        Returns:
            AnswerMetrics object with all calculated metrics
        """
        # Use instance default if not specified
        if inference_index is None:
            inference_index = self.inference_index
        
        # Check if question should be included
        if CONFIG_AVAILABLE and not should_include_question(question_id):
            return None
        
        # Get reference answer
        if question_id not in self.gold_standard:
            if not (CONFIG_AVAILABLE and should_skip_missing_reference()):
                print(f"[WARNING] No reference answer for question {question_id}")
            return None
        
        gold_entry = self.gold_standard[question_id]
        reference_answer = gold_entry.get('reference_answer', '')
        
        if not reference_answer:
            if not (CONFIG_AVAILABLE and should_skip_missing_reference()):
                print(f"[WARNING] Empty reference answer for question {question_id}")
            return None
        
        # Get generated answer
        if question_id not in self.inference_logs:
            if not (CONFIG_AVAILABLE and should_skip_missing_generated()):
                print(f"[WARNING] No inference logs for question {question_id}")
            return None
        
        inference_log = self.inference_logs[question_id][inference_index]
        generated_answer = inference_log.get('answer', '')
        
        if not generated_answer:
            print(f"[WARNING] Empty generated answer for question {question_id}")
            return None
        
        # Calculate all metrics
        metrics = AnswerMetrics(
            question_id=question_id,
            question=gold_entry.get('question', ''),
            reference_answer=reference_answer,
            generated_answer=generated_answer,
            model_name=inference_log.get('model_name', 'unknown'),
            reference_length=len(reference_answer),
            generated_length=len(generated_answer),
            length_ratio=len(generated_answer) / len(reference_answer) if reference_answer else 0,
            response_time_seconds=inference_log.get('response_time_seconds'),
        )
        
        # Semantic similarity
        metrics.semantic_similarity = self.calculate_semantic_similarity(
            reference_answer,
            generated_answer
        )
        
        # ROUGE scores
        rouge_scores = self.calculate_rouge_scores(reference_answer, generated_answer)
        for key, value in rouge_scores.items():
            setattr(metrics, key, value)
        
        # BLEU score
        metrics.bleu_score = self.calculate_bleu_score(reference_answer, generated_answer)
        
        # TF-IDF similarity
        metrics.tfidf_similarity = self.calculate_tfidf_similarity(
            reference_answer,
            generated_answer
        )
        
        return metrics
    
    def evaluate_all(
        self,
        inference_index: int = None
    ) -> Dict[str, Any]:
        """
        Evaluate all questions across one or more inference sessions.
        
        Args:
            inference_index: Which inference to use (None uses instance default)
        
        Returns:
            Dictionary with results and aggregate statistics (or list of session dicts if multiple)
        """
        # If evaluating multiple sessions, delegate to multi-session method
        if self.num_inference_eval > 1:
            return self._evaluate_multiple_sessions()
        
        # Single session evaluation (original behavior)
        results = []
        
        # Find questions with both reference and generated answers
        evaluatable_questions = set(self.gold_standard.keys()) & set(self.inference_logs.keys())
        
        # Apply max_questions limit if configured
        if self.max_questions is not None:
            evaluatable_questions = list(sorted(evaluatable_questions))[:self.max_questions]
        else:
            evaluatable_questions = sorted(evaluatable_questions)
        
        print(f"\nEvaluating {len(evaluatable_questions)} questions...")
        
        for qid in evaluatable_questions:
            metrics = self.evaluate_answer(qid, inference_index)
            if metrics:
                results.append(metrics.to_dict())
        
        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_stats(results)
        
        # Extract session name from inference log using the same inference_index
        # Use instance default if not specified
        actual_inference_index = inference_index if inference_index is not None else self.inference_index
        session_name = "unknown"
        if self.inference_logs:
            # Get first log entry to extract session name using the same inference index
            first_qid = next(iter(self.inference_logs.keys()))
            if self.inference_logs[first_qid]:
                # Use the same inference_index that was used for evaluation
                session_name = self.inference_logs[first_qid][actual_inference_index].get('session_name', 'unknown')
        
        return {
            'evaluation_timestamp': datetime.now().isoformat(),
            'session_name': session_name,
            'gold_standard_path': str(self.gold_standard_path),
            'inference_log_path': str(self.inference_log_path),
            'embedding_model': self.embedding_model_name,
            'inference_index_used': actual_inference_index,
            'max_questions_limit': self.max_questions,
            'total_questions_evaluated': len(results),
            'per_question_results': results,
            'aggregate_stats': aggregate_stats,
        }
    
    def _evaluate_multiple_sessions(self) -> Dict:
        """
        Evaluate multiple inference sessions starting from the last entry backwards.
        
        Returns:
            Dictionary with list of session results
        """
        # Determine how many sessions are available
        min_sessions = float('inf')
        for qid, logs in self.inference_logs.items():
            min_sessions = min(min_sessions, len(logs))
        
        if min_sessions == 0:
            print("[WARNING] No inference logs available")
            return {'sessions': []}
        
        # Determine actual number of sessions to evaluate
        num_to_eval = min(self.num_inference_eval, int(min_sessions))
        
        print(f"\nEvaluating {num_to_eval} inference sessions (from last backwards)...")
        print(f"Available sessions per question: {int(min_sessions)}")
        
        all_session_results = []
        
        # Evaluate each session starting from -1 (last) backwards: -1, -2, -3, ...
        for i in range(num_to_eval):
            inference_idx = -(i + 1)  # -1, -2, -3, ...
            
            print(f"\n{'='*80}")
            print(f"Evaluating session {i+1}/{num_to_eval} (inference index: {inference_idx})")
            print(f"{'='*80}")
            
            results = []
            
            # Find questions with both reference and generated answers
            evaluatable_questions = set(self.gold_standard.keys()) & set(self.inference_logs.keys())
            
            # Apply max_questions limit if configured
            if self.max_questions is not None:
                evaluatable_questions = list(sorted(evaluatable_questions))[:self.max_questions]
            else:
                evaluatable_questions = sorted(evaluatable_questions)
            
            print(f"Evaluating {len(evaluatable_questions)} questions...")
            
            for qid in evaluatable_questions:
                metrics = self.evaluate_answer(qid, inference_idx)
                if metrics:
                    results.append(metrics.to_dict())
            
            # Calculate aggregate statistics
            aggregate_stats = self._calculate_aggregate_stats(results)
            
            # Extract session name
            session_name = "unknown"
            if self.inference_logs:
                first_qid = next(iter(self.inference_logs.keys()))
                if self.inference_logs[first_qid]:
                    session_name = self.inference_logs[first_qid][inference_idx].get('session_name', 'unknown')
            
            session_result = {
                'session_number': i + 1,
                'inference_index': inference_idx,
                'session_name': session_name,
                'total_questions_evaluated': len(results),
                'per_question_results': results,
                'aggregate_stats': aggregate_stats,
            }
            
            all_session_results.append(session_result)
            
            # Print summary for this session
            self._print_session_summary(session_result, i + 1, num_to_eval)
        
        return {
            'evaluation_timestamp': datetime.now().isoformat(),
            'gold_standard_path': str(self.gold_standard_path),
            'inference_log_path': str(self.inference_log_path),
            'embedding_model': self.embedding_model_name,
            'num_sessions_evaluated': num_to_eval,
            'max_questions_limit': self.max_questions,
            'sessions': all_session_results,
        }
    
    def _calculate_aggregate_stats(self, results: List[Dict]) -> Dict:
        """Calculate aggregate statistics."""
        if not results:
            return {}
        
        stats = {}
        
        # List of metric keys to aggregate
        metric_keys = [
            'semantic_similarity',
            'rouge_1_f', 'rouge_1_p', 'rouge_1_r',
            'rouge_2_f', 'rouge_2_p', 'rouge_2_r',
            'rouge_l_f', 'rouge_l_p', 'rouge_l_r',
            'bleu_score',
            'tfidf_similarity',
            'length_ratio',
            'response_time_seconds',
        ]
        
        for metric_key in metric_keys:
            values = [r[metric_key] for r in results if r.get(metric_key) is not None]
            
            if values:
                stats[metric_key] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values),
                }
        
        return stats
    
    def _print_session_summary(self, session_result: Dict, session_num: int, total_sessions: int):
        """Print summary for a single session."""
        print(f"\n[OK] Session {session_num}/{total_sessions} completed")
        print(f"Session Name: {session_result['session_name']}")
        print(f"Questions Evaluated: {session_result['total_questions_evaluated']}")
        
        stats = session_result.get('aggregate_stats', {})
        if stats:
            if 'semantic_similarity' in stats:
                sem = stats['semantic_similarity']
                print(f"Semantic Similarity: {sem['mean']:.4f} ± {sem['std']:.4f}")
            if 'rouge_1_f' in stats:
                print(f"ROUGE-1 F1: {stats['rouge_1_f']['mean']:.4f} ± {stats['rouge_1_f']['std']:.4f}")
            if 'response_time_seconds' in stats:
                rt = stats['response_time_seconds']
                print(f"Response Time: {rt['mean']:.2f}s ± {rt['std']:.2f}s")
    
    def print_summary(self, results: Dict):
        """Print a human-readable summary."""
        # Check if this is a multi-session result
        if 'sessions' in results:
            print("\n" + "="*80)
            print("MULTI-SESSION ANSWER QUALITY EVALUATION SUMMARY")
            print("="*80)
            print(f"\nTotal Sessions Evaluated: {results['num_sessions_evaluated']}")
            print(f"Embedding Model: {results['embedding_model']}")
            
            for session in results['sessions']:
                print("\n" + "-"*80)
                print(f"SESSION {session['session_number']}: {session['session_name']}")
                print("-"*80)
                print(f"Questions Evaluated: {session['total_questions_evaluated']}")
                self._print_aggregate_metrics(session.get('aggregate_stats', {}))
            
            print("\n" + "="*80)
            return
        
        # Single session result (original behavior)
        print("\n" + "="*80)
        print("ANSWER QUALITY EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\nSession Name: {results.get('session_name', 'unknown')}")
        print(f"Total Questions Evaluated: {results['total_questions_evaluated']}")
        print(f"Embedding Model: {results['embedding_model']}")
        
        self._print_aggregate_metrics(results.get('aggregate_stats', {}))
        print("\n" + "="*80)
    
    def _print_aggregate_metrics(self, stats: Dict):
        """Print aggregate metrics for a session."""
        if not stats:
            print("\n[WARNING] No metrics calculated (missing dependencies?)")
            return
        
        print("\n" + "-"*80)
        print("AGGREGATE METRICS (Mean ± Std)")
        print("-"*80)
        
        # Semantic Similarity
        if 'semantic_similarity' in stats:
            sem_stats = stats['semantic_similarity']
            print(f"\n[METRIC] Semantic Similarity (Embedding-based):")
            print(f"   Mean: {sem_stats['mean']:.4f} ± {sem_stats['std']:.4f}")
            print(f"   Range: [{sem_stats['min']:.4f}, {sem_stats['max']:.4f}]")
        
        # ROUGE Scores
        if 'rouge_1_f' in stats:
            print(f"\n[METRIC] ROUGE Scores:")
            print(f"   ROUGE-1 F1: {stats['rouge_1_f']['mean']:.4f} ± {stats['rouge_1_f']['std']:.4f}")
            print(f"   ROUGE-2 F1: {stats['rouge_2_f']['mean']:.4f} ± {stats['rouge_2_f']['std']:.4f}")
            print(f"   ROUGE-L F1: {stats['rouge_l_f']['mean']:.4f} ± {stats['rouge_l_f']['std']:.4f}")
        
        # BLEU Score
        if 'bleu_score' in stats:
            bleu_stats = stats['bleu_score']
            print(f"\n[METRIC] BLEU Score:")
            print(f"   Mean: {bleu_stats['mean']:.4f} ± {bleu_stats['std']:.4f}")
        
        # TF-IDF Similarity
        if 'tfidf_similarity' in stats:
            tfidf_stats = stats['tfidf_similarity']
            print(f"\n[METRIC] TF-IDF Cosine Similarity:")
            print(f"   Mean: {tfidf_stats['mean']:.4f} ± {tfidf_stats['std']:.4f}")
        
        # Response Time
        if 'response_time_seconds' in stats:
            time_stats = stats['response_time_seconds']
            print(f"\n[TIME] Response Time:")
            print(f"   Mean: {time_stats['mean']:.2f}s ± {time_stats['std']:.2f}s")
            print(f"   Range: [{time_stats['min']:.2f}s, {time_stats['max']:.2f}s]")
        
        # Answer Length
        if 'length_ratio' in stats:
            len_stats = stats['length_ratio']
            print(f"\n[LENGTH] Answer Length Ratio (Generated/Reference):")
            print(f"   Mean: {len_stats['mean']:.2f} ± {len_stats['std']:.2f}")
    
    def export_to_json(self, results: Dict, output_path: str):
        """Export results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[OK] Results exported to: {output_path}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG answer quality")
    parser.add_argument(
        '--gold-standard',
        required=True,
        help='Path to gold standard JSON file'
    )
    parser.add_argument(
        '--inference-log',
        required=True,
        help='Path to inference log JSONL file'
    )
    parser.add_argument(
        '--output-json',
        help='Path to save JSON results'
    )
    parser.add_argument(
        '--embedding-model',
        default=None,
        help='Sentence transformer model for semantic similarity (default from config or Qwen3-Embedding-0.6B)'
    )
    parser.add_argument(
        '--inference-index',
        type=int,
        default=None,
        help='Which inference log to use: -1 for last (most recent), 0 for first (default from config or -1)'
    )
    parser.add_argument(
        '--max-questions',
        type=int,
        default=None,
        help='Maximum number of questions to evaluate (default from config or all)'
    )
    parser.add_argument(
        '--num-inference-eval',
        type=int,
        default=None,
        help='Number of inference sessions to evaluate starting from last backwards (default from config or 1)'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = AnswerEvaluator(
        gold_standard_path=args.gold_standard,
        inference_log_path=args.inference_log,
        embedding_model=args.embedding_model,
        inference_index=args.inference_index,
        max_questions=args.max_questions,
        num_inference_eval=args.num_inference_eval
    )
    
    # Run evaluation
    results = evaluator.evaluate_all()
    
    # Print summary
    evaluator.print_summary(results)
    
    # Export if requested
    if args.output_json:
        evaluator.export_to_json(results, args.output_json)
