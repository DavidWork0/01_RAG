"""
Inference Performance Logger
=============================
Module for logging and tracking RAG system inference performance over time.

Author: Generated for 01_RAG project
Date: November 6, 2025
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


class InferenceLogger:
    """Logger for tracking RAG system inference performance."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize the inference logger.
        
        Args:
            log_dir: Directory to store log files (if None, uses project_root/tests/logs)
        """
        if log_dir is None:
            # Get project root (parent of src directory)
            project_root = Path(__file__).parent.parent
            self.log_dir = project_root / "tests" / "logs"
        else:
            self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON Lines log file for detailed records
        self.jsonl_log = self.log_dir / "inference_log.jsonl"
        
        # CSV summary log for quick analysis
        self.csv_log = self.log_dir / "inference_summary.csv"
        
        # Initialize CSV if it doesn't exist
        if not self.csv_log.exists():
            self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize the CSV log file with headers."""
        headers = [
            "timestamp",
            "session_name",
            "question_id",
            "question",
            "model_name",
            "response_time_seconds",
            "num_chunks_retrieved",
            "num_chunks_used",
            "answer_length",
            "has_thinking",
            "thinking_length",
            "success",
            "error_message"
        ]
        
        with open(self.csv_log, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_inference(
        self,
        question_id: Optional[int],
        question: str,
        answer: str,
        model_name: str,
        response_time: float,
        num_chunks_retrieved: int,
        num_chunks_used: Optional[int] = None,
        thinking: Optional[str] = None,
        sources: Optional[List[Dict]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None,
        session_name: Optional[str] = None
    ) -> Dict:
        """
        Log an inference test result.
        
        Args:
            question_id: ID of the test question (if from test set)
            question: The question asked
            answer: The generated answer
            model_name: Name of the LLM model used
            response_time: Time taken to generate response (seconds)
            num_chunks_retrieved: Number of chunks retrieved from RAG
            num_chunks_used: Number of chunks actually used in context
            thinking: Thinking process (if available)
            sources: List of source chunks with metadata
            error: Error message if inference failed
            metadata: Additional metadata to log
            session_name: Name of the test session for grouping
        
        Returns:
            Dictionary containing the logged entry
        """
        timestamp = datetime.now().isoformat()
        
        # Create detailed log entry
        log_entry = {
            "timestamp": timestamp,
            "session_name": session_name,
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "model_name": model_name,
            "response_time_seconds": response_time,
            "num_chunks_retrieved": num_chunks_retrieved,
            "num_chunks_used": num_chunks_used or num_chunks_retrieved,
            "answer_length": len(answer) if answer else 0,
            "has_thinking": thinking is not None,
            "thinking_length": len(thinking) if thinking else 0,
            "thinking": thinking,
            "success": error is None,
            "error_message": error,
            "sources": sources,
            "metadata": metadata or {}
        }
        
        # Write to JSON Lines log (detailed)
        with open(self.jsonl_log, 'a', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write('\n')
        
        # Write to CSV log (summary)
        with open(self.csv_log, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                session_name or "",
                question_id,
                question,
                model_name,
                response_time,
                num_chunks_retrieved,
                num_chunks_used or num_chunks_retrieved,
                len(answer) if answer else 0,
                thinking is not None,
                len(thinking) if thinking else 0,
                error is None,
                error or ""
            ])
        
        return log_entry
    
    def get_logs(
        self,
        limit: Optional[int] = None,
        model_name: Optional[str] = None,
        question_id: Optional[int] = None,
        success_only: bool = False
    ) -> List[Dict]:
        """
        Retrieve logged inference records.
        
        Args:
            limit: Maximum number of records to return (most recent first)
            model_name: Filter by model name
            question_id: Filter by question ID
            success_only: Only return successful inferences
        
        Returns:
            List of log entries
        """
        if not self.jsonl_log.exists():
            return []
        
        logs = []
        with open(self.jsonl_log, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        
        # Apply filters
        if model_name:
            logs = [log for log in logs if log.get('model_name') == model_name]
        
        if question_id is not None:
            logs = [log for log in logs if log.get('question_id') == question_id]
        
        if success_only:
            logs = [log for log in logs if log.get('success', False)]
        
        # Sort by timestamp (most recent first)
        logs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Apply limit
        if limit:
            logs = logs[:limit]
        
        return logs
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Get summary statistics as a pandas DataFrame.
        
        Returns:
            DataFrame with inference log summary
        """
        if not self.csv_log.exists():
            return pd.DataFrame()
        
        return pd.read_csv(self.csv_log)
    
    def get_statistics(self, model_name: Optional[str] = None) -> Dict:
        """
        Calculate statistics from logged inferences.
        
        Args:
            model_name: Filter statistics by model name
        
        Returns:
            Dictionary with statistical summary
        """
        logs = self.get_logs(model_name=model_name)
        
        if not logs:
            return {
                "total_inferences": 0,
                "successful_inferences": 0,
                "failed_inferences": 0,
                "success_rate": 0.0,
                "avg_response_time": 0.0,
                "avg_chunks_retrieved": 0.0,
                "avg_answer_length": 0.0
            }
        
        successful = [log for log in logs if log.get('success', False)]
        
        stats = {
            "total_inferences": len(logs),
            "successful_inferences": len(successful),
            "failed_inferences": len(logs) - len(successful),
            "success_rate": len(successful) / len(logs) * 100,
            "avg_response_time": sum(log.get('response_time_seconds', 0) for log in successful) / len(successful) if successful else 0.0,
            "avg_chunks_retrieved": sum(log.get('num_chunks_retrieved', 0) for log in successful) / len(successful) if successful else 0.0,
            "avg_answer_length": sum(log.get('answer_length', 0) for log in successful) / len(successful) if successful else 0.0,
            "min_response_time": min(log.get('response_time_seconds', 0) for log in successful) if successful else 0.0,
            "max_response_time": max(log.get('response_time_seconds', 0) for log in successful) if successful else 0.0,
        }
        
        return stats
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Compare performance across different models.
        
        Returns:
            DataFrame with model comparison statistics
        """
        logs = self.get_logs()
        
        if not logs:
            return pd.DataFrame()
        
        models = set(log.get('model_name') for log in logs)
        comparison = []
        
        for model in models:
            stats = self.get_statistics(model_name=model)
            stats['model_name'] = model
            comparison.append(stats)
        
        return pd.DataFrame(comparison)
    
    def get_question_performance(self, question_id: int) -> List[Dict]:
        """
        Get all inference attempts for a specific question.
        
        Args:
            question_id: The question ID to analyze
        
        Returns:
            List of log entries for the question
        """
        return self.get_logs(question_id=question_id)
    
    def export_to_excel(self, output_path: str):
        """
        Export logs to an Excel file with multiple sheets.
        
        Args:
            output_path: Path to save the Excel file
        """
        df_summary = self.get_summary_dataframe()
        df_comparison = self.get_model_comparison()
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            if not df_summary.empty:
                df_summary.to_excel(writer, sheet_name='Inference Log', index=False)
            
            if not df_comparison.empty:
                df_comparison.to_excel(writer, sheet_name='Model Comparison', index=False)
            
            # Add statistics sheet
            all_stats = self.get_statistics()
            df_stats = pd.DataFrame([all_stats])
            df_stats.to_excel(writer, sheet_name='Overall Statistics', index=False)
