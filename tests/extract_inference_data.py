"""
Inference Data Extractor
========================
Extracts question, answer, and retrieved chunks from inference session logs.

This module provides simple, reusable functions to parse the latest inference log
and extract structured data for analysis, evaluation, or further processing.

Author: Generated for 01_RAG project
Date: November 24, 2025
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class InferenceDataExtractor:
    """Extract structured data from inference session logs."""
    
    def __init__(self, log_file_path: Path):
        """
        Initialize extractor with log file path.
        
        Args:
            log_file_path: Path to the inference session log file
        """
        self.log_file_path = log_file_path
        self.log_content = self._read_log_file()
    
    def _read_log_file(self) -> str:
        """Read the entire log file content."""
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Failed to read log file: {e}")
    
    def extract_all(self) -> List[Dict]:
        """
        Extract all inference data (questions, answers, chunks) from log.
        
        Returns:
            List of dictionaries, each containing:
                - question_id: int
                - question: str
                - answer: str
                - thinking: str
                - chunks: List[Dict] with source, index, similarity, content
                - category: str
                - response_time: float
                - success: bool
        """
        results = []
        
        # Split log into question sections
        question_sections = self._split_into_questions()
        
        for section in question_sections:
            extracted_data = self._extract_from_section(section)
            if extracted_data:
                results.append(extracted_data)
        
        return results
    
    def _split_into_questions(self) -> List[Tuple[int, str]]:
        """
        Split log content into individual question sections.
        
        Returns:
            List of tuples (question_id, section_content)
        """
        # Pattern to match question headers with flexible separator lengths
        pattern = r'={70,}\s*\nQUESTION (\d+)\s*\n={70,}\s*\n(.*?)(?=\n={70,}\s*\nQUESTION \d+|$)'
        matches = re.findall(pattern, self.log_content, re.DOTALL)
        
        return [(int(qid), content) for qid, content in matches]
    
    def _extract_from_section(self, section_data: Tuple[int, str]) -> Optional[Dict]:
        """
        Extract data from a single question section.
        
        Args:
            section_data: Tuple of (question_id, section_content)
        
        Returns:
            Dictionary with extracted data or None if parsing fails
        """
        question_id, content = section_data
        
        try:
            # Extract question
            question = self._extract_question(content)
            if not question:
                return None
            
            # Extract answer from PARSED COMPONENTS section (complete answer)
            answer = self._extract_answer(content)
            
            # Extract thinking process
            thinking = self._extract_thinking(content)
            
            # Extract retrieved chunks
            chunks = self._extract_chunks(content)
            
            # Extract metadata
            category = self._extract_category(content)
            response_time = self._extract_response_time(content)
            success = self._extract_success_status(content)
            
            return {
                'question_id': question_id,
                'question': question.strip(),
                'answer': answer.strip() if answer else "",
                'thinking': thinking.strip() if thinking else "",
                'chunks': chunks,
                'category': category,
                'response_time': response_time,
                'success': success,
                'num_chunks': len(chunks)
            }
        
        except Exception as e:
            print(f"[!] Warning: Failed to parse question {question_id}: {e}")
            return None
    
    def _extract_question(self, content: str) -> Optional[str]:
        """Extract the question text."""
        match = re.search(r'Question:\s*\n(.+?)(?=\n\n|Category:)', content, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_answer(self, content: str) -> Optional[str]:
        """Extract the final answer from PARSED COMPONENTS section."""
        # Look for "Final Answer:" in PARSED COMPONENTS (complete answer)
        match = re.search(
            r'Final Answer:\s*\n(.+?)(?:\n\n-{70,}|PERFORMANCE METRICS)',
            content,
            re.DOTALL
        )
        return match.group(1).strip() if match else None
    
    def _extract_thinking(self, content: str) -> Optional[str]:
        """Extract the thinking/reasoning process."""
        match = re.search(
            r'PARSED COMPONENTS.*?Thinking Process:\s*\n(.+?)(?:\nFinal Answer:|\n\n)',
            content,
            re.DOTALL
        )
        return match.group(1).strip() if match else None
    
    def _extract_chunks(self, content: str) -> List[Dict]:
        """
        Extract all retrieved chunks with metadata.
        
        Returns:
            List of dictionaries with chunk information
        """
        chunks = []
        
        # Pattern to match chunk information
        # Captures everything until the next "Chunk X:" or end of section
        pattern = r'Chunk (\d+):\n\s+Source:\s*(.+?)\n\s+Chunk Index:\s*(.+?)\n\s+Similarity Score:\s*([\d.]+)%\n\s+Content Preview:\s*(.+?)(?=\n\nChunk \d+:|\n\n={70,}|\n\n-{70,}|\Z)'
        
        matches = re.findall(pattern, content, re.DOTALL)
        
        for chunk_num, source, index, similarity, chunk_content in matches:
            chunks.append({
                'chunk_number': int(chunk_num),
                'source': source.strip(),
                'chunk_index': index.strip(),
                'similarity_score': float(similarity),
                'content': chunk_content.strip()
            })
        
        return chunks
    
    def _extract_category(self, content: str) -> str:
        """Extract question category."""
        match = re.search(r'Category:\s*(.+)', content)
        return match.group(1).strip() if match else "unknown"
    
    def _extract_response_time(self, content: str) -> float:
        """Extract response time in seconds."""
        match = re.search(r'Response Time:\s*([\d.]+)\s*seconds', content)
        return float(match.group(1)) if match else 0.0
    
    def _extract_success_status(self, content: str) -> bool:
        """Extract success status."""
        # Check for explicit status field
        match = re.search(r'Status:\s*(\w+)', content)
        if match:
            return match.group(1).strip().upper() == 'SUCCESS'
        # If no explicit status and we have an answer, consider it successful
        return bool(self._extract_answer(content))


def find_latest_log(sessions_dir: Path) -> Optional[Path]:
    """
    Find the most recent log file in the sessions directory.
    
    Args:
        sessions_dir: Path to the sessions directory
    
    Returns:
        Path to the latest log file or None if no logs found
    """
    if not sessions_dir.exists():
        return None
    
    log_files = list(sessions_dir.glob("test_session_*.log"))
    
    if not log_files:
        return None
    
    # Sort by modification time and return most recent
    return max(log_files, key=lambda f: f.stat().st_mtime)


def extract_from_latest_log(sessions_dir: Optional[Path] = None) -> List[Dict]:
    """
    Extract inference data from the latest log file.
    
    Args:
        sessions_dir: Path to sessions directory (default: tests/logs/sessions)
    
    Returns:
        List of extracted inference data dictionaries
    """
    if sessions_dir is None:
        sessions_dir = project_root / "tests" / "logs" / "sessions"
    
    log_file = find_latest_log(sessions_dir)
    
    if not log_file:
        raise FileNotFoundError(f"No log files found in {sessions_dir}")
    
    print(f"[*] Extracting from: {log_file.name}")
    
    extractor = InferenceDataExtractor(log_file)
    data = extractor.extract_all()
    
    print(f"[OK] Extracted {len(data)} questions")  
    
    return data


def print_extraction_summary(data: List[Dict]):
    """Print a summary of extracted data."""
    
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80 + "\n")
    
    print(f"Total Questions: {len(data)}")
    print(f"Successful: {sum(1 for d in data if d['success'])}")
    print()
    
    # Answer statistics
    answer_lengths = [len(d['answer']) for d in data]
    print("Answer Statistics:")
    print(f"  Average length: {sum(answer_lengths) / len(answer_lengths):.0f} chars")
    print(f"  Min length: {min(answer_lengths)} chars")
    print(f"  Max length: {max(answer_lengths)} chars")
    print()
    
    # Chunk statistics
    chunk_counts = [d['num_chunks'] for d in data]
    print("Chunk Statistics:")
    print(f"  Average per question: {sum(chunk_counts) / len(chunk_counts):.1f}")
    print(f"  Min: {min(chunk_counts)}")
    print(f"  Max: {max(chunk_counts)}")
    print()
    
    # Performance statistics
    response_times = [d['response_time'] for d in data]
    print("Performance Statistics:")
    print(f"  Average response time: {sum(response_times) / len(response_times):.2f}s")
    print(f"  Min: {min(response_times):.2f}s")
    print(f"  Max: {max(response_times):.2f}s")
    print()
    
    # Category breakdown
    categories = {}
    for d in data:
        cat = d['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    print()
    
    # Sample questions
    print("="*80)
    print("SAMPLE EXTRACTIONS")
    print("="*80 + "\n")
    
    for i, d in enumerate(data[:3], 1):
        print(f"Question {d['question_id']} ({d['category']}):")
        print(f"  Q: {d['question'][:70]}...")
        print(f"  A: {d['answer'][:100]}...")
        print(f"  Thinking: {d['thinking'][:80]}..." if d['thinking'] else "  Thinking: (none)")
        print(f"  Chunks: {d['num_chunks']}")
        print(f"  Top chunk similarity: {d['chunks'][0]['similarity_score']:.1f}%" if d['chunks'] else "  (no chunks)")
        print()


def export_to_json(data: List[Dict], output_file: Optional[Path] = None) -> Path:
    """
    Export extracted data to JSON file.
    
    Args:
        data: Extracted inference data
        output_file: Output file path (default: tests/logs/extracted_inference_data.json)
    
    Returns:
        Path to saved JSON file
    """
    import json
    
    if output_file is None:
        output_file = project_root / "tests" / "logs" / "extracted_inference_data.json"
    
    output_data = {
        'extraction_timestamp': datetime.now().isoformat(),
        'total_questions': len(data),
        'data': data
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"[SAVE] Exported to: {output_file.relative_to(project_root)}")
    
    return output_file


def main():
    """Main entry point for data extraction."""
    
    print("="*80)
    print("INFERENCE DATA EXTRACTOR")
    print("="*80)
    print()
    
    try:
        # Extract data from latest log
        sessions_dir = project_root / "tests" / "logs" / "sessions"
        data = extract_from_latest_log(sessions_dir)
        
        # Print summary
        print_extraction_summary(data)
        
        # Export to JSON
        export_to_json(data)
        
        print("="*80)
        print("[OK] Extraction complete!")
        print("="*80)
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
