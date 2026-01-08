"""
Pytest for verifying random chunk extraction completeness.

Tests random sample of chunks to ensure consistent extraction quality.
"""
import sys
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

# Import with try/except for better error handling
try:
    from tests.extract_inference_data import extract_from_latest_log, find_latest_log
except ImportError:
    import extract_inference_data
    extract_from_latest_log = extract_inference_data.extract_from_latest_log
    find_latest_log = extract_inference_data.find_latest_log


@pytest.fixture(scope="module")
def extracted_data():
    """Fixture to extract data once for all tests."""
    sessions_dir = Path(__file__).parent / "logs" / "sessions"
    
    # Skip if sessions directory doesn't exist (CI/CD environment)
    if not sessions_dir.exists():
        pytest.skip("Session logs directory not available (CI/CD environment)", allow_module_level=True)
    
    log_file = find_latest_log(sessions_dir)
    
    if not log_file:
        pytest.skip("No session log files found", allow_module_level=True)
    
    data = extract_from_latest_log(sessions_dir)
    return data


@pytest.fixture(scope="module")
def all_chunks(extracted_data):
    """Fixture to collect all chunks from all questions."""
    chunks = []
    for q_idx, question in enumerate(extracted_data):
        for c_idx, chunk in enumerate(question['chunks']):
            chunks.append({
                'question_id': question['question_id'],
                'chunk_num': chunk['chunk_number'],
                'chunk': chunk,
                'q_idx': q_idx,
                'c_idx': c_idx
            })
    return chunks


@pytest.fixture(scope="module")
def sample_chunks(all_chunks):
    """Fixture to get 10 random chunks for testing."""
    random.seed(42)  # For reproducibility
    return random.sample(all_chunks, min(10, len(all_chunks)))


def test_random_chunks_available(all_chunks):
    """Test that chunks are available for random sampling."""
    assert len(all_chunks) > 0, "No chunks available for testing"
    assert len(all_chunks) >= 10, f"Expected at least 10 chunks, got {len(all_chunks)}"


def test_random_chunks_substantial_length(sample_chunks):
    """Test that random chunks have substantial content length."""
    for item in sample_chunks:
        chunk = item['chunk']
        content_length = len(chunk['content'])
        assert content_length > 500, \
            f"Chunk Q{item['question_id']}-{item['chunk_num']} too short: {content_length} chars"


def test_random_chunks_have_newlines(sample_chunks):
    """Test that random chunks contain newlines (paragraph breaks)."""
    for item in sample_chunks:
        chunk = item['chunk']
        content = chunk['content']
        assert '\n' in content, \
            f"Chunk Q{item['question_id']}-{item['chunk_num']} missing newlines"


def test_random_chunks_end_correctly(sample_chunks):
    """Test that random chunks end with ellipsis as expected."""
    for item in sample_chunks:
        chunk = item['chunk']
        content = chunk['content'].strip()
        if len(content) > 100:  # Only check longer chunks
            assert content.endswith('...'), \
                f"Chunk Q{item['question_id']}-{item['chunk_num']} doesn't end with '...'"


def test_random_chunks_word_count(sample_chunks):
    """Test that random chunks have reasonable word count."""
    for item in sample_chunks:
        chunk = item['chunk']
        word_count = len(chunk['content'].split())
        assert word_count > 50, \
            f"Chunk Q{item['question_id']}-{item['chunk_num']} has too few words: {word_count}"


def test_random_chunks_not_empty(sample_chunks):
    """Test that no random chunks are empty or suspiciously short."""
    for item in sample_chunks:
        chunk = item['chunk']
        content = chunk['content']
        
        assert content, f"Chunk Q{item['question_id']}-{item['chunk_num']} is empty"
        assert len(content) >= 50, \
            f"Chunk Q{item['question_id']}-{item['chunk_num']} suspiciously short: {len(content)} chars"


def test_random_chunks_metadata_valid(sample_chunks):
    """Test that random chunk metadata is valid."""
    for item in sample_chunks:
        chunk = item['chunk']
        
        # Check source
        assert chunk['source'], f"Chunk Q{item['question_id']}-{item['chunk_num']} missing source"
        
        # Check similarity score
        score = chunk['similarity_score']
        assert 0 <= score <= 100, \
            f"Chunk Q{item['question_id']}-{item['chunk_num']} invalid similarity: {score}"
        
        # Check chunk number
        assert chunk['chunk_number'] > 0, \
            f"Chunk Q{item['question_id']}-{item['chunk_num']} invalid chunk_number"


def test_random_chunks_formatting_consistent(sample_chunks):
    """Test that random chunks have consistent formatting."""
    long_chunks = [item for item in sample_chunks if len(item['chunk']['content']) > 100]
    
    for item in long_chunks:
        chunk = item['chunk']
        content = chunk['content']
        
        # Should have newlines for long content
        assert '\n' in content, \
            f"Long chunk Q{item['question_id']}-{item['chunk_num']} missing paragraph breaks"


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v"])
