"""
Pytest for verifying chunk extraction completeness.

Tests that chunks are extracted completely without truncation,
including embedded paragraphs and blank lines.
"""
import sys
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
    # Try direct import if tests module is not available
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


def test_extraction_successful(extracted_data):
    """Test that data extraction completed successfully."""
    assert extracted_data is not None
    assert len(extracted_data) > 0, "No questions extracted"
    assert all('chunks' in q for q in extracted_data), "Missing chunks field"


def test_chunk_completeness(extracted_data):
    """Test that chunks are extracted completely without truncation."""
    # Get chunks from first question
    first_question = extracted_data[0]
    assert len(first_question['chunks']) > 0, "No chunks extracted for first question"
    
    chunk1 = first_question['chunks'][0]
    
    # Verify chunk structure
    assert 'content' in chunk1, "Missing content field"
    assert 'source' in chunk1, "Missing source field"
    assert 'chunk_index' in chunk1, "Missing chunk_index field"
    assert 'similarity_score' in chunk1, "Missing similarity_score field"
    
    # Check chunk content is substantial (not truncated)
    assert len(chunk1['content']) > 500, f"Chunk 1 seems truncated: only {len(chunk1['content'])} chars"
    
    # Check for embedded newlines (paragraphs preserved)
    assert '\n' in chunk1['content'], "Chunk missing newlines - paragraphs may be lost"
    
    # Check multiple paragraphs exist (blank lines preserved)
    paragraph_count = chunk1['content'].count('\n\n')
    assert paragraph_count > 0, "Chunk missing paragraph breaks"


def test_multiple_chunks_complete(extracted_data):
    """Test that multiple chunks are all extracted completely."""
    first_question = extracted_data[0]
    chunks = first_question['chunks']
    
    # Test at least 2 chunks if available
    if len(chunks) >= 2:
        chunk1 = chunks[0]
        chunk2 = chunks[1]
        
        # Both should be substantial
        assert len(chunk1['content']) > 500, f"Chunk 1 truncated: {len(chunk1['content'])} chars"
        assert len(chunk2['content']) > 500, f"Chunk 2 truncated: {len(chunk2['content'])} chars"
        
        # Both should have newlines
        assert '\n' in chunk1['content'], "Chunk 1 missing newlines"
        assert '\n' in chunk2['content'], "Chunk 2 missing newlines"


def test_all_chunks_have_content(extracted_data):
    """Test that all chunks across all questions have content."""
    total_chunks = 0
    empty_chunks = 0
    truncated_chunks = 0
    
    for question in extracted_data:
        for chunk in question['chunks']:
            total_chunks += 1
            
            if not chunk['content'] or len(chunk['content']) == 0:
                empty_chunks += 1
            elif len(chunk['content']) < 100:  # Suspiciously short
                truncated_chunks += 1
    
    assert total_chunks > 0, "No chunks found across all questions"
    assert empty_chunks == 0, f"Found {empty_chunks} empty chunks out of {total_chunks}"
    assert truncated_chunks == 0, f"Found {truncated_chunks} suspiciously short chunks out of {total_chunks}"


def test_chunk_similarity_scores(extracted_data):
    """Test that similarity scores are valid."""
    for question in extracted_data:
        for chunk in question['chunks']:
            score = chunk['similarity_score']
            assert isinstance(score, (int, float)), f"Invalid similarity score type: {type(score)}"
            assert 0 <= score <= 100, f"Similarity score out of range: {score}"


def test_chunk_metadata_complete(extracted_data):
    """Test that all chunk metadata fields are present and valid."""
    first_question = extracted_data[0]
    
    for chunk in first_question['chunks']:
        # Required fields
        assert chunk['source'], "Missing or empty source"
        assert chunk['chunk_index'], "Missing or empty chunk_index"
        assert isinstance(chunk['chunk_number'], int), "chunk_number must be integer"
        assert chunk['chunk_number'] > 0, "chunk_number must be positive"


def test_chunk_extraction_consistency(extracted_data):
    """Test that chunk extraction is consistent across questions."""
    questions_with_chunks = [q for q in extracted_data if len(q['chunks']) > 0]
    
    if questions_with_chunks:
        # All chunks should end with ... (as per log format)
        for question in questions_with_chunks[:3]:  # Check first 3
            for chunk in question['chunks']:
                if len(chunk['content']) > 100:  # Skip very short chunks
                    assert chunk['content'].strip().endswith('...'), \
                        f"Chunk doesn't end with '...' as expected from log format"


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v"])
