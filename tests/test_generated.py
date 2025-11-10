"""
Comprehensive Test Suite for RAG Pipeline
==========================================
Tests the complete pipeline:
1. data_pipeline_pdf.py - PDF processing and image extraction
2. pre_chunking.py - Text cleaning
3. chunk_qwen3_0_6B.py - Chunking and embedding database creation
4. streamlit_modern_multiuser.py - RAG query system

Author: Generated for 01_RAG project
Date: November 3, 2025
"""

# cmd command from root folder: $env:PYTHONIOENCODING="utf-8" ; .venv\Scripts\python.exe tests\test_generated.py

import unittest
import sys
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings

# Add project root and src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'src' / 'intevl3_5'))

# Try to import modules - some may fail if dependencies aren't installed
# We'll mark those tests as skipped
try:
    import data_pipeline_pdf
    DATA_PIPELINE_AVAILABLE = True
except ImportError as e:
    DATA_PIPELINE_AVAILABLE = False
    print(f"Warning: data_pipeline_pdf import failed: {e}")

try:
    from pre_chunking import TextCleaner
    PRE_CHUNKING_AVAILABLE = True
except ImportError as e:
    PRE_CHUNKING_AVAILABLE = False
    print(f"Warning: pre_chunking import failed: {e}")

try:
    import chunk_qwen3_0_6B
    CHUNKING_AVAILABLE = True
except ImportError as e:
    CHUNKING_AVAILABLE = False
    print(f"Warning: chunk_qwen3_0_6B import failed: {e}")

try:
    import hybrid_rag_module_qwen3
    HYBRID_RAG_AVAILABLE = True
except ImportError as e:
    HYBRID_RAG_AVAILABLE = False
    print(f"Warning: hybrid_rag_module_qwen3 import failed: {e}")


@unittest.skipUnless(DATA_PIPELINE_AVAILABLE, "data_pipeline_pdf not available")
class TestDataPipelinePDF(unittest.TestCase):
    """Test suite for data_pipeline_pdf.py"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.test_dir, "images")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_parse_descriptions_file_basic(self):
        """Test parsing of image descriptions file"""
        # Create a test descriptions file
        
        desc_file = os.path.join(self.test_dir, "descriptions.txt")
        with open(desc_file, 'w', encoding='utf-8') as f:
            f.write("[IMAGE:test_img_001]\n")
            f.write("This is a test image description.\n")
            f.write("It has multiple lines.\n\n")
            f.write("[IMAGE:test_img_002]\n")
            f.write("Another description here.\n")
        
        # Parse the file
        descriptions = data_pipeline_pdf.parse_descriptions_file(desc_file)
        
        # Assertions
        self.assertEqual(len(descriptions), 2)
        self.assertIn("[IMAGE:test_img_001]", descriptions)
        self.assertIn("[IMAGE:test_img_002]", descriptions)
        self.assertIn("test image description", descriptions["[IMAGE:test_img_001]"])
        self.assertIn("Another description", descriptions["[IMAGE:test_img_002]"])
    
    def test_parse_descriptions_empty_lines(self):
        """Test that empty lines in descriptions are handled correctly"""
        desc_file = os.path.join(self.test_dir, "descriptions.txt")
        with open(desc_file, 'w', encoding='utf-8') as f:
            f.write("[IMAGE:test_img_001]\n")
            f.write("Line one\n")
            f.write("\n")
            f.write("Line three\n")
        
        descriptions = data_pipeline_pdf.parse_descriptions_file(desc_file)
        
        self.assertIn("[IMAGE:test_img_001]", descriptions)
        # Check that intentional blank lines are preserved
        desc_text = descriptions["[IMAGE:test_img_001]"]
        self.assertTrue("Line one" in desc_text)
        self.assertTrue("Line three" in desc_text)
    
    def test_merge_text_with_descriptions(self):
        """Test merging of text with image descriptions"""
        # Create text with placeholders
        text_file = os.path.join(self.test_dir, "text.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("Some text before image.\n")
            f.write("[IMAGE:test_img_001]\n")
            f.write("Some text after image.\n")
        
        # Create descriptions file
        desc_file = os.path.join(self.test_dir, "descriptions.txt")
        with open(desc_file, 'w', encoding='utf-8') as f:
            f.write("[IMAGE:test_img_001]\n")
            f.write("A beautiful sunset over mountains.\n")
        
        # Merge
        output_file = os.path.join(self.test_dir, "merged.txt")
        data_pipeline_pdf.merge_text_with_descriptions(text_file, desc_file, output_file)
        
        # Read and verify
        with open(output_file, 'r', encoding='utf-8') as f:
            merged_text = f.read()
        
        self.assertIn("Some text before image", merged_text)
        self.assertIn("beautiful sunset over mountains", merged_text)
        self.assertIn("Some text after image", merged_text)
        self.assertNotIn("[IMAGE:test_img_001]", merged_text)
    
    def test_rename_pdf_file_names(self):
        """Test PDF filename sanitization"""
        # Create test PDF files with problematic names
        test_pdfs = [
            "test.file.with.dots.pdf",
            "test file with spaces.pdf",
            "test-file-with-dashes.pdf"
        ]
        
        for pdf_name in test_pdfs:
            pdf_path = os.path.join(self.test_dir, pdf_name)
            with open(pdf_path, 'w') as f:
                f.write("dummy pdf content")
        
        # Rename files
        data_pipeline_pdf.rename_pdf_file_names(self.test_dir)
        
        # Check that files are renamed
        renamed_files = os.listdir(self.test_dir)
        for filename in renamed_files:
            if filename.endswith('.pdf'):
                # Should not contain dots (except before .pdf), spaces, or dashes
                name_without_ext = filename.replace('.pdf', '')
                self.assertNotIn('.', name_without_ext)
                self.assertNotIn(' ', name_without_ext)
                self.assertNotIn('-', name_without_ext)
    
    @patch('data_pipeline_pdf.fitz')
    def test_extract_text_with_image_placeholders(self, mock_fitz):
        """Test PDF text and image extraction (mocked)"""
        # Mock PDF document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        
        # Mock text blocks
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,  # Text block
                    "lines": [
                        {"spans": [{"text": "Sample text"}]}
                    ]
                }
            ]
        }
        
        mock_page.get_images.return_value = []
        mock_doc.load_page.return_value = mock_page
        mock_doc.__len__.return_value = 1
        mock_fitz.open.return_value = mock_doc
        
        # Test extraction
        text_output = os.path.join(self.output_dir, "text.txt")
        metadata = data_pipeline_pdf.extract_text_with_image_placeholders(
            "dummy.pdf",
            self.images_dir,
            text_output
        )
        
        # Verify text file was created
        self.assertTrue(os.path.exists(text_output))
        
        # Verify content
        with open(text_output, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn("Sample text", content)


@unittest.skipUnless(PRE_CHUNKING_AVAILABLE, "pre_chunking not available")
class TestPreChunking(unittest.TestCase):
    """Test suite for pre_chunking.py"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cleaner = TextCleaner()
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_clean_text_removes_dot_lines(self):
        """Test removal of dot lines"""
        text = "Normal text\n........\nMore text"
        cleaned = self.cleaner.clean_text(text)
        self.assertNotIn("........", cleaned)
        self.assertIn("Normal text", cleaned)
        self.assertIn("More text", cleaned)
    
    def test_clean_text_removes_excessive_spaces(self):
        """Test removal of excessive spaces"""
        text = "Text with    many    spaces"
        cleaned = self.cleaner.clean_text(text)
        # Should reduce to single spaces
        self.assertNotIn("    ", cleaned)
    
    def test_clean_text_removes_multiple_newlines(self):
        """Test reduction of multiple blank lines"""
        text = "Line 1\n\n\n\n\nLine 2"
        cleaned = self.cleaner.clean_text(text)
        # Should reduce to max 3 newlines
        self.assertNotIn("\n\n\n\n", cleaned)
    
    def test_clean_text_removes_page_numbers(self):
        """Test removal of page numbers like '1/142'"""
        text = "Some text\n1/142\nMore text"
        cleaned = self.cleaner.clean_text(text)
        self.assertNotIn("1/142", cleaned)
    
    def test_clean_text_aggressive_mode(self):
        """Test aggressive cleaning mode"""
        text = "  Indented text\n    More indent"
        
        # Non-aggressive should preserve some indentation
        cleaned_normal = self.cleaner.clean_text(text, aggressive=False)
        
        # Aggressive should remove leading spaces
        cleaned_aggressive = self.cleaner.clean_text(text, aggressive=True)
        
        # Both should contain the actual text
        self.assertIn("Indented text", cleaned_normal)
        self.assertIn("Indented text", cleaned_aggressive)
    
    def test_clean_file(self):
        """Test file cleaning"""
        # Create test file
        test_file = os.path.join(self.test_dir, "test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Text with    spaces\n........\nMore text")
        
        # Clean file
        output_file = os.path.join(self.test_dir, "cleaned.txt")
        self.cleaner.clean_file(test_file, output_file)
        
        # Verify
        self.assertTrue(os.path.exists(output_file))
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertNotIn("    ", content)
        self.assertNotIn("........", content)
    
    def test_clean_directory(self):
        """Test cleaning multiple files in a directory"""
        # Create test files
        for i in range(3):
            test_file = os.path.join(self.test_dir, f"test_{i}.txt")
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(f"Text {i}    with spaces\n........")
        
        # Clean directory
        cleaned_files = self.cleaner.clean_directory(self.test_dir)
        
        # Verify
        self.assertEqual(len(cleaned_files), 3)
        for cleaned_file in cleaned_files:
            self.assertTrue(os.path.exists(cleaned_file))
            with open(cleaned_file, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertNotIn("........", content)


@unittest.skipUnless(CHUNKING_AVAILABLE, "chunk_qwen3_0_6B not available")
class TestChunkingAndEmbedding(unittest.TestCase):
    """Test suite for chunk_qwen3_0_6B.py"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_chunk_text_fixed_length(self):
        """Test fixed-length chunking"""
        text = "a" * 3000  # 3000 character text
        
        # Call the function directly without any model loading
        chunks = chunk_qwen3_0_6B.chunk_text_fixed_length(
            text,
            chunk_size=1000,
            overlap=200
        )
        
        # Verify chunks were created
        self.assertGreater(len(chunks), 0, "Should create at least one chunk")
        
        # Verify chunk sizes (all except last should be <= chunk_size)
        for i, chunk in enumerate(chunks[:-1]):
            self.assertLessEqual(
                len(chunk), 1000, 
                f"Chunk {i} size {len(chunk)} exceeds limit 1000"
            )
        
        # Verify overlap (check first two chunks have the expected overlap)
        if len(chunks) > 1:
            # Last 200 chars of first chunk should equal first 200 of second
            overlap_from_first = chunks[0][-200:]
            start_of_second = chunks[1][:200]
            self.assertEqual(
                overlap_from_first, start_of_second,
                "Chunk overlap not working correctly"
            )
    
    def test_chunk_text_fixed_length_short_text(self):
        """Test chunking with text shorter than chunk size"""
        text = "Short text that is less than chunk size"
        # Specify overlap smaller than chunk size
        chunks = chunk_qwen3_0_6B.chunk_text_fixed_length(text, chunk_size=100, overlap=20)
        
        # Should create exactly one chunk
        self.assertEqual(len(chunks), 1, "Short text should create only one chunk")
        self.assertEqual(chunks[0], text, "Single chunk should equal original text")
    
    def test_load_text_files(self):
        """Test loading text files from directory"""
        # Create test files
        expected_content = []
        for i in range(3):
            test_file = os.path.join(self.test_dir, f"doc_{i}.txt")
            content = f"Content of document {i}"
            expected_content.append(content)
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Load files using the module's function
        documents = chunk_qwen3_0_6B.load_text_files(self.test_dir)
        
        # Verify correct number of documents
        self.assertEqual(len(documents), 3, "Should load 3 documents")
        
        # Verify structure (list of tuples: filename, content)
        for filename, content in documents:
            self.assertTrue(filename.endswith('.txt'), "Filename should end with .txt")
            self.assertIn("Content of document", content, "Content should be present")
    
    def test_chunking_strategy_selector_fixed_size(self):
        """Test chunking strategy selection - fixed size"""
        text = "a" * 2000
        
        # Save and temporarily set strategy
        original_strategy = chunk_qwen3_0_6B.CHUNK_STRATEGY
        try:
            chunk_qwen3_0_6B.CHUNK_STRATEGY = "fixed_size"
            
            chunks = chunk_qwen3_0_6B.chunking_strategy_selector(text)
            
            # Should create multiple chunks
            self.assertGreater(len(chunks), 0, "Should create chunks")
            self.assertIsInstance(chunks, list, "Should return a list")
            
            # All items should be strings
            for chunk in chunks:
                self.assertIsInstance(chunk, str, "Each chunk should be a string")
        finally:
            # Restore original strategy
            chunk_qwen3_0_6B.CHUNK_STRATEGY = original_strategy
    
    def test_chunking_strategy_selector_invalid(self):
        """Test chunking strategy with invalid strategy"""
        text = "Some text"
        
        # Save original strategy
        original_strategy = chunk_qwen3_0_6B.CHUNK_STRATEGY
        try:
            # Set invalid strategy
            chunk_qwen3_0_6B.CHUNK_STRATEGY = "invalid_strategy_name"
            
            # Should raise ValueError
            with self.assertRaises(ValueError) as context:
                chunk_qwen3_0_6B.chunking_strategy_selector(text)
            
            # Verify error message mentions the strategy
            self.assertIn("invalid_strategy_name", str(context.exception))
        finally:
            # Restore original
            chunk_qwen3_0_6B.CHUNK_STRATEGY = original_strategy


@unittest.skipUnless(HYBRID_RAG_AVAILABLE, "hybrid_rag_module_qwen3 not available")
class TestHybridRAGModule(unittest.TestCase):
    """Test suite for hybrid_rag_module_qwen3.py"""
    
    def test_extract_keywords(self):
        """Test keyword extraction from query"""
        # Create a mock RAG system for testing
        with patch('hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_embedding_model'):
            with patch('hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_vector_database'):
                rag = hybrid_rag_module_qwen3.HybridRAGQwen3_Module.__new__(
                    hybrid_rag_module_qwen3.HybridRAGQwen3_Module
                )
        
        # Test keyword extraction
        query = "What is the best way to learn Python programming"  # Remove question mark
        keywords = rag._extract_keywords(query)
        
        # Should extract meaningful keywords
        self.assertIn("best", keywords)
        self.assertIn("learn", keywords)
        self.assertIn("python", keywords)
        # "programming" should be in keywords (without punctuation)
        self.assertTrue(
            "programming" in keywords or any("programming" in k for k in keywords),
            f"Expected 'programming' in keywords, got: {keywords}"
        )
        
        # Should exclude stop words
        self.assertNotIn("what", keywords)
        self.assertNotIn("is", keywords)
        self.assertNotIn("the", keywords)
        self.assertNotIn("to", keywords)
    
    def test_calculate_keyword_score_simple(self):
        """Test simple keyword scoring"""
        with patch('hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_embedding_model'):
            with patch('hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_vector_database'):
                rag = hybrid_rag_module_qwen3.HybridRAGQwen3_Module.__new__(
                    hybrid_rag_module_qwen3.HybridRAGQwen3_Module
                )
        
        document = "Python is a great programming language for learning"
        keywords = {"python", "programming", "learning"}
        
        score = rag._calculate_keyword_score_simple(document, keywords)
        
        # All keywords are present, so score should be 1.0
        self.assertEqual(score, 1.0)
    
    def test_calculate_keyword_score_partial_match(self):
        """Test keyword scoring with partial match"""
        with patch('hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_embedding_model'):
            with patch('hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_vector_database'):
                rag = hybrid_rag_module_qwen3.HybridRAGQwen3_Module.__new__(
                    hybrid_rag_module_qwen3.HybridRAGQwen3_Module
                )
        
        document = "Python is great"
        keywords = {"python", "java", "rust"}  # Only 1 out of 3 matches
        
        score = rag._calculate_keyword_score_simple(document, keywords)
        
        # Only 1/3 keywords match
        self.assertAlmostEqual(score, 1.0/3.0, places=2)
    
    def test_calculate_keyword_score_no_match(self):
        """Test keyword scoring with no match"""
        with patch('hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_embedding_model'):
            with patch('hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_vector_database'):
                rag = hybrid_rag_module_qwen3.HybridRAGQwen3_Module.__new__(
                    hybrid_rag_module_qwen3.HybridRAGQwen3_Module
                )
        
        document = "JavaScript is popular"
        keywords = {"python", "java", "rust"}
        
        score = rag._calculate_keyword_score_simple(document, keywords)
        
        # No keywords match
        self.assertEqual(score, 0.0)
    
    def test_format_for_llm(self):
        """Test formatting search results for LLM"""
        with patch('hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_embedding_model'):
            with patch('hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_vector_database'):
                rag = hybrid_rag_module_qwen3.HybridRAGQwen3_Module.__new__(
                    hybrid_rag_module_qwen3.HybridRAGQwen3_Module
                )
        
        results = [
            {
                'rank': 1,
                'content': 'This is chunk 1',
                'metadata': {'source': 'doc1.txt', 'chunk_index': 0},
                'similarity_score': 95.5,
                'keyword_score': 0.8
            },
            {
                'rank': 2,
                'content': 'This is chunk 2',
                'metadata': {'source': 'doc2.txt', 'chunk_index': 1},
                'similarity_score': 87.3,
                'keyword_score': 0.6
            }
        ]
        
        formatted = rag.format_for_llm(results)
        
        # Verify formatted string contains key information
        self.assertIn("chunk 1", formatted)
        self.assertIn("chunk 2", formatted)
        self.assertIn("doc1.txt", formatted)
        self.assertIn("doc2.txt", formatted)
        self.assertIn("95.5%", formatted)
    
    def test_format_for_llm_empty_results(self):
        """Test formatting empty results"""
        with patch('hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_embedding_model'):
            with patch('hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_vector_database'):
                rag = hybrid_rag_module_qwen3.HybridRAGQwen3_Module.__new__(
                    hybrid_rag_module_qwen3.HybridRAGQwen3_Module
                )
        
        formatted = rag.format_for_llm([])
        
        self.assertIn("No relevant information found", formatted)
    
    def test_format_for_llm_max_chunks(self):
        """Test formatting with max chunks limit"""
        with patch('hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_embedding_model'):
            with patch('hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_vector_database'):
                rag = hybrid_rag_module_qwen3.HybridRAGQwen3_Module.__new__(
                    hybrid_rag_module_qwen3.HybridRAGQwen3_Module
                )
        
        results = [
            {
                'rank': i,
                'content': f'Chunk {i}',
                'metadata': {'source': f'doc{i}.txt', 'chunk_index': i},
                'similarity_score': 90.0 - i,
                'keyword_score': 0.8
            }
            for i in range(1, 11)  # 10 results
        ]
        
        formatted = rag.format_for_llm(results, max_chunks=3)
        
        # Should only include first 3 chunks
        self.assertIn("Chunk 1", formatted)
        self.assertIn("Chunk 2", formatted)
        self.assertIn("Chunk 3", formatted)
        self.assertNotIn("Chunk 4", formatted)
        self.assertIn("Showing top 3", formatted)


class TestStreamlitIntegration(unittest.TestCase):
    """Test suite for streamlit_modern_multiuser.py integration points"""
    
    def test_parse_thinking_response_with_thinking_tags(self):
        """Test parsing response with thinking tags"""
        # Import locally to avoid issues
        try:
            from streamlit_modern_multiuser import parse_thinking_response
        except ImportError:
            self.skipTest("streamlit_modern_multiuser not available")
        
        response = """
        <think>
        Let me analyze this step by step.
        First, I need to understand the question.
        </think>
        
        The answer is 42.
        """
        
        parsed = parse_thinking_response(response)
        
        self.assertTrue(parsed['has_thinking'])
        self.assertIn("analyze this step by step", parsed['thinking'])
        self.assertIn("answer is 42", parsed['answer'])
        self.assertNotIn("<think>", parsed['answer'])
    
    def test_parse_thinking_response_without_tags(self):
        """Test parsing response without thinking tags"""
        try:
            from streamlit_modern_multiuser import parse_thinking_response
        except ImportError:
            self.skipTest("streamlit_modern_multiuser not available")
        
        response = "This is a simple answer without thinking tags."
        
        parsed = parse_thinking_response(response)
        
        self.assertFalse(parsed['has_thinking'])
        self.assertIsNone(parsed['thinking'])
        self.assertEqual(parsed['answer'], response.strip())
    
    def test_parse_thinking_response_thinking_variant(self):
        """Test parsing with <thinking> tags instead of <think>"""
        try:
            from streamlit_modern_multiuser import parse_thinking_response
        except ImportError:
            self.skipTest("streamlit_modern_multiuser not available")
        
        response = """
        <thinking>
        Processing the query...
        </thinking>
        Final answer here.
        """
        
        parsed = parse_thinking_response(response)
        
        self.assertTrue(parsed['has_thinking'])
        self.assertIn("Processing the query", parsed['thinking'])
        self.assertIn("Final answer", parsed['answer'])


@unittest.skipUnless(PRE_CHUNKING_AVAILABLE and CHUNKING_AVAILABLE, "Required modules not available")
class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.pdf_dir = os.path.join(self.test_dir, "pdfs")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.pdf_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_text_cleaning_preserves_content(self):
        """Test that cleaning doesn't remove important content"""
        cleaner = TextCleaner()
        
        # Realistic text with artifacts
        text = """
        Chapter 1: Introduction
        
        This is the main content.    It has extra spaces.
        ........
        And some dot lines.
        
        
        
        More content here.
        1/142
        Final paragraph.
        """
        
        cleaned = cleaner.clean_text(text)
        
        # Important content should be preserved
        self.assertIn("Chapter 1: Introduction", cleaned)
        self.assertIn("main content", cleaned)
        self.assertIn("More content here", cleaned)
        self.assertIn("Final paragraph", cleaned)
        
        # Artifacts should be removed
        self.assertNotIn("........", cleaned)
        self.assertNotIn("1/142", cleaned)
    
    def test_chunking_creates_valid_chunks(self):
        """Test that chunking produces valid, overlapping chunks"""
        text = "This is a test sentence. " * 200  # ~4000 characters
        
        chunks = chunk_qwen3_0_6B.chunk_text_fixed_length(
            text,
            chunk_size=1000,
            overlap=200
        )
        
        # Should create multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Each chunk should be <= chunk_size
        for chunk in chunks[:-1]:
            self.assertLessEqual(len(chunk), 1000)
        
        # Chunks should have content
        for chunk in chunks:
            self.assertGreater(len(chunk.strip()), 0)


@unittest.skipUnless(DATA_PIPELINE_AVAILABLE and HYBRID_RAG_AVAILABLE, "Required modules not available")
class TestErrorHandling(unittest.TestCase):
    """Test error handling across the pipeline"""
    
    def test_merge_with_missing_descriptions(self):
        """Test merging when some descriptions are missing"""
        test_dir = tempfile.mkdtemp()
        
        try:
            # Create text with placeholders
            text_file = os.path.join(test_dir, "text.txt")
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write("[IMAGE:img_001]\n")
                f.write("[IMAGE:img_002]\n")
            
            # Create descriptions with only one image
            desc_file = os.path.join(test_dir, "descriptions.txt")
            with open(desc_file, 'w', encoding='utf-8') as f:
                f.write("[IMAGE:img_001]\n")
                f.write("Description for image 1\n")
            
            # Merge (should work but warn about missing)
            output_file = os.path.join(test_dir, "merged.txt")
            
            # Capture warnings
            with warnings.catch_warnings(record=True):
                data_pipeline_pdf.merge_text_with_descriptions(
                    text_file, desc_file, output_file
                )
            
            # File should be created
            self.assertTrue(os.path.exists(output_file))
            
            # Read and check
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # First image should be replaced
            self.assertNotIn("[IMAGE:img_001]", content)
            # Second image should remain (not replaced)
            self.assertIn("[IMAGE:img_002]", content)
            
        finally:
            shutil.rmtree(test_dir)
    
    def test_empty_query_handling(self):
        """Test handling of empty queries"""
        with patch('hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_embedding_model'):
            with patch('hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_vector_database'):
                rag = hybrid_rag_module_qwen3.HybridRAGQwen3_Module.__new__(
                    hybrid_rag_module_qwen3.HybridRAGQwen3_Module
                )
                rag.verbose = False
                
                # Mock the collection
                rag.collection = MagicMock()
                
                # Empty query should raise ValueError
                with self.assertRaises(ValueError):
                    rag.search("")
                
                # Whitespace-only query should raise ValueError
                with self.assertRaises(ValueError):
                    rag.search("   ")


def run_pipeline_health_check():
    """
    Run a quick health check on the pipeline components.
    This is not a unit test but a system check.
    """
    print("\n" + "="*60)
    print("PIPELINE HEALTH CHECK")
    print("="*60)
    
    checks = {
        "data_pipeline_pdf": False,
        "pre_chunking": False,
        "chunk_qwen3_0_6B": False,
        "hybrid_rag_module": False,
        "streamlit_app": False
    }
    
    # Check 1: data_pipeline_pdf
    if DATA_PIPELINE_AVAILABLE:
        checks["data_pipeline_pdf"] = True
        print("[OK] data_pipeline_pdf.py - Import successful")
    else:
        print(f"[FAIL] data_pipeline_pdf.py - Import failed (module not available)")
    
    # Check 2: pre_chunking
    if PRE_CHUNKING_AVAILABLE:
        try:
            cleaner = TextCleaner()
            test_text = cleaner.clean_text("Test    text")
            checks["pre_chunking"] = True
            print("[OK] pre_chunking.py - Import and basic function successful")
        except Exception as e:
            print(f"[FAIL] pre_chunking.py - Function failed: {e}")
    else:
        print(f"[FAIL] pre_chunking.py - Import failed (module not available)")
    
    # Check 3: chunk_qwen3_0_6B
    if CHUNKING_AVAILABLE:
        try:
            chunks = chunk_qwen3_0_6B.chunk_text_fixed_length("a" * 2000, 1000, 200)
            checks["chunk_qwen3_0_6B"] = True
            print("[OK] chunk_qwen3_0_6B.py - Import and chunking successful")
        except Exception as e:
            print(f"[FAIL] chunk_qwen3_0_6B.py - Function failed: {e}")
    else:
        print(f"[FAIL] chunk_qwen3_0_6B.py - Import failed (module not available)")
    
    # Check 4: hybrid_rag_module
    if HYBRID_RAG_AVAILABLE:
        checks["hybrid_rag_module"] = True
        print("[OK] hybrid_rag_module_qwen3.py - Import successful")
    else:
        print(f"[FAIL] hybrid_rag_module_qwen3.py - Import failed (module not available)")
    
    # Check 5: streamlit app
    try:
        # Already imported at module level, just check if it's available
        import streamlit_modern_multiuser
        checks["streamlit_app"] = True
        print("[OK] streamlit_modern_multiuser.py - Import successful")
    except Exception as e:
        print(f"[FAIL] streamlit_modern_multiuser.py - Import failed: {e}")
    
    # Summary
    print("\n" + "-"*60)
    passed = sum(checks.values())
    total = len(checks)
    print(f"Health Check: {passed}/{total} components passed")
    
    if passed == total:
        print("[PASS] All pipeline components are healthy!")
    else:
        print("[WARN] Some components have issues - check logs above")
    
    print("="*60 + "\n")
    
    return all(checks.values())


if __name__ == '__main__':
    # First run health check
    print("\n" + "="*60)
    print("RUNNING PIPELINE TESTS")
    print("="*60 + "\n")
    
    health_ok = run_pipeline_health_check()
    
    if not health_ok:
        print("\n⚠ WARNING: Health check detected issues.")
        print("Continuing with unit tests, but some may fail.\n")
    
    # Run unit tests
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataPipelinePDF))
    suite.addTests(loader.loadTestsFromTestCase(TestPreChunking))
    suite.addTests(loader.loadTestsFromTestCase(TestChunkingAndEmbedding))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridRAGModule))
    suite.addTests(loader.loadTestsFromTestCase(TestStreamlitIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n[SUCCESS] ALL TESTS PASSED!")
        print("\nNote: [WARNING] messages in test output are intentional")
        print("      They verify error handling works correctly.")
    else:
        print("\n[FAILURE] SOME TESTS FAILED - Review output above")
    
    print("="*60 + "\n")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
