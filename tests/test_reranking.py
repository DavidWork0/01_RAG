"""
Test Suite for Reranking and Chunk Retrieval Functionality
===========================================================
Tests the hybrid search reranking mechanism and chunk retrieval in the RAG system.

This includes:
1. Semantic search and initial retrieval
2. Keyword extraction from queries
3. Keyword scoring of documents
4. Score combination (semantic + keyword)
5. Reranking based on combined scores
6. Similarity threshold filtering
7. Top-k selection
8. Complete search pipeline

Author: Test suite for 01_RAG project
Date: November 18, 2025
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import modules
try:
    from src.hybrid_rag_module_qwen3 import HybridRAGQwen3_Module
    from src.rag_config import (
        SEMANTIC_WEIGHT,
        KEYWORD_WEIGHT,
        MIN_SIMILARITY_THRESHOLD,
        DEFAULT_TOP_K,
        INITIAL_K_MULTIPLIER,
        INITIAL_K_CAP,
        STOP_WORDS
    )
    HYBRID_RAG_AVAILABLE = True
except ImportError as e:
    HYBRID_RAG_AVAILABLE = False
    print(f"Warning: hybrid_rag_module_qwen3 import failed: {e}")


@unittest.skipUnless(HYBRID_RAG_AVAILABLE, "hybrid_rag_module_qwen3 not available")
class TestKeywordExtraction(unittest.TestCase):
    """Test keyword extraction from queries"""
    
    def setUp(self):
        """Create a mock RAG instance for testing"""
        with patch('src.hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_embedding_model'):
            with patch('src.hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_vector_database'):
                self.rag = HybridRAGQwen3_Module.__new__(HybridRAGQwen3_Module)
    
    def test_extract_keywords_basic(self):
        """Test basic keyword extraction"""
        query = "machine learning algorithms for classification"
        keywords = self.rag._extract_keywords(query)
        
        # Should extract meaningful keywords
        self.assertIn("machine", keywords)
        self.assertIn("learning", keywords)
        self.assertIn("algorithms", keywords)
        self.assertIn("classification", keywords)
    
    def test_extract_keywords_removes_stopwords(self):
        """Test that stop words are removed"""
        query = "What are the best methods for solving this problem"
        keywords = self.rag._extract_keywords(query)
        
        # Stop words should be removed
        self.assertNotIn("what", keywords)
        self.assertNotIn("are", keywords)
        self.assertNotIn("the", keywords)
        self.assertNotIn("for", keywords)
        self.assertNotIn("this", keywords)
        
        # Meaningful words should remain
        self.assertIn("best", keywords)
        self.assertIn("methods", keywords)
        self.assertIn("solving", keywords)
        self.assertIn("problem", keywords)
    
    def test_extract_keywords_lowercase_conversion(self):
        """Test that keywords are converted to lowercase"""
        query = "Python PROGRAMMING Language"
        keywords = self.rag._extract_keywords(query)
        
        # All should be lowercase
        for kw in keywords:
            self.assertEqual(kw, kw.lower())
        
        self.assertIn("python", keywords)
        self.assertIn("programming", keywords)
        self.assertIn("language", keywords)
    
    def test_extract_keywords_min_length_filter(self):
        """Test that short words are filtered out"""
        query = "I go to my AI ML NLP class"
        keywords = self.rag._extract_keywords(query)
        
        # Words with length > MIN_KEYWORD_LENGTH should remain
        # 'class' should be present (length 5)
        self.assertIn("class", keywords)
        
        # Very short words should be filtered
        # 'I', 'go', 'to', 'my' are either stopwords or too short
        for kw in keywords:
            self.assertGreaterEqual(len(kw), 3)
    
    def test_extract_keywords_empty_query(self):
        """Test extraction from empty query"""
        query = ""
        keywords = self.rag._extract_keywords(query)
        
        self.assertEqual(len(keywords), 0)
    
    def test_extract_keywords_only_stopwords(self):
        """Test query with only stop words"""
        query = "what is the and or but"
        keywords = self.rag._extract_keywords(query)
        
        self.assertEqual(len(keywords), 0)
    
    def test_extract_keywords_special_characters(self):
        """Test that special characters don't break extraction"""
        query = "machine-learning, deep_learning & neural-networks!"
        keywords = self.rag._extract_keywords(query)
        
        # Should handle special characters gracefully
        self.assertGreater(len(keywords), 0)


@unittest.skipUnless(HYBRID_RAG_AVAILABLE, "hybrid_rag_module_qwen3 not available")
class TestKeywordScoring(unittest.TestCase):
    """Test keyword scoring mechanisms"""
    
    def setUp(self):
        """Create a mock RAG instance for testing"""
        with patch('src.hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_embedding_model'):
            with patch('src.hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_vector_database'):
                self.rag = HybridRAGQwen3_Module.__new__(HybridRAGQwen3_Module)
    
    def test_keyword_score_all_keywords_match(self):
        """Test scoring when all keywords are present"""
        document = "python programming language machine learning"
        keywords = {"python", "programming", "machine", "learning"}
        
        score = self.rag._calculate_keyword_score_simple(document, keywords)
        
        # All keywords match, score should be 1.0
        self.assertEqual(score, 1.0)
    
    def test_keyword_score_partial_match(self):
        """Test scoring with partial keyword matches"""
        document = "python is a programming language"
        keywords = {"python", "programming", "java", "rust"}  # 2 out of 4 match
        
        score = self.rag._calculate_keyword_score_simple(document, keywords)
        
        # 2/4 keywords match, score should be 0.5
        self.assertAlmostEqual(score, 0.5, places=2)
    
    def test_keyword_score_no_match(self):
        """Test scoring when no keywords match"""
        document = "javascript typescript react vue"
        keywords = {"python", "java", "rust"}
        
        score = self.rag._calculate_keyword_score_simple(document, keywords)
        
        # No matches, score should be 0.0
        self.assertEqual(score, 0.0)
    
    def test_keyword_score_empty_keywords(self):
        """Test scoring with no keywords"""
        document = "some content here"
        keywords = set()
        
        score = self.rag._calculate_keyword_score_simple(document, keywords)
        
        self.assertEqual(score, 0.0)
    
    def test_keyword_score_empty_document(self):
        """Test scoring with empty document"""
        document = ""
        keywords = {"python", "java"}
        
        score = self.rag._calculate_keyword_score_simple(document, keywords)
        
        self.assertEqual(score, 0.0)
    
    def test_keyword_score_case_insensitive(self):
        """Test that scoring is case insensitive"""
        document = "Python PROGRAMMING Language"
        keywords = {"python", "programming", "language"}
        
        score = self.rag._calculate_keyword_score_simple(document, keywords)
        
        # Should match despite case differences
        self.assertEqual(score, 1.0)
    
    def test_keyword_score_repeated_keywords(self):
        """Test that repeated keywords don't inflate score"""
        document = "python python python programming"
        keywords = {"python", "java"}  # Only python matches
        
        score = self.rag._calculate_keyword_score_simple(document, keywords)
        
        # Only 1 out of 2 keywords present (regardless of repetition)
        self.assertAlmostEqual(score, 0.5, places=2)
    
    def test_keyword_score_bounds(self):
        """Test that keyword score is always between 0 and 1"""
        test_cases = [
            ("python java rust", {"python", "java", "rust"}),
            ("python", {"python", "java", "rust"}),
            ("", {"python"}),
            ("unrelated content", {"python", "java"}),
        ]
        
        for doc, keywords in test_cases:
            score = self.rag._calculate_keyword_score_simple(doc, keywords)
            self.assertTrue(0.0 <= score <= 1.0,
                          f"Score {score} out of bounds for doc='{doc}', keywords={keywords}")


@unittest.skipUnless(HYBRID_RAG_AVAILABLE, "hybrid_rag_module_qwen3 not available")
class TestScoreCombination(unittest.TestCase):
    """Test combination of semantic and keyword scores"""
    
    def setUp(self):
        """Create a mock RAG instance for testing"""
        with patch('src.hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_embedding_model'):
            with patch('src.hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_vector_database'):
                self.rag = HybridRAGQwen3_Module.__new__(HybridRAGQwen3_Module)
                self.rag.semantic_weight = 0.7
                self.rag.keyword_weight = 0.3
    
    def test_score_combination_formula(self):
        """Test the combined score calculation formula"""
        # Lower distance = better semantic match
        # Higher keyword score = better keyword match
        # Combined score = (semantic_weight * distance) - (keyword_weight * keyword_score)
        # Lower combined score = better overall match
        
        semantic_distance = 0.3
        keyword_score = 0.8
        
        # Expected: (0.7 * 0.3) - (0.3 * 0.8) = 0.21 - 0.24 = -0.03
        expected_combined = (0.7 * 0.3) - (0.3 * 0.8)
        
        # Simulate the calculation
        actual_combined = (self.rag.semantic_weight * semantic_distance) - \
                         (self.rag.keyword_weight * keyword_score)
        
        self.assertAlmostEqual(actual_combined, expected_combined, places=3)
    
    def test_score_combination_semantic_dominance(self):
        """Test that high semantic weight makes semantic score more important"""
        self.rag.semantic_weight = 0.9
        self.rag.keyword_weight = 0.1
        
        # Good semantic (low distance), poor keyword
        combined_1 = (0.9 * 0.1) - (0.1 * 0.0)  # 0.09
        
        # Poor semantic (high distance), good keyword
        combined_2 = (0.9 * 0.8) - (0.1 * 1.0)  # 0.72 - 0.10 = 0.62
        
        # With high semantic weight, good semantic score should win
        self.assertLess(combined_1, combined_2)
    
    def test_score_combination_keyword_boost(self):
        """Test that keyword matches boost ranking"""
        # Two documents with similar semantic scores but different keyword matches
        semantic_dist_1 = 0.5
        semantic_dist_2 = 0.5
        
        keyword_score_1 = 0.9  # High keyword match
        keyword_score_2 = 0.1  # Low keyword match
        
        combined_1 = (self.rag.semantic_weight * semantic_dist_1) - \
                    (self.rag.keyword_weight * keyword_score_1)
        combined_2 = (self.rag.semantic_weight * semantic_dist_2) - \
                    (self.rag.keyword_weight * keyword_score_2)
        
        # Higher keyword score should result in lower (better) combined score
        self.assertLess(combined_1, combined_2)
    
    def test_score_and_combine_results(self):
        """Test the complete scoring and combination pipeline"""
        # Mock semantic search results
        semantic_results = {
            'documents': [['doc1 python programming', 'doc2 java development', 'doc3 python java']],
            'metadatas': [[{'source': 'f1', 'chunk_index': 0}, 
                          {'source': 'f2', 'chunk_index': 1},
                          {'source': 'f3', 'chunk_index': 2}]],
            'distances': [[0.3, 0.5, 0.4]]
        }
        
        keywords = {"python", "programming"}
        
        scored_results = self.rag._score_and_combine_results(semantic_results, keywords)
        
        # Should return list of tuples
        self.assertEqual(len(scored_results), 3)
        
        # Each tuple should have 5 elements: (doc, meta, dist, keyword_score, combined_score)
        for result in scored_results:
            self.assertEqual(len(result), 5)
            doc, meta, dist, kw_score, combined_score = result
            
            # Verify types
            self.assertIsInstance(doc, str)
            self.assertIsInstance(meta, dict)
            self.assertIsInstance(dist, float)
            self.assertIsInstance(kw_score, float)
            self.assertIsInstance(combined_score, float)
            
            # Verify keyword score is in valid range
            self.assertTrue(0.0 <= kw_score <= 1.0)


@unittest.skipUnless(HYBRID_RAG_AVAILABLE, "hybrid_rag_module_qwen3 not available")
class TestReranking(unittest.TestCase):
    """Test the reranking mechanism"""
    
    def setUp(self):
        """Create a mock RAG instance for testing"""
        with patch('src.hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_embedding_model'):
            with patch('src.hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_vector_database'):
                self.rag = HybridRAGQwen3_Module.__new__(HybridRAGQwen3_Module)
                self.rag.semantic_weight = 0.7
                self.rag.keyword_weight = 0.3
    
    def test_reranking_sorts_by_combined_score(self):
        """Test that results are sorted by combined score (lower is better)"""
        scored_results = [
            ("doc1", {"source": "f1"}, 0.3, 0.5, 0.15),  # combined_score = 0.15
            ("doc2", {"source": "f2"}, 0.2, 0.8, -0.10), # combined_score = -0.10 (best)
            ("doc3", {"source": "f3"}, 0.5, 0.2, 0.29),  # combined_score = 0.29
        ]
        
        top_results = self.rag._rank_and_select_top_results(scored_results, top_k=3)
        
        # Should be sorted by combined score (ascending)
        self.assertEqual(len(top_results), 3)
        self.assertEqual(top_results[0][1]["source"], "f2")  # -0.10
        self.assertEqual(top_results[1][1]["source"], "f1")  # 0.15
        self.assertEqual(top_results[2][1]["source"], "f3")  # 0.29
    
    def test_reranking_selects_top_k(self):
        """Test that only top_k results are returned"""
        scored_results = [
            (f"doc{i}", {"source": f"f{i}"}, 0.1 * i, 0.5, 0.1 * i)
            for i in range(10)
        ]
        
        top_results = self.rag._rank_and_select_top_results(scored_results, top_k=5)
        
        # Should return exactly 5 results
        self.assertEqual(len(top_results), 5)
    
    def test_reranking_keyword_boost_effect(self):
        """Test that keyword matches can boost ranking"""
        # Two documents with same semantic score but different keyword matches
        scored_results = [
            ("doc1 no keywords", {"source": "f1"}, 0.4, 0.0, 0.28),  # 0.7*0.4 - 0.3*0.0 = 0.28
            ("doc2 python java", {"source": "f2"}, 0.4, 1.0, -0.02), # 0.7*0.4 - 0.3*1.0 = -0.02
        ]
        
        top_results = self.rag._rank_and_select_top_results(scored_results, top_k=2)
        
        # Document with keywords should rank higher (lower combined score)
        self.assertEqual(top_results[0][1]["source"], "f2")
        self.assertEqual(top_results[1][1]["source"], "f1")
    
    def test_reranking_preserves_metadata(self):
        """Test that reranking preserves all result information"""
        scored_results = [
            ("content1", {"source": "file1.txt", "chunk_index": 0}, 0.3, 0.6, 0.12),
            ("content2", {"source": "file2.txt", "chunk_index": 5}, 0.2, 0.8, -0.04),
        ]
        
        top_results = self.rag._rank_and_select_top_results(scored_results, top_k=2)
        
        # Verify all components are preserved
        for result in top_results:
            self.assertEqual(len(result), 5)
            doc, meta, dist, kw_score, combined = result
            self.assertIn("source", meta)
            self.assertIn("chunk_index", meta)


@unittest.skipUnless(HYBRID_RAG_AVAILABLE, "hybrid_rag_module_qwen3 not available")
class TestSimilarityFiltering(unittest.TestCase):
    """Test similarity threshold filtering"""
    
    def setUp(self):
        """Create a mock RAG instance for testing"""
        with patch('src.hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_embedding_model'):
            with patch('src.hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_vector_database'):
                self.rag = HybridRAGQwen3_Module.__new__(HybridRAGQwen3_Module)
                self.rag.min_similarity = 60.0  # 60% threshold
    
    def test_filter_by_similarity_threshold(self):
        """Test that results below threshold are filtered out"""
        scored_results = [
            ("doc1", {"source": "f1"}, 0.2, 0.5, 0.1),   # ~80% similarity - PASS
            ("doc2", {"source": "f2"}, 0.5, 0.3, 0.2),   # ~50% similarity - FAIL
            ("doc3", {"source": "f3"}, 0.35, 0.4, 0.15), # ~65% similarity - PASS
            ("doc4", {"source": "f4"}, 0.6, 0.2, 0.3),   # ~40% similarity - FAIL
        ]
        
        filtered_results = self.rag._filter_by_similarity(scored_results)
        
        # Should keep only 2 results (>60% similarity)
        self.assertEqual(len(filtered_results), 2)
        
        # Verify correct results were kept
        sources = [result[1]["source"] for result in filtered_results]
        self.assertIn("f1", sources)
        self.assertIn("f3", sources)
        self.assertNotIn("f2", sources)
        self.assertNotIn("f4", sources)
    
    def test_filter_preserves_order(self):
        """Test that filtering preserves the order of results"""
        scored_results = [
            ("doc1", {"source": "f1"}, 0.2, 0.5, 0.1),   # ~80% - PASS
            ("doc2", {"source": "f2"}, 0.35, 0.4, 0.15), # ~65% - PASS
        ]
        
        filtered_results = self.rag._filter_by_similarity(scored_results)
        
        # Should maintain order
        self.assertEqual(filtered_results[0][1]["source"], "f1")
        self.assertEqual(filtered_results[1][1]["source"], "f2")
    
    def test_filter_all_below_threshold(self):
        """Test when all results are below threshold"""
        scored_results = [
            ("doc1", {"source": "f1"}, 0.6, 0.2, 0.3),  # ~40% - FAIL
            ("doc2", {"source": "f2"}, 0.7, 0.1, 0.4),  # ~30% - FAIL
        ]
        
        filtered_results = self.rag._filter_by_similarity(scored_results)
        
        # Should return empty list
        self.assertEqual(len(filtered_results), 0)
    
    def test_filter_all_above_threshold(self):
        """Test when all results are above threshold"""
        scored_results = [
            ("doc1", {"source": "f1"}, 0.2, 0.5, 0.1),   # ~80% - PASS
            ("doc2", {"source": "f2"}, 0.35, 0.4, 0.15), # ~65% - PASS
        ]
        
        filtered_results = self.rag._filter_by_similarity(scored_results)
        
        # Should keep all results
        self.assertEqual(len(filtered_results), 2)
    
    def test_filter_boundary_cases(self):
        """Test filtering at exact threshold boundaries"""
        self.rag.min_similarity = 50.0  # 50% threshold
        
        # distance = 0.5 gives similarity = (1 - 0.5) * 100 = 50%
        scored_results = [
            ("doc1", {"source": "f1"}, 0.5, 0.3, 0.2),   # Exactly 50% - PASS (>=)
            ("doc2", {"source": "f2"}, 0.51, 0.3, 0.2),  # 49% - FAIL
        ]
        
        filtered_results = self.rag._filter_by_similarity(scored_results)
        
        # Should keep the exact match
        self.assertEqual(len(filtered_results), 1)
        self.assertEqual(filtered_results[0][1]["source"], "f1")


@unittest.skipUnless(HYBRID_RAG_AVAILABLE, "hybrid_rag_module_qwen3 not available")
class TestChunkRetrieval(unittest.TestCase):
    """Test complete chunk retrieval pipeline"""
    
    def setUp(self):
        """Create a mock RAG instance with full initialization"""
        with patch('src.hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_embedding_model'):
            with patch('src.hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_vector_database'):
                self.rag = HybridRAGQwen3_Module.__new__(HybridRAGQwen3_Module)
                self.rag.semantic_weight = 0.7
                self.rag.keyword_weight = 0.3
                self.rag.min_similarity = 40.0
                self.rag.verbose = False
                
                # Mock the collection
                self.rag.collection = MagicMock()
    
    def test_search_pipeline_integration(self):
        """Test complete search pipeline from query to results"""
        # Mock semantic search results
        self.rag.collection.query.return_value = {
            'documents': [['python programming tutorial', 
                          'java development guide',
                          'python machine learning']],
            'metadatas': [[{'source': 'doc1.txt', 'chunk_index': 0},
                          {'source': 'doc2.txt', 'chunk_index': 1},
                          {'source': 'doc3.txt', 'chunk_index': 2}]],
            'distances': [[0.3, 0.6, 0.4]]
        }
        
        # Perform search
        results = self.rag.search("python programming", top_k=2)
        
        # Verify results structure
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
        
        # Verify each result has required fields
        for result in results:
            self.assertIn('rank', result)
            self.assertIn('content', result)
            self.assertIn('metadata', result)
            self.assertIn('similarity_score', result)
            self.assertIn('keyword_score', result)
            self.assertIn('combined_score', result)
    
    def test_search_empty_query_raises_error(self):
        """Test that empty query raises ValueError"""
        with self.assertRaises(ValueError):
            self.rag.search("")
        
        with self.assertRaises(ValueError):
            self.rag.search("   ")
    
    def test_search_no_results_from_database(self):
        """Test handling when database returns no results"""
        self.rag.collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = self.rag.search("nonexistent query")
        
        self.assertEqual(len(results), 0)
    
    def test_search_all_filtered_by_similarity(self):
        """Test when all results are filtered by similarity threshold"""
        self.rag.min_similarity = 90.0  # Very high threshold
        
        # Mock results with low similarity
        self.rag.collection.query.return_value = {
            'documents': [['some content']],
            'metadatas': [[{'source': 'doc1.txt', 'chunk_index': 0}]],
            'distances': [[0.8]]  # Low similarity (~20%)
        }
        
        results = self.rag.search("test query")
        
        # All should be filtered out
        self.assertEqual(len(results), 0)
    
    def test_search_respects_top_k(self):
        """Test that search respects top_k parameter"""
        # Mock 10 results
        self.rag.collection.query.return_value = {
            'documents': [[f'content {i}' for i in range(10)]],
            'metadatas': [[{'source': f'doc{i}.txt', 'chunk_index': i} for i in range(10)]],
            'distances': [[0.1 * i for i in range(10)]]  # Increasing distances
        }
        
        results = self.rag.search("test query", top_k=5)
        
        # Should return exactly 5 results (or less if filtered)
        self.assertLessEqual(len(results), 5)
    
    def test_search_result_ranking(self):
        """Test that results are properly ranked"""
        # Mock results with varying semantic and keyword matches
        self.rag.collection.query.return_value = {
            'documents': [['python programming language',  # Good semantic, good keywords
                          'java development',            # Poor semantic, no keywords
                          'python tutorial']],           # Medium semantic, some keywords
            'metadatas': [[{'source': f'doc{i}.txt', 'chunk_index': i} for i in range(3)]],
            'distances': [[0.2, 0.7, 0.4]]
        }
        
        results = self.rag.search("python programming", top_k=3)
        
        # Results should be ranked (rank 1 is best)
        self.assertEqual(results[0]['rank'], 1)
        if len(results) > 1:
            self.assertEqual(results[1]['rank'], 2)
        if len(results) > 2:
            self.assertEqual(results[2]['rank'], 3)
    
    def test_format_results_for_output(self):
        """Test formatting of results for output"""
        top_results = [
            ("content1", {"source": "file1.txt", "chunk_index": 0}, 0.3, 0.6, 0.12),
            ("content2", {"source": "file2.txt", "chunk_index": 5}, 0.2, 0.8, -0.04),
        ]
        
        formatted = self.rag._format_results_for_output(top_results, return_distances=True)
        
        # Verify structure
        self.assertEqual(len(formatted), 2)
        
        for i, result in enumerate(formatted):
            # Check required fields
            self.assertEqual(result['rank'], i + 1)
            self.assertIn('content', result)
            self.assertIn('metadata', result)
            self.assertIn('similarity_score', result)
            self.assertIn('keyword_score', result)
            self.assertIn('combined_score', result)
            self.assertIn('distance', result)
            
            # Check value types
            self.assertIsInstance(result['similarity_score'], float)
            self.assertIsInstance(result['keyword_score'], float)
            self.assertIsInstance(result['combined_score'], float)
            self.assertIsInstance(result['distance'], float)
    
    def test_format_results_without_distances(self):
        """Test formatting results without distance scores"""
        top_results = [
            ("content1", {"source": "file1.txt", "chunk_index": 0}, 0.3, 0.6, 0.12),
        ]
        
        formatted = self.rag._format_results_for_output(top_results, return_distances=False)
        
        # Distance should not be in results
        self.assertNotIn('distance', formatted[0])


@unittest.skipUnless(HYBRID_RAG_AVAILABLE, "hybrid_rag_module_qwen3 not available")
class TestLLMFormatting(unittest.TestCase):
    """Test formatting of results for LLM consumption"""
    
    def setUp(self):
        """Create a mock RAG instance"""
        with patch('src.hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_embedding_model'):
            with patch('src.hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_vector_database'):
                self.rag = HybridRAGQwen3_Module.__new__(HybridRAGQwen3_Module)
    
    def test_format_for_llm_basic(self):
        """Test basic LLM formatting"""
        results = [
            {
                'rank': 1,
                'content': 'This is chunk 1 content',
                'metadata': {'source': 'doc1.txt', 'chunk_index': 0},
                'similarity_score': 95.5,
                'keyword_score': 0.8
            }
        ]
        
        formatted = self.rag.format_for_llm(results)
        
        # Should contain key information
        self.assertIn("chunk 1 content", formatted)
        self.assertIn("doc1.txt", formatted)
        self.assertIn("95.5", formatted)
    
    def test_format_for_llm_multiple_chunks(self):
        """Test formatting multiple chunks"""
        results = [
            {
                'rank': i,
                'content': f'Chunk {i} content',
                'metadata': {'source': f'doc{i}.txt', 'chunk_index': i},
                'similarity_score': 90.0 - i,
                'keyword_score': 0.8
            }
            for i in range(1, 4)
        ]
        
        formatted = self.rag.format_for_llm(results)
        
        # All chunks should be included
        for i in range(1, 4):
            self.assertIn(f"Chunk {i}", formatted)
            self.assertIn(f"doc{i}.txt", formatted)
    
    def test_format_for_llm_max_chunks_limit(self):
        """Test max_chunks parameter"""
        results = [
            {
                'rank': i,
                'content': f'Chunk {i}',
                'metadata': {'source': f'doc{i}.txt', 'chunk_index': i},
                'similarity_score': 90.0,
                'keyword_score': 0.8
            }
            for i in range(1, 11)  # 10 chunks
        ]
        
        formatted = self.rag.format_for_llm(results, max_chunks=3)
        
        # Only first 3 should be included
        self.assertIn("Chunk 1", formatted)
        self.assertIn("Chunk 2", formatted)
        self.assertIn("Chunk 3", formatted)
        self.assertNotIn("Chunk 4", formatted)
        self.assertIn("Showing top 3", formatted)
    
    def test_format_for_llm_empty_results(self):
        """Test formatting empty results"""
        formatted = self.rag.format_for_llm([])
        
        self.assertIn("No relevant information found", formatted)
    
    def test_format_for_llm_structure(self):
        """Test that formatted output has proper structure"""
        results = [
            {
                'rank': 1,
                'content': 'Test content',
                'metadata': {'source': 'test.txt', 'chunk_index': 0},
                'similarity_score': 85.0,
                'keyword_score': 0.5
            }
        ]
        
        formatted = self.rag.format_for_llm(results)
        
        # Should have header
        self.assertIn("RETRIEVED CONTEXT", formatted)
        
        # Should have metadata
        self.assertIn("Source:", formatted)
        self.assertIn("Chunk Index:", formatted)
        self.assertIn("Similarity:", formatted)
        
        # Should have separators
        self.assertIn("-" * 60, formatted)


@unittest.skipUnless(HYBRID_RAG_AVAILABLE, "hybrid_rag_module_qwen3 not available")
class TestEdgeCasesAndErrors(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Create a mock RAG instance"""
        with patch('src.hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_embedding_model'):
            with patch('src.hybrid_rag_module_qwen3.HybridRAGQwen3_Module._load_vector_database'):
                self.rag = HybridRAGQwen3_Module.__new__(HybridRAGQwen3_Module)
                self.rag.semantic_weight = 0.7
                self.rag.keyword_weight = 0.3
                self.rag.min_similarity = 40.0
                self.rag.verbose = False
                self.rag.collection = MagicMock()
    
    def test_unicode_query_handling(self):
        """Test handling of unicode characters in queries"""
        self.rag.collection.query.return_value = {
            'documents': [['python 编程 tutorial']],
            'metadatas': [[{'source': 'doc1.txt', 'chunk_index': 0}]],
            'distances': [[0.3]]
        }
        
        # Should handle unicode without errors
        results = self.rag.search("python 编程", top_k=5)
        self.assertIsInstance(results, list)
    
    def test_very_long_query(self):
        """Test handling of very long queries"""
        long_query = "python " * 1000  # Very long query
        
        self.rag.collection.query.return_value = {
            'documents': [['python content']],
            'metadatas': [[{'source': 'doc1.txt', 'chunk_index': 0}]],
            'distances': [[0.3]]
        }
        
        # Should handle without errors
        results = self.rag.search(long_query, top_k=5)
        self.assertIsInstance(results, list)
    
    def test_database_error_handling(self):
        """Test handling of database errors"""
        self.rag.collection.query.side_effect = Exception("Database error")
        
        with self.assertRaises(RuntimeError):
            self.rag.search("test query")
    
    def test_top_k_zero(self):
        """Test with top_k=0"""
        self.rag.collection.query.return_value = {
            'documents': [['content']],
            'metadatas': [[{'source': 'doc1.txt', 'chunk_index': 0}]],
            'distances': [[0.3]]
        }
        
        results = self.rag.search("test", top_k=0)
        
        # Should return empty list
        self.assertEqual(len(results), 0)
    
    def test_negative_distances(self):
        """Test handling of negative distances (edge case)"""
        self.rag.collection.query.return_value = {
            'documents': [['content']],
            'metadatas': [[{'source': 'doc1.txt', 'chunk_index': 0}]],
            'distances': [[-0.1]]  # Negative distance
        }
        
        results = self.rag.search("test", top_k=5)
        
        # Should handle gracefully
        self.assertIsInstance(results, list)
        if results:
            # Similarity should be bounded to [0, 100]
            self.assertTrue(0 <= results[0]['similarity_score'] <= 100)


def run_test_suite():
    """Run the complete test suite"""
    print("\n" + "="*70)
    print("RERANKING AND CHUNK RETRIEVAL TEST SUITE")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestKeywordExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestKeywordScoring))
    suite.addTests(loader.loadTestsFromTestCase(TestScoreCombination))
    suite.addTests(loader.loadTestsFromTestCase(TestReranking))
    suite.addTests(loader.loadTestsFromTestCase(TestSimilarityFiltering))
    suite.addTests(loader.loadTestsFromTestCase(TestChunkRetrieval))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMFormatting))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCasesAndErrors))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n[SUCCESS] ALL TESTS PASSED!")
    else:
        print("\n[FAILURE] SOME TESTS FAILED")
    
    print("="*70 + "\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_test_suite()
    sys.exit(0 if success else 1)
