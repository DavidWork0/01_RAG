"""
Test script to verify similarity threshold filtering
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.hybrid_rag_module_qwen3 import HybridRAGQwen3_Module

def test_similarity_filtering():
    """Test that similarity filtering works correctly"""
    
    print("="*70)
    print("SIMILARITY FILTERING TEST")
    print("="*70)
    
    # Initialize RAG system with different thresholds
    print("\n1. Testing with default threshold (50%)...")
    rag_default = HybridRAGQwen3_Module(verbose=True)
    
    print("\n2. Testing with lower threshold (30%)...")
    rag_low = HybridRAGQwen3_Module(min_similarity=30.0, verbose=False)
    
    print("\n3. Testing with higher threshold (70%)...")
    rag_high = HybridRAGQwen3_Module(min_similarity=70.0, verbose=False)
    
    # Test query
    test_query = "Can I make an instance owner with the name 'instanceowner' on RHEL?"
    
    print(f"\n{'='*70}")
    print(f"TEST QUERY: {test_query}")
    print(f"{'='*70}")
    
    # Test with default threshold (50%)
    print("\n--- Results with 50% threshold ---")
    results_default = rag_default.search(test_query, top_k=5)
    print(f"Results found: {len(results_default)}")
    if results_default:
        for i, result in enumerate(results_default, 1):
            print(f"  {i}. Similarity: {result['similarity_score']:.1f}%")
    
    # Test with lower threshold (30%)
    print("\n--- Results with 30% threshold ---")
    results_low = rag_low.search(test_query, top_k=5)
    print(f"Results found: {len(results_low)}")
    if results_low:
        for i, result in enumerate(results_low, 1):
            print(f"  {i}. Similarity: {result['similarity_score']:.1f}%")
    
    # Test with higher threshold (70%)
    print("\n--- Results with 70% threshold ---")
    results_high = rag_high.search(test_query, top_k=5)
    print(f"Results found: {len(results_high)}")
    if results_high:
        for i, result in enumerate(results_high, 1):
            print(f"  {i}. Similarity: {result['similarity_score']:.1f}%")
    
    # Verify filtering logic
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")
    
    # Check that all results meet their thresholds
    checks_passed = []
    
    if results_default:
        min_score_default = min(r['similarity_score'] for r in results_default)
        check = min_score_default >= 50.0
        checks_passed.append(check)
        print(f"✓ Default (50%): Min score = {min_score_default:.1f}% {'PASS' if check else 'FAIL'}")
    
    if results_low:
        min_score_low = min(r['similarity_score'] for r in results_low)
        check = min_score_low >= 30.0
        checks_passed.append(check)
        print(f"✓ Low (30%): Min score = {min_score_low:.1f}% {'PASS' if check else 'FAIL'}")
    
    if results_high:
        min_score_high = min(r['similarity_score'] for r in results_high)
        check = min_score_high >= 70.0
        checks_passed.append(check)
        print(f"✓ High (70%): Min score = {min_score_high:.1f}% {'PASS' if check else 'FAIL'}")
    
    # Final result
    print(f"\n{'='*70}")
    if all(checks_passed) and len(checks_passed) > 0:
        print("✅ ALL TESTS PASSED - Similarity filtering is working correctly!")
    else:
        print("❌ SOME TESTS FAILED - Please check the implementation")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    test_similarity_filtering()
