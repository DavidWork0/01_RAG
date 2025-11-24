"""Verify random chunk extraction completeness."""
import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.extract_inference_data import extract_from_latest_log

data = extract_from_latest_log()

# Collect all chunks from all questions
all_chunks = []
for q_idx, question in enumerate(data):
    for c_idx, chunk in enumerate(question['chunks']):
        all_chunks.append({
            'question_id': question['question_id'],
            'chunk_num': chunk['chunk_number'],
            'chunk': chunk,
            'q_idx': q_idx,
            'c_idx': c_idx
        })

print(f"\nTotal chunks available: {len(all_chunks)}")

# Select 10 random chunks
random.seed(42)  # For reproducibility
sample_chunks = random.sample(all_chunks, min(10, len(all_chunks)))

print("\n" + "="*80)
print("RANDOM CHUNK VERIFICATION")
print("="*80 + "\n")

for i, item in enumerate(sample_chunks, 1):
    chunk = item['chunk']
    print(f"{i}. Question {item['question_id']}, Chunk {item['chunk_num']}:")
    print(f"   Source: {chunk['source']}")
    print(f"   Index: {chunk['chunk_index']}")
    print(f"   Similarity: {chunk['similarity_score']}%")
    print(f"   Length: {len(chunk['content'])} chars")
    
    # Check if content looks complete (not truncated)
    content = chunk['content']
    has_ellipsis = content.endswith('...')
    has_newlines = '\n' in content
    word_count = len(content.split())
    
    print(f"   Ends with '...': {has_ellipsis}")
    print(f"   Contains newlines: {has_newlines}")
    print(f"   Word count: {word_count}")
    
    # Show first and last 100 chars
    if len(content) > 200:
        print(f"   First 100 chars: {content[:100].replace(chr(10), ' ')[:100]}...")
        print(f"   Last 100 chars: ...{content[-100:].replace(chr(10), ' ')[-100:]}")
    else:
        print(f"   Full content: {content.replace(chr(10), ' ')[:150]}...")
    
    # Check for potential truncation issues
    if len(content) < 50 and not has_ellipsis:
        print("   ⚠️ WARNING: Very short chunk without ellipsis - possible truncation")
    elif not has_newlines and len(content) > 100:
        print("   ⚠️ WARNING: Long chunk without newlines - possible formatting issue")
    else:
        print("   ✅ Looks complete")
    
    print()

print("="*80)
print("VERIFICATION COMPLETE")
print("="*80)
