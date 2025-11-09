"""
Hybrid RAG Command-Line Interface
==================================
This script provides an easy-to-use command-line interface for Hybrid RAG (Retrieval-Augmented Generation).
It uses the Qwen3-Embedding-0.6B model for embeddings and combines semantic search with keyword matching.

Features:
- Hybrid Search: Combines semantic similarity + keyword matching
- Interactive Question Answering
- Clear output formatting
- Works with existing ChromaDB created by chunk_qwen3_0_6B.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import time
import argparse

# What processor was used in the experiment?

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Database Configuration
DB_PATH = "./data/output/chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024" 
#DB_PATH = "./data/output/chroma_db_by_sentence_Qwen_Qwen3-Embedding-0.6B_1024"
COLLECTION_NAME = "documents"

# Embedding Model Configuration
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024
MODEL_CACHE_DIR = './models/huggingface'

# Search Configuration
DEFAULT_TOP_K = 25  # Number of results to retrieve
SEMANTIC_WEIGHT = 0.75  # Weight for semantic similarity (0-1)
KEYWORD_WEIGHT = 0.25   # Weight for keyword matching (0-1)


# =============================================================================
# EMBEDDING FUNCTION
# =============================================================================

class Qwen3EmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for ChromaDB using Qwen3-Embedding model.
    This class handles the conversion of text to embeddings.
    """
    
    def __init__(self, tokenizer, model, device='cuda'):
        """
        Initialize the embedding function.
        
        Args:
            tokenizer: Hugging Face tokenizer
            model: Qwen3 embedding model
            device: 'cuda' or 'cpu'
        """
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
    
    def mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on token embeddings.
        This averages all token embeddings to get a single document embedding.
        """
        token_embeddings = model_output[0]  # Get token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for input documents.
        
        Args:
            input: List of text documents
            
        Returns:
            List of embedding vectors
        """
        # Tokenize the input texts
        encoded_input = self.tokenizer(
            input, 
            padding=True, 
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform mean pooling
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings (makes cosine similarity = dot product)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().tolist()


# =============================================================================
# HYBRID SEARCH FUNCTION
# =============================================================================

def hybrid_search(collection, query: str, top_k: int = DEFAULT_TOP_K) -> Tuple[List[str], List[Dict], List[float]]:
    """
    Perform hybrid search combining semantic similarity with keyword matching.
    
    How it works:
    1. Semantic Search: Find documents similar in meaning using embeddings
    2. Keyword Matching: Count how many query keywords appear in each document
    3. Combine Scores: Weight and merge both scores for better results
    
    Args:
        collection: ChromaDB collection to search
        query: User's search query
        top_k: Number of results to return
        
    Returns:
        Tuple of (documents, metadatas, distances)
    """
    print(f"\n🔍 Searching for: '{query}'")
    print(f"   Strategy: {int(SEMANTIC_WEIGHT*100)}% semantic + {int(KEYWORD_WEIGHT*100)}% keyword matching")
    
    # Step 1: Get more results initially for re-ranking
    initial_k = min(top_k * 3, 50)  # Get 3x results but cap at 50
    
    try:
        semantic_results = collection.query(
            query_texts=[query],
            n_results=initial_k
        )
    except Exception as e:
        print(f"❌ Error during search: {str(e)}")
        return [], [], []
    
    if not semantic_results['documents'][0]:
        print("⚠️  No results found in database")
        return [], [], []
    
    # Step 2: Extract query keywords (filter out common stop words)
    stop_words = {
        'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 
        'was', 'were', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for',
        'and', 'or', 'but', 'with', 'from', 'of', 'by', 'as'
    }
    
    keywords = set(
        word.lower() for word in query.split() 
        if word.lower() not in stop_words and len(word) > 2
    )
    
    print(f"   Keywords extracted: {keywords if keywords else 'none'}")
    
    # Step 3: Score results combining semantic similarity and keyword matching
    scored_results = []
    
    for doc, meta, dist in zip(
        semantic_results['documents'][0],
        semantic_results['metadatas'][0],
        semantic_results['distances'][0]
    ):
        # Calculate keyword match score
        doc_words = set(doc.lower().split())
        keyword_matches = len(keywords.intersection(doc_words)) if keywords else 0
        
        # Normalize keyword score (0 to 1)
        keyword_score = keyword_matches / len(keywords) if keywords else 0
        
        # Combine scores
        # Note: Lower distance is better, higher keyword match is better
        combined_score = (SEMANTIC_WEIGHT * dist) - (KEYWORD_WEIGHT * keyword_score)
        
        scored_results.append((doc, meta, combined_score, dist, keyword_score))
    
    # Step 4: Sort by combined score and return top_k
    scored_results.sort(key=lambda x: x[2])
    
    print(f"✓ Found {len(scored_results)} results, returning top {top_k}")
    
    return (
        [r[0] for r in scored_results[:top_k]],  # documents
        [r[1] for r in scored_results[:top_k]],  # metadatas
        [r[3] for r in scored_results[:top_k]]   # original distances
    )


# =============================================================================
# RESULT DISPLAY FUNCTIONS
# =============================================================================

def display_results(documents: List[str], metadatas: List[Dict], distances: List[float]):
    """
    Display search results in a clear, formatted way.
    
    Args:
        documents: Retrieved text chunks
        metadatas: Metadata for each chunk (source file, chunk index)
        distances: Distance scores (lower is better)
    """
    if not documents:
        print("\n❌ No results found.")
        return
    
    print(f"\n{'='*80}")
    print(f"📄 SEARCH RESULTS ({len(documents)} chunks)")
    print(f"{'='*80}\n")
    
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
        # Calculate similarity percentage (lower distance = higher similarity)
        similarity = max(0, (1 - dist) * 100)
        
        print(f"[Result {i}]")
        print(f"📂 Source: {meta['source']}")
        print(f"📍 Chunk: {meta['chunk_index']}")
        print(f"📊 Similarity: {similarity:.1f}% (distance: {dist:.4f})")
        print(f"\n📝 Content:")
        print(f"{'-'*80}")
        
        # Display content entirely or truncate if too long
        content = doc[:] + "..." if len(doc) > 500 else doc
        print(content)
        print(f"{'-'*80}\n")


def display_compact_results(documents: List[str], metadatas: List[Dict], distances: List[float]):
    """
    Display search results in a compact format.
    """
    if not documents:
        print("\n❌ No results found.")
        return
    
    print(f"\n{'='*80}")
    print(f"📄 SEARCH RESULTS ({len(documents)} chunks)")
    print(f"{'='*80}\n")
    
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
        similarity = max(0, (1 - dist) * 100)
        print(f"[{i}] {meta['source']} (chunk {meta['chunk_index']}) - Similarity: {similarity:.1f}%")
        content = doc[:150] + "..." if len(doc) > 150 else doc
        print(f"    {content}\n")


# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

def load_embedding_model(device='cuda'):
    """
    Load the Qwen3 embedding model.
    
    Args:
        device: 'cuda' or 'cpu'
        
    Returns:
        Tuple of (tokenizer, model, embedding_function)
    """
    print("\n📦 Loading Qwen3-Embedding-0.6B model...")
    print(f"   Cache directory: {MODEL_CACHE_DIR}")
    print(f"   Device: {device}")
    
    start_time = time.time()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL,
        cache_dir=MODEL_CACHE_DIR,
        trust_remote_code=True
    )
    
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL,
        cache_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    ).to(device)
    
    model.eval()  # Set to evaluation mode
    
    embedding_fn = Qwen3EmbeddingFunction(tokenizer, model, device)
    
    elapsed = time.time() - start_time
    print(f"✓ Model loaded successfully in {elapsed:.2f}s")
    
    return tokenizer, model, embedding_fn


def load_vector_database(embedding_fn, db_path=DB_PATH):
    """
    Load the ChromaDB vector database.
    
    Args:
        embedding_fn: Embedding function to use
        db_path: Path to the database
        
    Returns:
        ChromaDB collection
    """
    print(f"\n💾 Loading vector database...")
    print(f"   Path: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"❌ Error: Database not found at {db_path}")
        print(f"   Please run chunk_qwen3_0_6B.py first to create the database.")
        sys.exit(1)
    
    start_time = time.time()
    
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn
        )
        
        chunk_count = collection.count()
        elapsed = time.time() - start_time
        
        print(f"✓ Database loaded successfully in {elapsed:.2f}s")
        print(f"   Collection: {COLLECTION_NAME}")
        print(f"   Total chunks: {chunk_count}")
        
        return collection
        
    except Exception as e:
        print(f"❌ Error loading collection: {str(e)}")
        print(f"   Make sure the database was created with the same embedding model.")
        sys.exit(1)


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode(collection):
    """
    Run the RAG system in interactive mode.
    
    Args:
        collection: ChromaDB collection to search
    """
    print(f"\n{'='*80}")
    print("🤖 HYBRID RAG - INTERACTIVE MODE")
    print(f"{'='*80}")
    print("\nCommands:")
    print("  - Type your question to search")
    print("  - 'compact' - Toggle compact/detailed view")
    print("  - 'topk N' - Set number of results (e.g., 'topk 10')")
    print("  - 'quit' or 'exit' - Exit the program")
    print(f"\nCurrent settings: top_k={DEFAULT_TOP_K}, view=detailed")
    print(f"{'='*80}\n")
    
    top_k = DEFAULT_TOP_K
    compact_view = False
    
    while True:
        try:
            user_input = input("\n💬 Your query: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'compact':
                compact_view = not compact_view
                print(f"✓ View mode: {'compact' if compact_view else 'detailed'}")
                continue
            
            if user_input.lower().startswith('topk '):
                try:
                    new_k = int(user_input.split()[1])
                    if new_k > 0:
                        top_k = new_k
                        print(f"✓ Number of results set to: {top_k}")
                    else:
                        print("❌ Please provide a positive number")
                except (ValueError, IndexError):
                    print("❌ Usage: topk <number> (e.g., 'topk 10')")
                continue
            
            # Perform search
            query_start = time.time()
            documents, metadatas, distances = hybrid_search(collection, user_input, top_k=top_k)
            query_time = time.time() - query_start
            
            # Display results
            if compact_view:
                display_compact_results(documents, metadatas, distances)
            else:
                display_results(documents, metadatas, distances)
            
            print(f"⏱️  Query completed in {query_time:.2f}s")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            continue


# =============================================================================
# SINGLE QUERY MODE
# =============================================================================

def single_query_mode(collection, query: str, top_k: int, compact: bool):
    """
    Run a single query and exit.
    
    Args:
        collection: ChromaDB collection to search
        query: Search query
        top_k: Number of results
        compact: Use compact view
    """
    print(f"\n{'='*80}")
    print("🤖 HYBRID RAG - SINGLE QUERY MODE")
    print(f"{'='*80}")
    
    query_start = time.time()
    documents, metadatas, distances = hybrid_search(collection, query, top_k=top_k)
    query_time = time.time() - query_start
    
    if compact:
        display_compact_results(documents, metadatas, distances)
    else:
        display_results(documents, metadatas, distances)
    
    print(f"\n⏱️  Query completed in {query_time:.2f}s")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main entry point for the Hybrid RAG command-line interface.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Hybrid RAG Command-Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python hybrid_rag_cmd.py

  Single query:
    python hybrid_rag_cmd.py -q "What are the main findings?"
    
  Custom number of results:
    python hybrid_rag_cmd.py -q "mechanical properties" -k 10
    
  Compact view:
    python hybrid_rag_cmd.py -q "study methodology" --compact
        """
    )
    
    parser.add_argument('-q', '--query', type=str, help='Single query to search (non-interactive mode)')
    parser.add_argument('-k', '--topk', type=int, default=DEFAULT_TOP_K, help=f'Number of results to return (default: {DEFAULT_TOP_K})')
    parser.add_argument('--compact', action='store_true', help='Use compact view for results')
    parser.add_argument('--db', type=str, default=DB_PATH, help='Path to ChromaDB database')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage (default: use CUDA if available)')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print("🚀 HYBRID RAG SYSTEM - Qwen3-Embedding-0.6B")
    print("="*80)
    
    # Determine device
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device == 'cpu' and not args.cpu:
        print("\n⚠️  CUDA not available, using CPU (this will be slower)")
    
    # Initialize system
    print("\n[1/2] Initializing embedding model...")
    tokenizer, model, embedding_fn = load_embedding_model(device=device)
    
    print("\n[2/2] Loading vector database...")
    collection = load_vector_database(embedding_fn, db_path=args.db)
    
    print("\n✅ System ready!")
    
    # Run in appropriate mode
    if args.query:
        # Single query mode
        single_query_mode(collection, args.query, args.topk, args.compact)
    else:
        # Interactive mode
        interactive_mode(collection)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
