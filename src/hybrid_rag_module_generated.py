"""
Hybrid RAG Module
=================
This module provides a programmatic interface for Hybrid RAG (Retrieval-Augmented Generation).
It returns structured results suitable for LLM consumption and Streamlit dashboards.

Key Features:
- Query, model, and top_k as input parameters
- Returns all chunks with complete metadata
- Designed for integration with LLMs and dashboards
- Supports multiple embedding models
- Hybrid search: semantic similarity + keyword matching

Usage:
    from hybrid_rag_module import HybridRAG
    
    rag = HybridRAG(
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        db_path="./data/output/chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024"
    )
    
    results = rag.search(
        query="What are mechanical properties?",
        top_k=10
    )
    
    # Access results
    for result in results:
        print(f"Content: {result['content']}")
        print(f"Source: {result['metadata']['source']}")
        print(f"Similarity: {result['similarity_score']}")
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import time


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
# HYBRID RAG CLASS
# =============================================================================

class HybridRAG:
    """
    Hybrid RAG system that combines semantic search with keyword matching.
    
    This class provides a clean interface for:
    - Loading embedding models
    - Connecting to vector databases
    - Performing hybrid searches
    - Returning structured results for downstream applications
    """
    
    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        db_path: str = "./data/output/chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024",
        collection_name: str = "documents",
        model_cache_dir: str = './models/huggingface',
        device: Optional[str] = None,
        semantic_weight: float = 0.75,
        keyword_weight: float = 0.25,
        verbose: bool = True
    ):
        """
        Initialize the Hybrid RAG system.
        
        Args:
            embedding_model: HuggingFace model identifier for embeddings
            db_path: Path to the ChromaDB database
            collection_name: Name of the ChromaDB collection
            model_cache_dir: Directory for cached models
            device: 'cuda', 'cpu', or None (auto-detect)
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
            verbose: Print initialization messages
        """
        self.embedding_model = embedding_model
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_cache_dir = model_cache_dir
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.verbose = verbose
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize components
        self._print(f"Initializing Hybrid RAG System")
        self._print(f"  Embedding Model: {embedding_model}")
        self._print(f"  Database Path: {db_path}")
        self._print(f"  Device: {self.device}")
        
        self.tokenizer, self.model, self.embedding_fn = self._load_embedding_model()
        self.collection = self._load_vector_database()
        
        self._print(f"✓ System initialized successfully")
    
    def _print(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def _load_embedding_model(self):
        """
        Load the embedding model.
        
        Returns:
            Tuple of (tokenizer, model, embedding_function)
        """
        self._print(f"\n📦 Loading embedding model...")
        start_time = time.time()
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model,
            cache_dir=self.model_cache_dir,
            trust_remote_code=True
        )
        
        model = AutoModel.from_pretrained(
            self.embedding_model,
            cache_dir=self.model_cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device)
        
        model.eval()  # Set to evaluation mode
        
        embedding_fn = Qwen3EmbeddingFunction(tokenizer, model, self.device)
        
        elapsed = time.time() - start_time
        self._print(f"✓ Model loaded in {elapsed:.2f}s")
        
        return tokenizer, model, embedding_fn
    
    def _load_vector_database(self):
        """
        Load the ChromaDB vector database.
        
        Returns:
            ChromaDB collection
        """
        self._print(f"\n💾 Loading vector database...")
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"Database not found at {self.db_path}. "
                f"Please run chunk_qwen3_0_6B.py first to create the database."
            )
        
        start_time = time.time()
        
        client = chromadb.PersistentClient(path=self.db_path)
        
        try:
            collection = client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn
            )
            
            chunk_count = collection.count()
            elapsed = time.time() - start_time
            
            self._print(f"✓ Database loaded in {elapsed:.2f}s")
            self._print(f"  Collection: {self.collection_name}")
            self._print(f"  Total chunks: {chunk_count}")
            
            return collection
            
        except Exception as e:
            raise RuntimeError(
                f"Error loading collection '{self.collection_name}': {str(e)}. "
                f"Make sure the database was created with a compatible embedding model."
            )
    
    def _extract_keywords(self, query: str) -> set:
        """
        Extract meaningful keywords from query.
        
        Args:
            query: Search query
            
        Returns:
            Set of keywords
        """
        stop_words = {
            'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 
            'was', 'were', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for',
            'and', 'or', 'but', 'with', 'from', 'of', 'by', 'as', 'that',
            'this', 'these', 'those', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
            'can', 'may', 'might', 'must', 'shall'
        }
        
        keywords = set(
            word.lower() for word in query.split() 
            if word.lower() not in stop_words and len(word) > 2
        )
        
        return keywords
    
    def _calculate_keyword_score(self, document: str, keywords: set) -> float:
        """
        Calculate keyword matching score for a document.
        
        Args:
            document: Document text
            keywords: Set of query keywords
            
        Returns:
            Normalized keyword score (0-1)
        """
        if not keywords:
            return 0.0
        
        doc_words = set(document.lower().split())
        keyword_matches = len(keywords.intersection(doc_words))
        
        # Normalize by number of keywords
        return keyword_matches / len(keywords)
    
    def search(
        self,
        query: str,
        top_k: int = 25,
        return_distances: bool = True
    ) -> List[Dict]:
        """
        Perform hybrid search and return structured results.
        
        This is the main method for querying the RAG system. It combines
        semantic similarity with keyword matching to find relevant chunks.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            return_distances: Include distance scores in results
            
        Returns:
            List of result dictionaries, each containing:
                - content: The text chunk
                - metadata: Dict with source, chunk_index, and any other metadata
                - similarity_score: Similarity percentage (0-100)
                - distance: Raw distance score (if return_distances=True)
                - keyword_score: Keyword matching score (0-1)
                - combined_score: Final hybrid score used for ranking
                - rank: Position in results (1-based)
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        self._print(f"\n🔍 Searching for: '{query}'")
        self._print(f"   Strategy: {int(self.semantic_weight*100)}% semantic + {int(self.keyword_weight*100)}% keyword")
        
        start_time = time.time()
        
        # Step 1: Get more results initially for re-ranking
        initial_k = min(top_k * 3, 100)  # Get 3x results but cap at 100
        
        try:
            semantic_results = self.collection.query(
                query_texts=[query],
                n_results=initial_k
            )
        except Exception as e:
            raise RuntimeError(f"Error during database query: {str(e)}")
        
        if not semantic_results['documents'][0]:
            self._print("⚠️  No results found in database")
            return []
        
        # Step 2: Extract keywords from query
        keywords = self._extract_keywords(query)
        self._print(f"   Keywords: {keywords if keywords else 'none'}")
        
        # Step 3: Score and combine results
        scored_results = []
        
        for doc, meta, dist in zip(
            semantic_results['documents'][0],
            semantic_results['metadatas'][0],
            semantic_results['distances'][0]
        ):
            # Calculate keyword match score
            keyword_score = self._calculate_keyword_score(doc, keywords)
            
            # Combine scores
            # Note: Lower distance is better, higher keyword match is better
            combined_score = (self.semantic_weight * dist) - (self.keyword_weight * keyword_score)
            
            scored_results.append((doc, meta, dist, keyword_score, combined_score))
        
        # Step 4: Sort by combined score and select top_k
        scored_results.sort(key=lambda x: x[4])
        top_results = scored_results[:top_k]
        
        # Step 5: Format results for output
        formatted_results = []
        
        for rank, (doc, meta, dist, kw_score, comb_score) in enumerate(top_results, 1):
            # Calculate similarity percentage (lower distance = higher similarity)
            similarity = max(0, min(100, (1 - dist) * 100))
            
            result = {
                'rank': rank,
                'content': doc,
                'metadata': meta,
                'similarity_score': round(similarity, 2),
                'keyword_score': round(kw_score, 4),
                'combined_score': round(comb_score, 4)
            }
            
            if return_distances:
                result['distance'] = round(dist, 4)
            
            formatted_results.append(result)
        
        elapsed = time.time() - start_time
        self._print(f"✓ Found {len(formatted_results)} results in {elapsed:.2f}s")
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the vector database collection.
        
        Returns:
            Dictionary containing:
                - total_chunks: Total number of chunks
                - collection_name: Name of the collection
                - embedding_model: Model used for embeddings
        """
        return {
            'total_chunks': self.collection.count(),
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model,
            'db_path': self.db_path
        }
    
    def format_for_llm(self, results: List[Dict], max_chunks: Optional[int] = None) -> str:
        """
        Format search results as a context string for LLM consumption.
        
        Args:
            results: List of result dictionaries from search()
            max_chunks: Maximum number of chunks to include (None = all)
            
        Returns:
            Formatted string with all relevant information
        """
        if not results:
            return "No relevant information found in the knowledge base."
        
        chunks_to_use = results[:max_chunks] if max_chunks else results
        
        context_parts = [
            "=== RETRIEVED CONTEXT ===\n",
            f"Found {len(results)} relevant chunks. Showing top {len(chunks_to_use)}.\n"
        ]
        
        for result in chunks_to_use:
            context_parts.append(
                f"\n[Chunk {result['rank']} - Similarity: {result['similarity_score']:.1f}%]\n"
                f"Source: {result['metadata'].get('source', 'Unknown')}\n"
                f"Chunk Index: {result['metadata'].get('chunk_index', 'N/A')}\n"
                f"Content:\n{result['content']}\n"
                f"{'-'*60}"
            )
        
        return '\n'.join(context_parts)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_rag_system(
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    db_path: str = "./data/output/chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024",
    **kwargs
) -> HybridRAG:
    """
    Convenience function to create a HybridRAG instance.
    
    Args:
        embedding_model: HuggingFace model identifier
        db_path: Path to ChromaDB database
        **kwargs: Additional arguments passed to HybridRAG constructor
        
    Returns:
        Initialized HybridRAG instance
    """
    return HybridRAG(embedding_model=embedding_model, db_path=db_path, **kwargs)


# =============================================================================
# EXAMPLE USAGE (when run as script)
# =============================================================================

def main():
    """
    Example usage of the HybridRAG module.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Hybrid RAG Module - Example Usage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hybrid_rag_module.py -q "mechanical properties" -k 10
  python hybrid_rag_module.py -q "study findings" --model "Qwen/Qwen3-Embedding-0.6B"
  python hybrid_rag_module.py -q "methodology" --db "./data/output/my_db" -k 5
        """
    )
    
    parser.add_argument('-q', '--query', required=True, help='Search query')
    parser.add_argument('-k', '--topk', type=int, default=25, help='Number of results (default: 25)')
    parser.add_argument('--model', default="Qwen/Qwen3-Embedding-0.6B", help='Embedding model')
    parser.add_argument('--db', default="./data/output/chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024", 
                       help='Database path')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Create RAG system
    rag = HybridRAG(
        embedding_model=args.model,
        db_path=args.db,
        device='cpu' if args.cpu else None,
        verbose=not args.quiet
    )
    
    # Perform search
    results = rag.search(query=args.query, top_k=args.topk)
    
    # Display results
    print("\n" + "="*80)
    print(f"SEARCH RESULTS ({len(results)} chunks)")
    print("="*80 + "\n")
    
    for result in results:
        print(f"[Rank {result['rank']}]")
        print(f"  Source: {result['metadata']['source']}")
        print(f"  Chunk: {result['metadata']['chunk_index']}")
        print(f"  Similarity: {result['similarity_score']}%")
        print(f"  Keyword Score: {result['keyword_score']}")
        print(f"  Content Preview: {result['content'][:200]}...")
        print()
    
    # Show LLM-formatted output
    print("\n" + "="*80)
    print("LLM-FORMATTED CONTEXT (Top 3)")
    print("="*80 + "\n")
    print(rag.format_for_llm(results, max_chunks=3))


if __name__ == "__main__":
    main()
