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

# Import configuration
from src.rag_config import (
    DEFAULT_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    MODEL_CACHE_DIR,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_THRESHOLD,
    SEMANTIC_WEIGHT,
    KEYWORD_WEIGHT,
    INITIAL_K_MULTIPLIER,
    INITIAL_K_CAP,
    STOP_WORDS,
    MIN_KEYWORD_LENGTH,
    KEYWORD_SCORING_METHOD,
    MAX_EMBEDDING_LENGTH,
    PADDING,
    TRUNCATION,
    VERBOSE_RAG,
    get_device,
    get_torch_dtype
)

# For backward compatibility
DB_PATH = DEFAULT_DB_PATH


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
            padding=PADDING, 
            truncation=TRUNCATION,
            max_length=MAX_EMBEDDING_LENGTH,
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

class HybridRAGQwen3_Module:
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
        embedding_model: str = EMBEDDING_MODEL,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        model_cache_dir: str = MODEL_CACHE_DIR,
        device: Optional[str] = None,
        semantic_weight: float = SEMANTIC_WEIGHT,
        keyword_weight: float = KEYWORD_WEIGHT,
        min_similarity: float = MIN_SIMILARITY_THRESHOLD,
        verbose: bool = VERBOSE_RAG
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
            min_similarity: Minimum similarity threshold (0-100)
            verbose: Print initialization messages
        """
        self.embedding_model = embedding_model
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_cache_dir = model_cache_dir
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.min_similarity = min_similarity
        self.verbose = verbose
        
        # Auto-detect device if not specified
        if device is None:
            self.device = get_device()
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
        self._print(f"\n Loading embedding model...")
        start_time = time.time()
        
        # Load tokenizer and model (offline mode - no internet connection needed)
        tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model,
            cache_dir=self.model_cache_dir,
            trust_remote_code=True,
            local_files_only=True
        )
        
        model = AutoModel.from_pretrained(
            self.embedding_model,
            cache_dir=self.model_cache_dir,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=get_torch_dtype()
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
        self._print(f"\n Loading vector database...")
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"Database not found at {self.db_path}. "
                f"Please run chunk_qwen3_0_6B.py first to create the database."
                #IDEA :  LOOK FOR ANOTHER COLLECTION if neccessary
            )
        
        start_time = time.time()
        
        client = chromadb.PersistentClient(path=self.db_path)
        
        try:
            # Use get_or_create_collection to avoid deserialization issues
            # This will reuse existing data but override the stored embedding function
            # with our current one, preventing "Could not import module" errors
            # when the database was created in a different environment
            collection = client.get_or_create_collection(
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
        keywords = set(
            word.lower() for word in query.split() 
            if word.lower() not in STOP_WORDS and len(word) > MIN_KEYWORD_LENGTH
        )
        
        return keywords
    
    def _calculate_keyword_score_simple(self, document: str, keywords: set) -> float:
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
    
    def _calculate_keyword_score_tfidf(self, document: str, keywords: set) -> float:
        """
        Calculate keyword matching score for a document using TF-IDF weighting.
        
        Args:
            document: Document text
            keywords: Set of query keywords
        Returns:
            Normalized keyword score (0-1)
        """
        if not keywords:
            return 0.0
        
        doc_words = document.lower().split()
        total_words = len(doc_words)
        
        if total_words == 0:
            return 0.0
        
        keyword_count = sum(1 for word in doc_words if word in keywords)
        
        # TF: term frequency
        tf = keyword_count / total_words

        
        return tf  # Returning TF as the score


    def _perform_semantic_search(self, query: str, initial_k: int) -> Dict:
        """
        Step 1: Perform semantic search on the vector database.
        
        Args:
            query: Search query text
            initial_k: Number of initial results to retrieve
            
        Returns:
            Dictionary containing semantic search results
            
        Raises:
            RuntimeError: If database query fails
        """
        try:
            semantic_results = self.collection.query(
                query_texts=[query],
                n_results=initial_k
            )
        except Exception as e:
            raise RuntimeError(f"Error during database query: {str(e)}")
        
        return semantic_results
    
    def _extract_query_keywords(self, query: str) -> set:
        """
        Step 2: Extract keywords from the search query.
        
        Args:
            query: Search query text
            
        Returns:
            Set of extracted keywords
        """
        keywords = self._extract_keywords(query)
        self._print(f"   Keywords: {keywords if keywords else 'none'}")
        return keywords
    
    def _score_and_combine_results(
        self,
        semantic_results: Dict,
        keywords: set
    ) -> List[Tuple]:
        """
        Step 3: Calculate keyword scores and combine with semantic scores.
        
        Args:
            semantic_results: Results from semantic search
            keywords: Extracted query keywords
            
        Returns:
            List of tuples containing (doc, metadata, distance, keyword_score, combined_score)
        """
        scored_results = []
        
        for doc, meta, dist in zip(
            semantic_results['documents'][0],
            semantic_results['metadatas'][0],
            semantic_results['distances'][0]
        ):
            # Calculate keyword match score
            keyword_score = self._calculate_keyword_score_simple(doc, keywords)
            
            # Combine scores
            # Note: Lower distance is better, higher keyword match is better
            combined_score = (self.semantic_weight * dist) - (self.keyword_weight * keyword_score)
            
            scored_results.append((doc, meta, dist, keyword_score, combined_score))
        
        return scored_results
    
    def _filter_by_similarity(
        self,
        scored_results: List[Tuple]
    ) -> List[Tuple]:
        """
        Step 5: Filter results by minimum similarity threshold.
        
        Applied AFTER reranking to ensure keyword matches are properly
        considered before filtering out low-similarity results.
        
        Args:
            scored_results: List of scored results (doc, meta, dist, keyword_score, combined_score)
            
        Returns:
            List of filtered results above similarity threshold
        """
        filtered_results = []
        
        for doc, meta, dist, kw_score, comb_score in scored_results:
            # Calculate similarity percentage (lower distance = higher similarity)
            similarity = max(0, min(100, (1 - dist) * 100))
            
            # Only keep results above the minimum threshold
            if similarity >= self.min_similarity:
                filtered_results.append((doc, meta, dist, kw_score, comb_score))
        
        return filtered_results
    
    def _rank_and_select_top_results(
        self,
        scored_results: List[Tuple],
        top_k: int
    ) -> List[Tuple]:
        """
        Step 4: Sort results by combined score and select top_k.
        
        This reranks results by combining semantic similarity with keyword
        matching scores. Lower combined_score = better result.
        
        Args:
            scored_results: List of scored results
            top_k: Number of top results to return
            
        Returns:
            List of top_k results sorted by combined score
        """
        scored_results.sort(key=lambda x: x[4])  # Sort by combined_score
        top_results = scored_results[:top_k]
        return top_results
    
    def _format_results_for_output(
        self,
        top_results: List[Tuple],
        return_distances: bool
    ) -> List[Dict]:
        """
        Step 7: Format results into structured output dictionaries.
        
        Args:
            top_results: List of top-ranked results
            return_distances: Whether to include distance scores
            
        Returns:
            List of formatted result dictionaries
        """
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
        
        return formatted_results

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        return_distances: bool = True
    ) -> List[Dict]:
        """
        Perform hybrid search and return structured results.
        
        This is the main method for querying the RAG system. It combines
        semantic similarity with keyword matching to find relevant chunks.
        Results are filtered to only include chunks above the minimum
        similarity threshold.
        
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
        
        self._print(f"\n Searching for: '{query}'")
        self._print(f"   Strategy: {int(self.semantic_weight*100)}% semantic + {int(self.keyword_weight*100)}% keyword")
        self._print(f"   Min Similarity: {self.min_similarity}%")
        
        start_time = time.time()
        
        # Step 1: Perform semantic search
        initial_k = min(top_k * INITIAL_K_MULTIPLIER, INITIAL_K_CAP)
        semantic_results = self._perform_semantic_search(query, initial_k)
        
        if not semantic_results['documents'][0]:
            self._print("⚠️  No results found in database")
            return []
        
        # Step 2: Extract keywords from query
        keywords = self._extract_query_keywords(query)
        
        # Step 3: Score and combine results
        scored_results = self._score_and_combine_results(semantic_results, keywords)
        
        # Step 4: Rank ALL results by combined score (BEFORE filtering)
        # This allows keyword matches to boost ranking before similarity filter removes them
        ranked_results = self._rank_and_select_top_results(scored_results, len(scored_results))
        
        # Step 5: Filter by minimum similarity threshold (AFTER reranking)
        filtered_results = self._filter_by_similarity(ranked_results)
        
        if not filtered_results:
            self._print(f"⚠️  No results above similarity threshold ({self.min_similarity}%)")
            return []
        
        # Step 6: Select final top-k from filtered and ranked results
        final_results = filtered_results[:top_k]
        
        # Step 7: Format results for output
        formatted_results = self._format_results_for_output(final_results, return_distances)
        
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
# BELOW FUNCTION TO CREATE RAG SYSTEM and INTERACTIVE MODE when run directly
# =============================================================================

def create_rag_system(
    embedding_model: str = EMBEDDING_MODEL,
    db_path: str = DEFAULT_DB_PATH,
    **kwargs
) -> HybridRAGQwen3_Module:
    """
    Convenience function to create a HybridRAG instance.
    
    Args:
        embedding_model: HuggingFace model identifier
        db_path: Path to ChromaDB database
        **kwargs: Additional arguments passed to HybridRAG constructor
        
    Returns:
        Initialized HybridRAG instance
    """
    return HybridRAGQwen3_Module(embedding_model=embedding_model, db_path=db_path, **kwargs)


# =============================================================================
# MAIN FUNCTION AND INTERACTIVE MODE
# =============================================================================

def main():
    # Initialize the RAG system
    rag_system = create_rag_system(
        embedding_model=EMBEDDING_MODEL,
        db_path=DB_PATH
    )

    # Start interactive mode
    interactive_mode(rag_system.collection)

# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode(collection):
    rag_system = create_rag_system(
        embedding_model=EMBEDDING_MODEL,
        db_path=DB_PATH
    )
    
    print("\n=== Hybrid RAG Interactive Search ===")
    print("Type your query and press Enter. Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("Enter your search query: ").strip()
            if query.lower() in {'exit', 'quit'}:
                print("Exiting interactive mode.")
                break
            
            results = rag_system.search(query, top_k=5, return_distances=True)
            
            if not results:
                print("No results found.\n")
                continue
            
            print(f"\nTop {len(results)} results:\n")
            for res in results:
                print(f"Rank: {res['rank']}")
                print(f"Similarity Score: {res['similarity_score']:.2f}%")
                print(f"Distance: {res.get('distance', 'N/A')}")
                print(f"Keyword Score: {res['keyword_score']:.4f}")
                print(f"Combined Score: {res['combined_score']:.4f}")
                print(f"Source: {res['metadata'].get('source', 'Unknown')}")
                print(f"Chunk Index: {res['metadata'].get('chunk_index', 'N/A')}")
                print(f"Content:\n{res['content']}\n")
                print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            continue

if __name__ == "__main__":
    main()