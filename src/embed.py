import os
from pathlib import Path
from typing import List
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
import torch


class SentenceTransformerEmbeddings(EmbeddingFunction):
    """Custom embedding function for ChromaDB using SentenceTransformer."""
    def __init__(self, model):
        self.model = model
    
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()


def load_text_files(folder_path: str) -> List[tuple[str, str]]:
    """Load all txt files from the folder."""
    documents = []
    folder = Path(folder_path)
    
    for file_path in folder.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append((file_path.name, content))
    
    return documents


def chunk_text(text: str, chunk_size: int = 750, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks


def similarity_search(collection, query: str, top_k: int = 5):
    """Perform similarity search in the ChromaDB collection."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    return results['documents'][0], results['metadatas'][0], results['distances'][0]


def create_embeddings_db(folder_path: str, db_path: str, embedding_fn):
    """Process text files and store embeddings in ChromaDB."""
    # Initialize ChromaDB with custom embedding function
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=embedding_fn
    )
    
    # Load and process documents
    documents = load_text_files(folder_path)
    
    all_chunks = []
    metadatas = []
    ids = []
    chunk_id = 0
    
    for filename, content in documents:
        chunks = chunk_text(content)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadatas.append({
                "source": filename,
                "chunk_index": i
            })
            ids.append(f"{filename}_{chunk_id}")
            chunk_id += 1
    
    # Add to database (embeddings generated automatically by ChromaDB)
    collection.add(
        documents=all_chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Added {len(all_chunks)} chunks to the database")
    return collection

def hybrid_search(collection, query: str, top_k: int = 5):
    """Combine semantic similarity with keyword matching."""
    
    # Semantic search
    semantic_results = collection.query(
        query_texts=[query],
        n_results=top_k * 2
    )
    
    # Keyword filtering - boost results containing query keywords
    keywords = set(query.lower().split())
    scored_results = []
    
    for doc, meta, dist in zip(semantic_results['documents'][0],
                               semantic_results['metadatas'][0],
                               semantic_results['distances'][0]):
        # Calculate keyword match score
        doc_words = set(doc.lower().split())
        keyword_matches = len(keywords.intersection(doc_words))
        
        # Combine semantic distance with keyword score
        # Lower distance is better, higher keyword match is better
        combined_score = dist - (keyword_matches * 0.1)
        
        scored_results.append((doc, meta, combined_score))
    
    # Sort by combined score and return top_k
    scored_results.sort(key=lambda x: x[2])
    return (
        [r[0] for r in scored_results[:top_k]],
        [r[1] for r in scored_results[:top_k]],
        [r[2] for r in scored_results[:top_k]]
    )

if __name__ == "__main__":
    import time

    start_time = time.time()
    folder_path = "./data/output/final_merged"
    db_path = "./data/output/chroma_db"
    
    # Initialize model
    model = SentenceTransformer(
        'all-MiniLM-L6-v2',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        cache_folder='./models/sentence_transformers_all-MiniLM-L6-v2'
    )
    
    # Create embedding function wrapper
    embedding_fn = SentenceTransformerEmbeddings(model)
    
    # Delete existing collection if it exists
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection(name="documents")
        print("Deleted existing collection")
    except Exception as e:
        print(f"No existing collection to delete: {e}")
    
    # Now create the collection with custom embedding function
    create_embeddings_db(folder_path, db_path=db_path, embedding_fn=embedding_fn)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    # Example similarity search
    print("\n\n--- Example Similarity Search ---")
    similarity_search_text = "What processor was used in the Galaxy image study?"
    
    collection = client.get_collection(
        name="documents",
        embedding_function=embedding_fn
    )
    
    documents, metadatas, distances = similarity_search(collection, query=similarity_search_text)
    
    # Display results
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        print(f"\n[Result {i+1}] (Distance: {dist:.4f})")
        print(f"Source: {meta['source']}, Chunk: {meta['chunk_index']}")
        print(f"Content: {doc}...")

