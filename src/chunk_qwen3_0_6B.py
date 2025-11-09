import os
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import time
import numpy as np

# Import configuration
from rag_config import (
    BATCH_SIZE,
    FIXED_SIZE_CHUNK_SIZE,
    FIXED_SIZE_OVERLAP,
    CHUNK_SIZE_MAX_BY_SENTENCE,
    CHUNK_STRATEGY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    COLLECTION_NAME,
    MODEL_CACHE_DIR,
    MAX_EMBEDDING_LENGTH,
    PADDING,
    TRUNCATION,
    VERBOSE_MODE,
    NLTK_QUIET,
    get_input_folder,
    get_db_path,
    get_device,
    get_torch_dtype,
    print_config_summary
)

# Get paths from config
FOLDER_PATH = get_input_folder(cleaned=True)
DB_PATH = get_db_path()

class Qwen3EmbeddingFunction(EmbeddingFunction):
    """Custom embedding function for ChromaDB using Qwen3-Embedding model."""
    def __init__(self, tokenizer, model, device='cuda'):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
    
    def mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on token embeddings."""
        token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for input documents."""
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
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().tolist()

def load_text_files(folder_path: str) -> List[Tuple[str, str]]:
    """Load all txt files from the folder."""
    documents = []
    folder = Path(folder_path)
    
    for file_path in folder.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append((file_path.name, content))
    
    print(f"Loaded {len(documents)} documents")
    return documents

def chunk_text_by_sentence(text: str, chunk_size: int = None) -> List[str]:
    """Split text into chunks based on sentence boundaries."""
    if chunk_size is None:
        chunk_size = CHUNK_SIZE_MAX_BY_SENTENCE
    
    import nltk
    nltk.download('punkt', quiet=NLTK_QUIET)
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def chunk_text_fixed_length(text: str, chunk_size: int = FIXED_SIZE_CHUNK_SIZE, overlap: int = FIXED_SIZE_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    # Validate inputs
    if overlap >= chunk_size:
        overlap = chunk_size - 1  # Adjust overlap to be less than chunk_size
        raise ValueError(f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})")
        
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

def create_embeddings_db(folder_path: str, db_path: str, embedding_fn) -> chromadb.Collection:
    """Process text files and store embeddings in ChromaDB."""
    print(f"\n[Creating Embeddings Database] from folder: {folder_path} to db path: {db_path}")
    
    # Initialize ChromaDB with custom embedding function
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    
    # Load and process documents
    documents = load_text_files(folder_path)
    
    all_chunks = []
    metadatas = []
    ids = []
    chunk_id = 0
    
    print("\nChunking documents...")
    for filename, content in documents:
        chunks = chunking_strategy_selector(content)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadatas.append({
                "source": filename,
                "chunk_index": i
            })
            ids.append(f"{filename}_{chunk_id}")
            chunk_id += 1
    
    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    
    # Add to database in batches (embeddings generated automatically by ChromaDB)
    print("\nAdding chunks to database...")
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(all_chunks))
        collection.add(
            documents=all_chunks[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end]
        )
        print(f"  Progress: {batch_end}/{len(all_chunks)} chunks added")
    
    print(f"\n✓ Successfully added {len(all_chunks)} chunks to the database")
    return collection

def chunking_strategy_selector(text: str) -> List[str]:
    """Select chunking strategy based on global CHUNK_STRATEGY variable."""
    if CHUNK_STRATEGY == "fixed_size":
        return chunk_text_fixed_length(text)
    elif CHUNK_STRATEGY == "by_sentence":
        return chunk_text_by_sentence(text, chunk_size=CHUNK_SIZE_MAX_BY_SENTENCE)
    else:
        raise ValueError(f"Unknown chunking strategy: {CHUNK_STRATEGY}")

def run_chunking_and_db_creation():
    
    # Print configuration summary
    print_config_summary()
    
    print("="*60)
    print(f" CHUNKING AND EMBEDDINGS DATABASE CREATION")
    print("="*60)
    
    # Step 1: Initialize embedding model
    print("\n[1/4] Loading embedding model...")
    start_time = time.time()
    
    device = get_device()
    torch_dtype = get_torch_dtype()
    
    # Load tokenizer and model
    print(f"  Loading tokenizer and model from cache: {MODEL_CACHE_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL,
        cache_dir=MODEL_CACHE_DIR,
        trust_remote_code=True
    )
    
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL,
        cache_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
        torch_dtype=torch_dtype
    ).to(device)
    
    model.eval()  # Set to evaluation mode
    
    embedding_fn = Qwen3EmbeddingFunction(tokenizer, model, device)
    
    print(f"✓ Embedding model loaded in {time.time() - start_time:.2f}s")
    print(f"  Device: {device}")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Embedding dimension: {EMBEDDING_DIMENSION}")
    
    # Step 2: Create/load database
    print("\n[2/4] Setting up vector database...")
    db_start = time.time()

    print("Creating new collection...")
    collection = create_embeddings_db(FOLDER_PATH, DB_PATH, embedding_fn)

    print(f"✓ Database ready in {time.time() - db_start:.2f}s")
    return collection


if __name__ == "__main__":

    # Measure total time
    total_start = time.time()
    collection = run_chunking_and_db_creation()
    if CHUNK_STRATEGY == "fixed_size":
        print(f"✓ Fixed size chunking strategy applied")
        print(f"\n✓ Total time taken: {time.time() - total_start:.2f}s | CHUNKING_STRATEGY: {CHUNK_STRATEGY}, BATCH_SIZE: {BATCH_SIZE}, FIXED_SIZE_CHUNK_SIZE: {FIXED_SIZE_CHUNK_SIZE}, FIXED_SIZE_OVERLAP: {FIXED_SIZE_OVERLAP}") #Total time with some metrics
    else:
        print(f"✓ By sentence chunking strategy applied")
        print(f"\n✓ Total time taken: {time.time() - total_start:.2f}s | CHUNKING_STRATEGY: {CHUNK_STRATEGY}, BATCH_SIZE: {BATCH_SIZE}") #Total time with some metrics
    
    # Verify database contents
    print(f"✓ Collection ready with {collection.count()} chunks")

    if VERBOSE_MODE:
        #Show first 5 entries
        results = collection.get(limit=5)
        for i in range(len(results['ids'])):
            print(f"\n[Entry {i+1}] ID: {results['ids'][i]}")
            print(f"Metadata: {results['metadatas'][i]}")
            print(f"Document snippet: {results['documents'][i][:1500]}...")
            print(f"Metadata: {results['metadatas'][i]}")

        #read back a random embedding with the corresponding vector
        sample_id = results['ids'][0]
        embedding_result = collection.get(ids=[sample_id], include=['embeddings', 'documents', 'metadatas'])
        
        print(f"\n[Embedding Retrieval Test]")
        print(f"Retrieved ID: {sample_id}")
        print(f"Type of embedding_result: {type(embedding_result)}")
        print(f"Keys in embedding_result: {list(embedding_result.keys()) if embedding_result else 'None'}")
        
        # Check if embeddings were retrieved
        if embedding_result is not None and 'embeddings' in embedding_result:
            embeddings_data = embedding_result['embeddings']
            print(f"Type of embeddings: {type(embeddings_data)}")
            print(f"Length of embeddings list: {len(embeddings_data) if embeddings_data is not None else 'None'}")
            
            if embeddings_data is not None and len(embeddings_data) > 0:
                embedding_vector = embeddings_data[0]
                print(f"✓ Embedding vector shape: {len(embedding_vector)}")
                print(f"embedding vector: {embedding_vector}")
                
                # Note: all-MiniLM-L6-v2 actually produces 384-dim embeddings, not 512
                actual_dim = len(embedding_vector)
                print(f"✓ Embedding successfully retrieved with {actual_dim} dimensions")
            else:
                print("✗ No embeddings retrieved - this indicates a problem with the vector database")
        else:
            print("✗ No embeddings key found in result")
        
        # Also test that we can retrieve the document and metadata
        if embedding_result is not None and 'documents' in embedding_result and embedding_result['documents']:
            print(f"✓ Document retrieved: {embedding_result['documents'][0][:200]}...")
        else:
            print("✗ No documents retrieved")
            
        if embedding_result is not None and 'metadatas' in embedding_result and embedding_result['metadatas']:
            print(f"✓ Metadata retrieved: {embedding_result['metadatas'][0]}")
        else:
            print("✗ No metadata retrieved")
            
        print(f"\n✓ Embedding retrieval test completed successfully!")
        
        # Bonus: Test similarity search using the retrieved embedding
        print(f"\n[Similarity Search Test]")
        query_text = "mechanikai tulajdonságok"
        print(f"Query: '{query_text}'")
        
        # Perform a similarity search
        search_results = collection.query(
            query_texts=[query_text],
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )
        
        print(f"Found {len(search_results['ids'][0])} similar chunks:")
        for i, (doc_id, distance, doc, metadata) in enumerate(zip(
            search_results['ids'][0], 
            search_results['distances'][0],
            search_results['documents'][0], 
            search_results['metadatas'][0]
        )):
            print(f"  [{i+1}] ID: {doc_id}")
            print(f"      Distance: {distance:.4f}")
            print(f"      Document: {doc[:500]}...")
            print(f"      Metadata: {metadata}")
            print()
