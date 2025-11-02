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

BATCH_SIZE = 20  # Number of chunks to process in each batch 25 for Qwen3-0.6B and >6GB VRAM || 100 for Qwen3-0.6B and 12GB VRAM --> Batch size 20 seems to be the sweetspot for time
FIXED_SIZE_CHUNK_SIZE = 1000  # Size of each text chunk
FIXED_SIZE_OVERLAP = 250  # Overlap between chunks
CHUNK_STRATEGY = "fixed_size"  # fixed_size, by_sentence implemented here
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B" 
EMBEDDING_DIMENSION = 1024  # Embedding vector length for Qwen3-Embedding-0.6B model 
FOLDER_PATH = "./data/output/final_merged"  # Folder containing .txt files
db_type = f"chroma_db_{CHUNK_STRATEGY}_{EMBEDDING_MODEL.replace('/', '_')}_{EMBEDDING_DIMENSION}"
DB_PATH = f"./data/output/{db_type}"  # Path to store ChromaDB

VERBOSE_MODE = False  # Whether to print detailed logs

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
            padding=True, 
            truncation=True, 
            max_length=512,
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

def chunk_text_by_sentence(text: str) -> List[str]:
    """Split text into chunks based on sentence boundaries."""
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= CHUNK_SIZE:
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
    print("\n[Creating Embeddings Database]")
    
    # Initialize ChromaDB with custom embedding function
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="documents", #name of the collection.
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
    batch_size = BATCH_SIZE
    print("\nAdding chunks to database...")
    for i in range(0, len(all_chunks), batch_size):
        batch_end = min(i + batch_size, len(all_chunks))
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
        return chunk_text_by_sentence(text)
    else:
        raise ValueError(f"Unknown chunking strategy: {CHUNK_STRATEGY}")

def run_chunking_and_db_creation():
    
    print("="*60)
    print(f" CHUNKING AND EMBEDDINGS DATABASE CREATION with ChromaDB source: {FOLDER_PATH} _embed_model: {EMBEDDING_MODEL} _chunk strategy: {CHUNK_STRATEGY}")
    print("="*60)
    
    # Step 1: Initialize embedding model
    print("\n[1/4] Loading embedding model...")
    start_time = time.time()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache_dir = './models/huggingface'
    
    # Load tokenizer and model
    print(f"  Loading tokenizer and model from cache: {cache_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL,
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
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

    """
    client = chromadb.PersistentClient(path=DB_PATH)
    # Try to get existing collection or create new one
    try:
        collection = client.get_collection(name="documents", embedding_function=embedding_fn)
        print(f"✓ Loaded existing collection with {collection.count()} chunks")
    except:
        print("Creating new collection...")
        collection = create_embeddings_db(FOLDER_PATH, DB_PATH, embedding_fn)
    """
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
