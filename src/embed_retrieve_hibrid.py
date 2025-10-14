import os
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import time


class SentenceTransformerEmbeddings(EmbeddingFunction):
    """Custom embedding function for ChromaDB using SentenceTransformer."""
    def __init__(self, model):
        self.model = model
    
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()


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


def hybrid_search(collection, query: str, top_k: int = 5) -> Tuple[List[str], List[Dict], List[float]]:
    """
    Combine semantic similarity with keyword matching for better retrieval.
    
    Args:
        collection: ChromaDB collection
        query: User query string
        top_k: Number of results to return
        
    Returns:
        Tuple of (documents, metadatas, scores)
    """
    # Get more results initially for re-ranking
    initial_k = top_k * 3
    
    # Semantic search
    semantic_results = collection.query(
        query_texts=[query],
        n_results=initial_k
    )
    #Diploma
    # Extract query keywords (filter out common words)
    stop_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 
                  'was', 'were', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for'}
    keywords = set(word.lower() for word in query.split() 
                   if word.lower() not in stop_words and len(word) > 2)
    
    # Score results combining semantic similarity and keyword matching
    scored_results = []
    
    for doc, meta, dist in zip(semantic_results['documents'][0],
                               semantic_results['metadatas'][0],
                               semantic_results['distances'][0]):
        # Calculate keyword match score
        doc_words = set(doc.lower().split())
        keyword_matches = len(keywords.intersection(doc_words))
        
        # Normalize keyword score (0 to 1)
        keyword_score = keyword_matches / len(keywords) if keywords else 0
        
        # Combine scores (lower distance is better, higher keyword match is better)
        # Weight: 70% semantic, 30% keyword
        combined_score = (0.7 * dist) - (0.3 * keyword_score)
        
        scored_results.append((doc, meta, combined_score, dist))
    
    # Sort by combined score and return top_k
    scored_results.sort(key=lambda x: x[2])
    
    return (
        [r[0] for r in scored_results[:top_k]],
        [r[1] for r in scored_results[:top_k]],
        [r[3] for r in scored_results[:top_k]]  # Return original distances
    )


def answer_question_with_rag(collection, query: str, llm_pipeline, top_k: int = 5) -> Dict:
    """
    Answer questions using RAG: hybrid retrieval + LLM generation.
    
    Args:
        collection: ChromaDB collection
        query: User question
        llm_pipeline: Hugging Face text generation pipeline
        top_k: Number of chunks to retrieve
        
    Returns:
        Dictionary with answer, sources, and context
    """
    print(f"\n{'='*60}")
    print(f"Question: {query}")
    print(f"{'='*60}")
    
    # Step 1: Retrieve relevant chunks using hybrid search
    print("\n[Step 1] Performing hybrid search...")
    documents, metadatas, distances = hybrid_search(collection, query, top_k=top_k)
    
    if not documents:
        return {
            "answer": "No relevant information found in the database.",
            "sources": [],
            "context_chunks": [],
            "distances": []
        }
    
    print(f"Retrieved {len(documents)} relevant chunks")
    for i, (meta, dist) in enumerate(zip(metadatas, distances)):
        print(f"  [{i+1}] {meta['source']} (chunk {meta['chunk_index']}, distance: {dist:.4f})")
    
    # Step 2: Format context from retrieved chunks
    print("\n[Step 2] Building context...")
    context_parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        context_parts.append(f"[Source {i+1}: {meta['source']}, Chunk {meta['chunk_index']}]\n{doc}")
    
    context = "\n\n".join(context_parts)
    
    # Step 3: Create RAG prompt
    print("\n[Step 3] Generating answer with LLM...")
    RAG_TEMPLATE = f"""<|system|>
You are a helpful AI assistant. Use the following context to answer the user's question accurately and concisely. 
If you cannot find the answer in the context, say "I don't have enough information to answer this question."
Always cite the source numbers (e.g., [Source 1]) when using information from the context.
<|end|>

<|user|>
Context:
{context}

Question: {query}
<|end|>

<|assistant|>
"""
    
    # Step 4: Generate answer using LLM
    try:
        outputs = llm_pipeline(
            RAG_TEMPLATE,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        # Extract the generated answer
        answer = outputs[0]['generated_text']
        
        # Remove the prompt from the answer
        if '<|assistant|>' in answer:
            answer = answer.split('<|assistant|>')[-1].strip()
        
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"
    
    # Step 5: Prepare response
    unique_sources = list(set([meta['source'] for meta in metadatas]))
    
    result = {
        "answer": answer,
        "sources": unique_sources,
        "context_chunks": documents,
        "distances": distances,
        "metadatas": metadatas
    }
    
    return result


def create_embeddings_db(folder_path: str, db_path: str, embedding_fn) -> chromadb.Collection:
    """Process text files and store embeddings in ChromaDB."""
    print("\n[Creating Embeddings Database]")
    
    # Initialize ChromaDB with custom embedding function
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=embedding_fn
    )
    
    # Check if collection already has documents
    if collection.count() > 0:
        print(f"Collection already contains {collection.count()} chunks")
        user_input = input("Do you want to recreate it? (yes/no): ").lower()
        if user_input == 'yes':
            client.delete_collection(name="documents")
            collection = client.get_or_create_collection(
                name="documents",
                embedding_function=embedding_fn
            )
        else:
            return collection
    
    # Load and process documents
    documents = load_text_files(folder_path)
    
    all_chunks = []
    metadatas = []
    ids = []
    chunk_id = 0
    
    print("\nChunking documents...")
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
    
    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    
    # Add to database in batches (embeddings generated automatically by ChromaDB)
    batch_size = 100
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


def display_answer(result: Dict):
    """Display the RAG answer in a formatted way."""
    print(f"\n{'='*60}")
    print("ANSWER:")
    print(f"{'='*60}")
    print(result['answer'])
    
    print(f"\n{'='*60}")
    print("SOURCES:")
    print(f"{'='*60}")
    for source in result['sources']:
        print(f"  • {source}")
    
    print(f"\n{'='*60}")
    print("RELEVANT CHUNKS:")
    print(f"{'='*60}")
    for i, (chunk, meta, dist) in enumerate(zip(result['context_chunks'], 
                                                  result['metadatas'],
                                                  result['distances'])):
        print(f"\n[Chunk {i+1}] {meta['source']} (chunk {meta['chunk_index']}, distance: {dist:.4f})")
        print(f"{chunk[:300]}..." if len(chunk) > 300 else chunk)


if __name__ == "__main__":
    
    # Configuration
    FOLDER_PATH = "./data/output/final_merged"  # Folder containing .txt files
    DB_PATH = "./data/output/chroma_db"  # Path to store ChromaDB
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # Better compatibility 
    
    print("="*60)
    print("RAG SYSTEM WITH HYBRID SEARCH")
    print("="*60)
    
    # Step 1: Initialize embedding model
    print("\n[1/4] Loading embedding model...")
    start_time = time.time()
    
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        cache_folder='./models/sentence_transformers_all-MiniLM-L6-v2'
    )
    embedding_fn = SentenceTransformerEmbeddings(embedding_model)
    
    print(f"✓ Embedding model loaded in {time.time() - start_time:.2f}s")
    print(f"  Device: {embedding_model.device}")
    
    # Step 2: Create/load database
    print("\n[2/4] Setting up vector database...")
    db_start = time.time()
    
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Try to get existing collection or create new one
    try:
        collection = client.get_collection(name="documents", embedding_function=embedding_fn)
        print(f"✓ Loaded existing collection with {collection.count()} chunks")
    except:
        print("Creating new collection...")
        collection = create_embeddings_db(FOLDER_PATH, DB_PATH, embedding_fn)
    
    print(f"✓ Database ready in {time.time() - db_start:.2f}s")
    
    # Step 3: Load LLM for answer generation
    print("\n[3/4] Loading language model for answer generation...")
    print(f"  Model: {LLM_MODEL}")
    print("  This may take a few minutes on first run...")
    llm_start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
    model_llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map='auto',
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    
    llm_pipeline = pipeline(
        "text-generation",
        model=model_llm,
        tokenizer=tokenizer,
        max_new_tokens=300,
        return_full_text=False
    )
    
    print(f"✓ Language model loaded in {time.time() - llm_start:.2f}s")
    
    # Step 4: Question answering interface
    print("\n[4/4] Ready for questions!")
    print("="*60)
    
    # Example questions
    example_questions = [
        "What processor was used in the Galaxy image study?",
        "What are the main findings?",
        "Summarize the methodology used in the study"
    ]
    
    # Interactive mode
    print("\nYou can ask questions about your documents.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'examples' to see example questions.\n")
    
    while True:
        user_query = input("\nYour question: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if user_query.lower() == 'examples':
            print("\nExample questions:")
            for i, q in enumerate(example_questions, 1):
                print(f"  {i}. {q}")
            continue
        
        if not user_query:
            continue
        
        # Answer the question
        query_start = time.time()
        result = answer_question_with_rag(collection, user_query, llm_pipeline, top_k=5)
        query_time = time.time() - query_start
        
        # Display results
        display_answer(result)
        print(f"\n⏱ Total time: {query_time:.2f}s")
