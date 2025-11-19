import os
import sys

# Add src to path before importing from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from transformers import AutoTokenizer, AutoModel
    from src.rag_config import MODEL_CACHE_DIR, EMBEDDING_MODEL
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure 'transformers' is installed and you are running from the project root or tests directory.")
    sys.exit(1)

print(f"Model Cache Dir: {MODEL_CACHE_DIR}")
print(f"Embedding Model: {EMBEDDING_MODEL}")

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL,
        cache_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
        local_files_only=True
    )
    print("Tokenizer loaded.")

    print("Loading model...")
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL,
        cache_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
        local_files_only=True
    )
    print("Model loaded.")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
