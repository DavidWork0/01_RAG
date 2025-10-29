# Basic usage
import streamlit as st
from src.hybrid_rag_module_generated import HybridRAG

rag = HybridRAG(
    embedding_model="Qwen/Qwen3-Embedding-0.6B",
    db_path="./data/output/chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024"
)

results = rag.search(query="mechanical properties", top_k=10)
"""
# For Streamlit dashboard
for result in results:
    st.write(f"**{result['metadata']['source']}** - {result['similarity_score']}%")
    st.write(result['content'])
"""

# For LLM context
context = rag.format_for_llm(results, max_chunks=5)
print(context)