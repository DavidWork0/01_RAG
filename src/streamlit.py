"""
Streamlit Dashboard for Hybrid RAG Inference
=============================================
Simple interface for querying the Hybrid RAG system.

Author: Generated for 01_RAG project
Date: October 29, 2025
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change working directory to project root for model paths
import os
os.chdir(project_root)

import streamlit as st
from typing import Optional, List, Dict
import time
import gc
import re

# Import the hybrid RAG module
try:
    from hybrid_rag_module_generated import HybridRAG
except ImportError as e:
    st.error(f"Error importing hybrid_rag_module: {e}")
    st.stop()

# =============================================================================
# CONSTANTS
# =============================================================================

TOP_K_RESULTS = 25  # Fixed number of results to retrieve

DEFAULT_DB_PATH = "data/output/chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024"

AVAILABLE_MODELS = [
    "InternVL3_5-2B-Q6_K",
    "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL"
]

MODEL_CONFIG = {
    "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL": {
        "path": "models/llamacpp/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
        "key": "llm_qwen",
        "n_ctx": 65536
    },
    "InternVL3_5-2B-Q6_K": {
        "path": "models/llamacpp/InternVL3_5-2B-Q6_K.gguf",
        "key": "llm_internvl",
        "n_ctx": 40960
    }
}

def load_llm_model(model_name: str):
    """Load LLM model into session state."""
    model_path, model_key = get_model_path(model_name)
    if not model_path or not model_key:
        return None
    
    # Check if already loaded
    if model_key in st.session_state.llm_models and st.session_state.llm_models[model_key] is not None:
        return st.session_state.llm_models[model_key]
    
    # Load the model
    try:
        with st.spinner(f"Loading {model_name}..."):
            from llama_cpp import Llama
            
            config = MODEL_CONFIG[model_name]
            st.session_state.llm_models[model_key] = Llama(
                model_path=str(model_path),
                n_ctx=config["n_ctx"],
                n_gpu_layers=-1,
                temperature=0.7,
                verbose=False
            )
        
        st.success(f"✅ Loaded {model_name}")
        return st.session_state.llm_models[model_key]
    
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Hybrid RAG",
    page_icon="🔍",
    layout="wide"
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        'rag_system': None,
        'search_results': None,
        'llm_response': None,
        'llm_loaded': None,
        'llm_model': None,
        'qa_history': [],
        'current_session_id': 0,
        'last_activity_time': time.time(),
        'auto_cleanup_enabled': False,
        'cleanup_timeout_minutes': 30,
        'initialized': False,
        'skip_auto_load': False,
        'llm_models': {}
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# =============================================================================
# MODEL MANAGEMENT FUNCTIONS
# =============================================================================

def get_model_path(model_name: str) -> tuple:
    """Get the model file path and storage key for a given model name."""
    if model_name not in MODEL_CONFIG:
        st.error(f"Unknown model: {model_name}")
        return None, None
    
    proj_root = Path(__file__).resolve().parent.parent
    config = MODEL_CONFIG[model_name]
    model_path = proj_root / config["path"]
    
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        return None, None
    
    return model_path, config["key"]

def check_and_cleanup_memory():
    """Check if system should auto-cleanup based on inactivity."""
    if not st.session_state.auto_cleanup_enabled:
        return
    
    current_time = time.time()
    time_elapsed = (current_time - st.session_state.last_activity_time) / 60  # Convert to minutes
    
    if time_elapsed >= st.session_state.cleanup_timeout_minutes:
        # Clear LLM models to free VRAM
        clear_all_llm_models()
        
        # Clear session state models
        st.session_state.llm_model = None
        st.session_state.llm_loaded = None
        st.session_state.rag_system = None
        
        st.warning(f"⚠️ System auto-cleaned after {st.session_state.cleanup_timeout_minutes} minutes of inactivity. Please reinitialize to continue.")
        st.session_state.last_activity_time = time.time()

def update_activity_time():
    """Update last activity timestamp."""
    st.session_state.last_activity_time = time.time()

def clear_all_llm_models():
    """Clear all LLM models from memory and VRAM."""
    if 'llm_models' in st.session_state:
        for key in list(st.session_state.llm_models.keys()):
            if st.session_state.llm_models[key] is not None:
                del st.session_state.llm_models[key]
        st.session_state.llm_models = {}
    
    # Force garbage collection
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def switch_llm_model(new_model_name: str):
    """Switch to a different LLM model, clearing the old one."""
    if st.session_state.llm_loaded is not None:
        old_model = st.session_state.llm_loaded
        with st.spinner(f"Unloading {old_model}..."):
            clear_all_llm_models()
            st.session_state.llm_model = None
            st.session_state.llm_loaded = None
            gc.collect()
            time.sleep(0.5)
            st.success(f"✅ Cleared {old_model} from memory")
    
    # Load new model
    llm_function = load_llm_model(new_model_name)
    if llm_function:
        st.session_state.llm_loaded = new_model_name
        st.session_state.llm_model = llm_function

# =============================================================================
# RAG SYSTEM FUNCTIONS
# =============================================================================

def get_default_db_path() -> str:
    """Get the absolute path to the default database."""
    proj_root = Path(__file__).resolve().parent.parent
    return str(proj_root / DEFAULT_DB_PATH)

def load_rag_system(db_path: str) -> Optional[HybridRAG]:
    """Load the Hybrid RAG system."""
    try:
        # Convert to absolute path if relative
        db_path_obj = Path(db_path)
        if not db_path_obj.is_absolute():
            # Recalculate project root from this file's location
            current_file = Path(__file__).resolve()
            proj_root = current_file.parent.parent
            db_path_obj = proj_root / db_path
        
        # Ensure the path exists
        db_path_str = str(db_path_obj.resolve())
        
        if not db_path_obj.exists():
            st.error(f"❌ Database path does not exist: {db_path_str}")
            st.info(f"💡 Current working directory: {os.getcwd()}")
            st.info(f"💡 Project root: {Path(__file__).resolve().parent.parent}")
            return None
        
        with st.spinner("Loading RAG system..."):
            rag = HybridRAG(
                embedding_model="Qwen/Qwen3-Embedding-0.6B",
                db_path=db_path_str,
                device='cuda',
                verbose=False
            )
        st.success("✅ RAG system loaded!")
        return rag
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


# =============================================================================
# LLM RESPONSE FUNCTIONS
# =============================================================================

def parse_thinking_response(response_text: str) -> Dict:
    """Parse LLM response to separate thinking process from final answer."""
    patterns = [
        (r'<think>(.*?)</think>', 'think'),
        (r'<thinking>(.*?)</thinking>', 'thinking'),
        (r'<thoughts>(.*?)</thoughts>', 'thoughts'),
        (r'\[THINKING\](.*?)\[/THINKING\]', 'bracket'),
    ]
    
    for pattern, tag_type in patterns:
        thinking_matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        if thinking_matches:
            thinking = '\n\n'.join([t.strip() for t in thinking_matches])
            final_answer = re.sub(pattern, '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()
            final_answer = re.sub(r'\n{3,}', '\n\n', final_answer).strip()
            
            return {
                'has_thinking': True,
                'thinking': thinking,
                'answer': final_answer if final_answer else "Answer extracted from thinking process."
            }
    
    return {
        'has_thinking': False,
        'thinking': None,
        'answer': response_text.strip()
    }


def generate_llm_response(
    llm_function,
    query: str,
    context: str,
    conversation_history: List[Dict] = None,
    max_tokens: int = 2048
) -> str:
    """Generate LLM response using retrieved context and conversation history."""
    system_message = """You are a helpful AI assistant. Answer the user's question based on the provided context from the knowledge base and the conversation history.

Before providing your final answer, show your reasoning process inside <think></think> tags. Then provide your clear, accurate, and concise answer outside the tags.

Example format:
<think>
Let me analyze the context... The key points are... Therefore...
</think>
Based on the analysis, the answer is..."""
    
    # Build prompt with conversation history
    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n"
    
    if conversation_history:
        for qa in conversation_history:
            prompt += f"<|im_start|>user\n{qa['question']}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{qa['answer']}<|im_end|>\n"
    
    prompt += f"""<|im_start|>user
Context from knowledge base:
{context}

Question: {query}<|im_end|>
<|im_start|>assistant
"""
    
    try:
        output = llm_function(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            echo=False
        )
        return output["choices"][0]["text"]
    except Exception as e:
        return f"Error generating response: {str(e)}"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

st.title("AUDI RAG System")

# Check for auto-cleanup
check_and_cleanup_memory()

# Auto-initialize RAG system on first run
if not st.session_state.initialized and st.session_state.rag_system is None:
    current_file = Path(__file__).resolve()
    proj_root = current_file.parent.parent
    default_db_path = str(proj_root / "data" / "output" / "chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024")
    
    st.session_state.rag_system = load_rag_system(default_db_path)
    st.session_state.initialized = True

# Sidebar - Configuration
with st.sidebar:
    st.header("Configuration")
    
    # Database Path - use absolute path
    current_file = Path(__file__).resolve()
    proj_root = current_file.parent.parent
    default_db_path = str(proj_root / "data" / "output" / "chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024")
    
    db_path = st.text_input(
        "Database Path:",
        value=default_db_path,
        help="Full path to your ChromaDB database"
    )
    
    # Show debug info
    with st.expander("🔧 Debug Info"):
        st.text(f"Project root: {proj_root}")
        st.text(f"Current dir: {os.getcwd()}")
        st.text(f"DB exists: {Path(default_db_path).exists()}")
    
    # Initialize Button
    if st.button("Initialize System", type="primary"):
        st.session_state.rag_system = load_rag_system(db_path)
    
    st.divider()
    
    # LLM Configuration
    st.subheader("LLM Settings")
    
    llm_model_display = st.selectbox(
        "LLM Model:",
        [
            "InternVL3_5-2B-Q6_K",
            "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL"
        ],
        help="Choose the language model for inference."
    )
    
    # Clean model name (remove warning text)
    llm_model = llm_model_display.split(" (⚠️")[0]
    
    # Only load/switch model if needed (not already loaded or user changed selection)
    should_load_model = (
        not st.session_state.skip_auto_load and 
        st.session_state.llm_loaded != llm_model
    )
    
    if should_load_model:
        # Clear previous model from memory if switching models
        if st.session_state.llm_loaded is not None:
            old_model = st.session_state.llm_loaded
            with st.spinner(f"Unloading {old_model}..."):
                # Clear all models using helper function
                clear_all_llm_models()
                
                # Clear session state
                st.session_state.llm_model = None
                st.session_state.llm_loaded = None
                
                # Force garbage collection with delay
                import gc
                import time
                gc.collect()
                time.sleep(0.5)
                
                st.success(f"✅ Cleared {old_model} from memory")
        
        # Load new model
        with st.spinner(f"Loading {llm_model}..."):
            llm_function = load_llm_model(llm_model)
            if llm_function:
                st.session_state.llm_loaded = llm_model
                st.session_state.llm_model = llm_function  # Store model in session state
    
    # Reset skip flag after first run
    if st.session_state.skip_auto_load:
        st.session_state.skip_auto_load = False
    
    if st.session_state.llm_loaded:
        st.success(f"✅ {st.session_state.llm_loaded} ready")
    
    st.divider()

# Main Area
if not st.session_state.rag_system:
    st.info("👈 Initialize the RAG system from the sidebar to begin")
    st.stop()

# Create Tabs - Q&A with LLM is default (first tab)
tab1, tab2 = st.tabs(["Q&A with LLM", "RAG Search for source chunks"])

# =============================================================================
# TAB 1: Q&A WITH LLM
# =============================================================================
with tab1:
    st.header("Q&A WITH LLM based on internal knowledge base")
    
    # Session Management
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.subheader(f"Session #{st.session_state.current_session_id + 1}")
    with col2:
        if st.button("🆕 New Session"):
            st.session_state.current_session_id += 1
            st.session_state.llm_response = None
            st.session_state.search_results = None
            st.rerun()
    with col3:
        max_tokens = st.number_input("Max tokens:", min_value=2048, max_value=8192, value=6144, step=2048,
                                     help="Response length", label_visibility="collapsed")
    
    st.divider()
    
    # Display conversation history for current session
    current_session_qa = [qa for qa in st.session_state.qa_history if qa['session_id'] == st.session_state.current_session_id]
    
    if current_session_qa:
        st.subheader("Conversation History")
        for i, qa in enumerate(current_session_qa):
            with st.container():
                st.markdown(f"**Q{i+1}:** {qa['question']}")
                
                # Check if response has thinking process (for thinking models)
                if qa.get('has_thinking', False) and qa.get('thinking'):
                    # Show thinking in expandable section
                    with st.expander("💭 Thinking Process"):
                        st.markdown(qa.get('thinking', ''))
                    
                    # Highlight final answer for thinking models
                    st.markdown(f"**A{i+1}:** :green[{qa['answer']}]")
                else:
                    # Normal answer without thinking
                    st.markdown(f"**A{i+1}:** {qa['answer']}")
                
                # Temporary debug to view full response
                with st.expander("🔍 View Full Response (Debug)"):
                    if 'raw_response' in qa:
                        st.text_area("Raw LLM Output:", value=qa.get('raw_response', ''), height=300, key=f"debug_raw_{i}", disabled=True)
                    st.text_area("Parsed Answer:", value=qa.get('answer', ''), height=200, key=f"debug_answer_{i}", disabled=True)
                    if qa.get('thinking'):
                        st.text_area("Extracted Thinking:", value=qa.get('thinking', ''), height=200, key=f"debug_think_{i}", disabled=True)
                    st.caption(f"Has thinking: {qa.get('has_thinking', False)}")
                
                with st.expander(f"📚 Sources ({qa['num_chunks']} chunks)"):
                    for result in qa['sources']:
                        st.caption(
                            f"**{result['metadata'].get('source', 'Unknown')}** - "
                            f"Chunk: {result['metadata'].get('chunk_index', 'N/A')} - "
                            f"Similarity: {result['similarity_score']:.1f}%"
                        )
                st.divider()
    
    # Question Input
    question = st.text_area(
        "Ask your question:",
        placeholder="e.g., What are the key mechanical properties discussed in the documents?",
        height=100,
        key=f"qa_question_{st.session_state.current_session_id}"
    )
    
    # Generate Answer Button
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        # Check if LLM is loaded via session state
        llm_ready = st.session_state.llm_loaded is not None
        generate_btn = st.button("🤖 Ask", disabled=not question or not llm_ready, type="primary", use_container_width=True)
    
    if generate_btn:
        update_activity_time()  # Update activity timestamp
        with st.spinner("Retrieving relevant information..."):
            try:
                # Search for relevant chunks
                results = st.session_state.rag_system.search(query=question, top_k=TOP_K_RESULTS)
                
                # Format context for LLM
                context = st.session_state.rag_system.format_for_llm(results, max_chunks=None)
                
                # Use the LLM model from session state
                llm_function = st.session_state.llm_model
                
                if llm_function:
                    with st.spinner("Generating answer..."):
                        # Get conversation history for current session
                        current_session_history = [qa for qa in st.session_state.qa_history if qa['session_id'] == st.session_state.current_session_id]
                        
                        # Generate response with conversation context
                        response = generate_llm_response(llm_function, question, context, current_session_history, max_tokens)
                        
                        # Parse response for thinking tags
                        parsed_response = parse_thinking_response(response)
                        
                        # Add to history (store raw response for debugging)
                        qa_entry = {
                            'session_id': st.session_state.current_session_id,
                            'question': question,
                            'answer': parsed_response['answer'],
                            'raw_response': response,  # Store raw for debugging
                            'sources': results,
                            'num_chunks': len(results),
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Only add thinking fields if thinking was detected
                        if parsed_response['has_thinking']:
                            qa_entry['has_thinking'] = True
                            qa_entry['thinking'] = parsed_response['thinking']
                        
                        st.session_state.qa_history.append(qa_entry)
                        
                        st.rerun()
                else:
                    st.error("LLM not loaded. Please check sidebar settings.")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Show all sessions history
    if len(st.session_state.qa_history) > 0:
        with st.expander("📜 All Sessions History"):
            sessions = {}
            for qa in st.session_state.qa_history:
                sid = qa['session_id']
                if sid not in sessions:
                    sessions[sid] = []
                sessions[sid].append(qa)
            
            for sid, qas in sessions.items():
                st.markdown(f"**Session #{sid + 1}** - {len(qas)} Q&A pairs")
                for i, qa in enumerate(qas):
                    st.caption(f"Q: {qa['question'][:100]}...")
                st.divider()

# =============================================================================
# TAB 2: RAG SEARCH
# =============================================================================

with tab2:
    st.header("Search Knowledge Base for chunks")
    
    # Query Input
    query = st.text_input("Enter your search query:", placeholder="What are you looking for?", key="search_query")

    # Search Button
    if st.button("🔍 Search", disabled=not query, key="search_btn"):
        update_activity_time()  # Update activity timestamp
        try:
            results = st.session_state.rag_system.search(query=query, top_k=TOP_K_RESULTS)
            st.session_state.search_results = results
            st.success(f"Found {len(results)} results")
        except Exception as e:
            st.error(f"Search error: {str(e)}")

    # Display Results
    if st.session_state.search_results:
        st.subheader("Results")
        
        for result in st.session_state.search_results:
            with st.expander(
                f"**#{result['rank']}** - {result['metadata'].get('source', 'Unknown')} "
                f"(Similarity: {result['similarity_score']:.1f}%)",
                expanded=(result['rank'] <= 3)
            ):
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Source", result['metadata'].get('source', 'Unknown'))
                col2.metric("Chunk", result['metadata'].get('chunk_index', 'N/A'))
                col3.metric("Similarity", f"{result['similarity_score']:.1f}%")
                
                # Content
                st.markdown("**Content:**")
                st.text_area(
                    "Content",
                    value=result['content'],
                    height=200,
                    key=f"content_{result['rank']}",
                    label_visibility="collapsed"
                )


