"""
Modern Chat Interface for Hybrid RAG System
===========================================
ChatGPT/Perplexity-style interface with multi-modal support

Author: Generated for 01_RAG project
Date: November 1, 2025
"""

import sys
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

# Add project root to path - resolve to absolute path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import os
# Don't change directory - keep current working directory
# os.chdir(project_root)  # Removed to avoid path issues

import streamlit as st
from typing import Optional, List, Dict
import time
import gc
import re

# Import the hybrid RAG module
try:
    from hybrid_rag_module_qwen3 import HybridRAGQwen3_Module
except ImportError as e:
    st.error(f"Error importing hybrid_rag_module: {e}")
    st.stop()

# =============================================================================
# CONSTANTS
# =============================================================================

TOP_K_RESULTS = 25
DEFAULT_DB_PATH = "data/output/chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024"

AVAILABLE_MODELS = [
    "InternVL3_5-2B-Q6_K",
]

MODEL_CONFIG = {
    "InternVL3_5-2B-Q6_K": {
        "path": "models/llamacpp/InternVL3_5-2B-Q6_K.gguf",
        "key": "llm_internvl",
        "n_ctx": 40960,
        "supports_vision": True
    }
}

# =============================================================================
# PAGE CONFIGURATION - Modern Dark Theme
# =============================================================================

st.set_page_config(
    page_title="RAG Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern chat interface
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #0e1117;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Message bubbles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 18px;
        margin: 10px 0;
        margin-left: auto;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background: #1e2127;
        color: #e8eaed;
        padding: 16px 20px;
        border-radius: 18px;
        margin: 10px 0;
        max-width: 80%;
        border: 1px solid #2d333b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Input area */
    .stTextArea textarea {
        background-color: #1e2127 !important;
        border: 2px solid #2d333b !important;
        border-radius: 12px !important;
        color: #e8eaed !important;
        font-size: 16px !important;
        padding: 12px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e2127 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    /* Thinking process */
    .thinking-box {
        background: #1a1d24;
        border-left: 4px solid #667eea;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 10px 0;
        font-style: italic;
        color: #a8b2c1;
    }
    
    /* Sources */
    .source-chip {
        display: inline-block;
        background: #2d333b;
        color: #8b949e;
        padding: 4px 12px;
        border-radius: 12px;
        margin: 4px;
        font-size: 12px;
        border: 1px solid #373e47;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    /* Image preview */
    .uploaded-image {
        max-width: 200px;
        border-radius: 12px;
        margin: 10px 0;
        border: 2px solid #2d333b;
    }
    
    /* Settings panel */
    .settings-panel {
        background: #1e2127;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid #2d333b;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        'rag_system': None,
        'llm_model': None,
        'llm_loaded': None,
        'chat_history': [],  # List of {role: 'user'/'assistant', content: str, images: [], timestamp: str, sources: []}
        'uploaded_images': [],
        'llm_models': {},
        'model_name': 'InternVL3_5-2B-Q6_K',
        'temperature': 0.7,
        'max_tokens': 4096,
        'top_k': 25,
        'initialized': False,
        'model_loading': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# =============================================================================
# MODEL MANAGEMENT FUNCTIONS
# =============================================================================

def load_llm_model(model_name: str):
    """Load LLM model into session state."""
    if model_name not in MODEL_CONFIG:
        st.error(f"Unknown model: {model_name}")
        return None
    
    # Check if already loaded
    model_key = MODEL_CONFIG[model_name]["key"]
    if model_key in st.session_state.llm_models and st.session_state.llm_models[model_key] is not None:
        return st.session_state.llm_models[model_key]
    
    # Load the model
    try:
        from llama_cpp import Llama
        
        config = MODEL_CONFIG[model_name]
        model_path = project_root / config["path"]
        
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            return None
        
        st.session_state.llm_models[model_key] = Llama(
            model_path=str(model_path),
            n_ctx=config["n_ctx"],
            n_gpu_layers=-1,
            temperature=st.session_state.temperature,
            verbose=False
        )
        
        return st.session_state.llm_models[model_key]
    
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None

def load_rag_system(db_path: str) -> Optional[HybridRAGQwen3_Module]:
    """Load the Hybrid RAG system."""
    try:
        db_path_obj = Path(db_path)
        
        # Ensure absolute path
        if not db_path_obj.is_absolute():
            db_path_obj = project_root / db_path
        
        # Resolve to absolute path
        db_path_obj = db_path_obj.resolve()
        
        if not db_path_obj.exists():
            st.error(f"❌ Database path does not exist: {db_path_obj}")
            st.info(f"💡 Project root: {project_root}")
            st.info(f"💡 Looking for: {project_root / DEFAULT_DB_PATH}")
            
            # Try to find the correct path
            possible_paths = [
                project_root / "data" / "output" / "chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024",
                Path(__file__).parent.parent / "data" / "output" / "chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024",
            ]
            
            st.info("🔍 Checking possible locations:")
            for p in possible_paths:
                exists = p.exists()
                st.info(f"  {'✅' if exists else '❌'} {p}")
            
            return None
        
        rag = HybridRAGQwen3_Module(
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            db_path=str(db_path_obj),
            device='cuda',
            verbose=False
        )
        st.success(f"✅ RAG system loaded from: {db_path_obj}")
        return rag
    except Exception as e:
        st.error(f"❌ Error loading RAG system: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# =============================================================================
# LLM RESPONSE FUNCTIONS
# =============================================================================

def parse_thinking_response(response_text: str) -> Dict:
    """Parse LLM response to separate thinking process from final answer."""
    patterns = [
        (r'<think>(.*?)</think>', 'think'),
        (r'<thinking>(.*?)</thinking>', 'thinking'),
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
    chat_history: List[Dict] = None,
    max_tokens: int = 4096
) -> str:
    """Generate LLM response using retrieved context and chat history."""
    model_name = st.session_state.llm_loaded
    
    if "InternVL" in model_name:
        system_message = """You are a helpful AI assistant specialized in hybrid Retrieval-Augmented Generation (RAG) tasks. Your role is to answer the user's question using both retrieved context from the knowledge base and reasoning based on prior conversation history.

Always:
- Analyze the retrieved context carefully before forming an answer.
- Separate your reasoning process and show it inside <think></think> tags. This section should logically outline how you arrive at your conclusion but should never include guesses unrelated to the provided data.
- Outside the tags, write your final answer clearly, accurately, and concisely in English.
- If information is missing or unclear, state that explicitly instead of assuming or fabricating details.
- Ensure all responses are entirely in English, regardless of the query language.

Example structure:
<think>
Step-by-step reasoning and evidence analysis...
</think>
Final, concise answer in English."""
    else:
        system_message = """You are a helpful AI assistant. Answer the user's question based on the provided context from the knowledge base and the conversation history."""
    
    # Build prompt with chat history
    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n"
    
    # Only add history if it exists and is not empty
    if chat_history and len(chat_history) > 0:
        # Only include last 5 exchanges to avoid context overflow
        recent_history = chat_history[-10:]
        for msg in recent_history:
            if msg['role'] == 'user':
                prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg['role'] == 'assistant':
                prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
    
    # Handle case when no context is provided (no relevant info found)
    if context and context.strip():
        prompt += f"""<|im_start|>user
Context from knowledge base:
{context}

Question: {query}<|im_end|>
<|im_start|>assistant
"""
    else:
        # No context available - LLM answers without RAG
        prompt += f"""<|im_start|>user
⚠️ Note: No relevant information was found in the knowledge base for this question.

Question: {query}

Please answer based on your general knowledge, but clearly state that this answer is not based on the provided documents.<|im_end|>
<|im_start|>assistant
"""
    
    try:
        output = llm_function(
            prompt,
            max_tokens=max_tokens,
            temperature=st.session_state.temperature,
            top_p=0.9,
            echo=False
        )
        return output["choices"][0]["text"]
    except Exception as e:
        return f"Error generating response: {str(e)}"

# =============================================================================
# UI HELPER FUNCTIONS
# =============================================================================

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def display_message(message: Dict):
    """Display a chat message."""
    role = message['role']
    content = message['content']
    timestamp = message.get('timestamp', '')
    
    if role == 'user':
        # User message - right aligned
        col1, col2 = st.columns([1, 4])
        with col2:
            st.markdown(f"""
            <div class="user-message">
                {content}
            </div>
            """, unsafe_allow_html=True)
    
    else:  # assistant
        # Assistant message - left aligned
        col1, col2 = st.columns([4, 1])
        with col1:
            # Check for thinking process
            if message.get('has_thinking') and message.get('thinking'):
                with st.expander("💭 Thinking Process", expanded=False):
                    st.markdown(f"<div class='thinking-box'>{message['thinking']}</div>", unsafe_allow_html=True)
            
            # Display main answer in styled div
            st.markdown(f"""
            <div class="assistant-message">
                {content}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources if available
            if message.get('sources'):
                with st.expander(f"📚 Sources ({len(message['sources'])} chunks)", expanded=False):
                    # Create a scrollable container for sources
                    st.markdown("""
                    <style>
                    .source-container {
                        max-height: 400px;
                        overflow-y: auto;
                        padding: 10px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Display each source with expandable content
                    for idx, result in enumerate(message['sources']):
                        source_name = result['metadata'].get('source', 'Unknown')
                        chunk_idx = result['metadata'].get('chunk_index', 'N/A')
                        similarity = result['similarity_score']
                        chunk_content = result['content']
                        
                        # Each chunk is its own expander
                        with st.expander(
                            f"#{idx+1} - {source_name} (Chunk {chunk_idx}) - {similarity:.1f}% match",
                            expanded=False
                        ):
                            st.text_area(
                                "Chunk Content:",
                                value=chunk_content,
                                height=200,
                                key=f"source_{message.get('timestamp', '')}_{idx}",
                                disabled=True,
                                label_visibility="collapsed"
                            )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Auto-initialize RAG system and model on first run
if not st.session_state.initialized:
    # Load RAG system
    default_db_path = str(project_root / DEFAULT_DB_PATH)
    with st.spinner("Initializing RAG system..."):
        st.session_state.rag_system = load_rag_system(default_db_path)
        if st.session_state.rag_system:
            st.session_state.initialized = True
        else:
            st.error(f"Failed to load RAG system. Expected path: {default_db_path}")
            st.info(f"Project root: {project_root}")
            st.info(f"DB exists: {(project_root / DEFAULT_DB_PATH).exists()}")
    
    # Auto-load model on startup
    if st.session_state.rag_system and not st.session_state.llm_loaded:
        with st.spinner(f"Loading {st.session_state.model_name}..."):
            llm = load_llm_model(st.session_state.model_name)
            if llm:
                st.session_state.llm_model = llm
                st.session_state.llm_loaded = st.session_state.model_name

# Header
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown("""
    <div class="main-header">
        <h1>💬 RAG Chat</h1>
        <p style="color: #8b949e; font-size: 1.3rem;">Ask anything about your documents</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Clear Chat Button - resets the entire chat session and context window
    if st.button("🗑️ Clear", use_container_width=True, key="clear_chat"):
        # Clear all chat-related session state
        st.session_state.chat_history = []
        st.session_state.uploaded_images = []
        
        # Reset the LLM context window (KV cache)
        if st.session_state.llm_model is not None:
            try:
                # Reset the llama-cpp context to clear conversation memory
                st.session_state.llm_model.reset()
            except Exception as e:
                # If reset fails, try to reload the model
                try:
                    model_name = st.session_state.llm_loaded
                    model_key = MODEL_CONFIG[model_name]["key"]
                    if model_key in st.session_state.llm_models:
                        del st.session_state.llm_models[model_key]
                    st.session_state.llm_model = load_llm_model(model_name)
                except:
                    pass
        
        st.rerun()

# Chat Container
chat_container = st.container()

with chat_container:
    # Display chat history
    for message in st.session_state.chat_history:
        display_message(message)

# Fixed input area at bottom
st.markdown("<br><br>", unsafe_allow_html=True)

# Input area
input_container = st.container()

with input_container:
    # Text input
    user_input = st.text_area(
        "Type your message...",
        value="",
        height=100,
        placeholder="Ask me anything about your documents...",
        key="message_input",
        label_visibility="collapsed"
    )
    
    # Send button
    col1, col2 = st.columns([5, 1])
    
    with col2:
        send_button = st.button("📤 Send", type="primary", use_container_width=True)

# Handle send button
if send_button and user_input.strip():
    if not st.session_state.rag_system:
        st.error("❌ RAG system not initialized.")
        st.stop()
    
    if not st.session_state.llm_loaded:
        st.error("❌ Model is still loading. Please wait...")
        st.stop()
    
    # Add user message to history
    user_message = {
        'role': 'user',
        'content': user_input,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.chat_history.append(user_message)
    
    # Clear inputs immediately
    st.session_state.uploaded_images = []
    
    # Generate response
    with st.spinner("🤔 Thinking..."):
        try:
            # Search for relevant chunks
            results = st.session_state.rag_system.search(
                query=user_input,
                top_k=st.session_state.top_k
            )
            
            # Filter out low similarity chunks (keep only > 30% similarity)
            SIMILARITY_THRESHOLD = 30.0  # Minimum similarity score percentage
            filtered_results = [r for r in results if r['similarity_score'] >= SIMILARITY_THRESHOLD]
            
            # Check if no relevant information found
            no_relevant_context = not filtered_results
            
            # If no results pass the threshold, use LLM without context
            if no_relevant_context:
                context = ""  # Empty context - LLM will answer without RAG
                st.warning(f"⚠️ No relevant information found in the knowledge base (all chunks below {SIMILARITY_THRESHOLD}% similarity). Answering with LLM only.")
            else:
                # Format context for LLM using filtered results
                context = st.session_state.rag_system.format_for_llm(filtered_results, max_chunks=None)
            
            # Generate response
            llm_function = st.session_state.llm_model
            
            # Get chat history excluding current user message
            # If only 1 message (current one), pass empty list to start fresh
            history_to_use = st.session_state.chat_history[:-1] if len(st.session_state.chat_history) > 1 else []
            
            response = generate_llm_response(
                llm_function,
                user_input,
                context,
                history_to_use,
                st.session_state.max_tokens
            )
            
            # Parse response
            parsed_response = parse_thinking_response(response)
            
            # Add warning to answer if no relevant context was found
            final_answer = parsed_response['answer']
            if no_relevant_context:
                final_answer = f"⚠️ **No relevant information found in knowledge base** - Answer based on general knowledge:\n\n{final_answer}"
            
            # Add assistant message to history (store filtered results)
            assistant_message = {
                'role': 'assistant',
                'content': final_answer,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'sources': filtered_results,  # Empty if no relevant context
                'has_thinking': parsed_response['has_thinking'],
                'thinking': parsed_response.get('thinking'),
                'no_context': no_relevant_context  # Flag to indicate LLM-only response
            }
            st.session_state.chat_history.append(assistant_message)
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Status bar
if st.session_state.llm_loaded:
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; color: #8b949e; font-size: 0.9rem;">
        ✅ {st.session_state.llm_loaded} ready | 💬 {len(st.session_state.chat_history)//2} exchanges
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #f59e0b; font-size: 0.9rem;">
        ⏳ Loading model, please wait...
    </div>
    """, unsafe_allow_html=True)
