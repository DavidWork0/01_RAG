"""
Modern Chat Interface for Hybrid RAG System - Multi-User Support
================================================================
Shared model resources with separate user sessions

MULTI-USER ARCHITECTURE:
------------------------
This application shares a single LLM model instance across all concurrent users
to optimize VRAM usage. Thread locking ensures safe sequential access.

KV CACHE BEHAVIOR:
------------------
- Single shared LLM instance maintains KV (Key-Value) cache state
- Thread lock prevents concurrent access (one user at a time)
- Each user's request includes full conversation history in the prompt
- llama-cpp-python creates independent cache for each new prompt
- No explicit cache reuse between different users' conversations

SAFETY GUARANTEES:
------------------
✓ Thread lock prevents race conditions
✓ Each prompt is self-contained with full context
✓ No explicit cache continuation features used
✓ User sessions are logically isolated

TRADEOFFS:
----------
✓ VRAM Efficient: One model instance (~4-8GB)
✗ Sequential Processing: Users queue when simultaneous requests occur
✓ Cache Isolation: Each call starts with fresh context from prompt

Alternative would be per-user model instances:
  Pros: Perfect isolation, true parallel processing
  Cons: N * model_size VRAM (expensive), N * load time

Author: Generated for 01_RAG project
Date: November 1, 2025
"""



import sys
from pathlib import Path
import uuid

# Add project root to path - resolve to absolute path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from typing import Optional, List, Dict
import time
import re
import threading

# Import the hybrid RAG module
try:
    from hybrid_rag_module_qwen3 import HybridRAGQwen3_Module
except ImportError as e:
    st.error(f"Error importing hybrid_rag_module: {e}")
    st.stop()

# Import shared configuration
try:
    from src.model_config import (
        DEFAULT_DB_PATH,
        TOP_K_RESULTS,
        MODEL_CONFIG,
        DEFAULT_MAX_TOKENS,
        MAX_TOKENS_OPTIONS,
        EMBEDDING_MODEL,
        DEFAULT_MODEL,
        SIMILARITY_THRESHOLD,
        get_system_message,
        get_model_config,
        get_available_models,
        parse_thinking_response,
        load_llm_model,
        generate_llm_response as _generate_llm_response_core,
        PROMPT_TEMPLATE,
        PROMPT_TEMPLATE_WITH_HISTORY
    )
except ImportError as e:
    st.error(f"Error importing model_config: {e}")
    st.stop()

# =============================================================================
# SHARED RESOURCES (CACHED GLOBALLY ACROSS ALL USERS)
# =============================================================================

@st.cache_resource
def get_llm_lock():
    """Get a shared lock for LLM access (must be cached to work across sessions)."""
    return threading.Lock()

@st.cache_resource(show_spinner="Loading RAG system...")
def get_shared_rag_system():
    """Load RAG system once and share across all users."""
    db_path = project_root / DEFAULT_DB_PATH
    
    if not db_path.exists():
        st.error(f"Database not found: {db_path}")
        return None
    
    try:
        rag = HybridRAGQwen3_Module(
            embedding_model=EMBEDDING_MODEL,
            db_path=str(db_path),
            device='cuda',
            verbose=False
        )
        return rag
    except Exception as e:
        st.error(f"Error loading RAG: {e}")
        return None

@st.cache_resource(show_spinner="Loading language model...")
def get_shared_llm_model(model_name: str):
    """
    Load LLM once and share across all users using shared config.
    
    ⚠️ KV CACHE CONSIDERATION:
    This shared model instance maintains KV cache state. While thread locking
    prevents concurrent access (avoiding race conditions), the KV cache from
    one user's conversation may persist when another user accesses the model.
    
    Mitigation:
    - Thread lock ensures sequential access (one user at a time)
    - llama.cpp resets KV cache on new prompts by default
    - For complete isolation, would need per-user model instances (high VRAM cost)
    
    Current tradeoff: VRAM efficiency > Perfect cache isolation
    """
    try:
        llm = load_llm_model(model_name, project_root)
        return llm
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading LLM: {e}")
        return None

# =============================================================================
# SESSION-SPECIFIC STATE (SEPARATE PER USER)
# =============================================================================

def initialize_user_session():
    """Initialize session-specific variables for each user."""
    
    # Generate unique session ID if not exists
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # User-specific chat data
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    
    # User preferences
    if 'model_name' not in st.session_state:
        st.session_state.model_name = DEFAULT_MODEL
    
    if 'temperature' not in st.session_state:
        model_config = get_model_config(DEFAULT_MODEL)
        st.session_state.temperature = model_config['temperature']
    
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = DEFAULT_MAX_TOKENS
    
    if 'top_k' not in st.session_state:
        st.session_state.top_k = TOP_K_RESULTS

initialize_user_session()

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="RAG Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 18px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background: #1e2127;
        color: #e8eaed;
        padding: 16px 20px;
        border-radius: 18px;
        margin: 10px 0;
        border: 1px solid #2d333b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
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
    
    .streamlit-expanderHeader {
        background-color: #1e2127 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    .thinking-box {
        background: #1a1d24;
        border-left: 4px solid #667eea;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 10px 0;
        font-style: italic;
        color: #a8b2c1;
    }
    
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
    
    .stAlert {
        border-radius: 12px !important;
        border-left-width: 4px !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_llm_response(
    llm_function,
    query: str,
    context: str,
    chat_history: List[Dict] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.7,
    status_placeholder = None
) -> str:
    """
    Generate LLM response with thread-safe locking for multi-user support.
    
    This wraps the shared generate_llm_response_core function with thread locking
    and UI status updates specific to Streamlit's multi-user environment.
    """
    # Get the shared lock
    llm_lock = get_llm_lock()
    
    try:
        # Try to acquire lock without blocking first
        lock_acquired = llm_lock.acquire(blocking=False)
        
        if not lock_acquired:
            # Lock is held by another user - show waiting message
            if status_placeholder:
                status_placeholder.warning("⏳ **LLM is currently busy** - Another user is being served. Please wait...")
            
            # Now wait for the lock (blocking)
            llm_lock.acquire()
            lock_acquired = True
            
            # Clear the waiting message once we got the lock
            if status_placeholder:
                status_placeholder.empty()
        
        # At this point, we have the lock
        try:
            # Update status to show we're processing
            if status_placeholder:
                status_placeholder.info("🤔 **Generating response...**")
            
            # CRITICAL: Lock prevents concurrent CUDA operations from different users
            # KV Cache Management: llama.cpp automatically manages cache state
            # Each new prompt starts fresh unless using explicit cache reuse
            # The thread lock ensures one user completes before the next starts
            
            # Use shared core function from model_config
            response = _generate_llm_response_core(
                llm_model=llm_function,
                query=query,
                context=context,
                model_name=st.session_state.model_name,
                max_tokens=max_tokens,
                chat_history=chat_history,
                temperature=temperature
            )
            
            # Clear processing message
            if status_placeholder:
                status_placeholder.empty()
                
            return response
        finally:
            # Always release the lock if we acquired it
            if lock_acquired:
                llm_lock.release()
            
    except Exception as e:
        if status_placeholder:
            status_placeholder.empty()
        return f"Error generating response: {str(e)}"

def display_message(message: Dict):
    """Display a chat message."""
    role = message['role']
    content = message['content']
    
    if role == 'user':
        col1, col2 = st.columns([1, 4])
        with col2:
            st.markdown(f"""
            <div class="user-message">
                {content}
            </div>
            """, unsafe_allow_html=True)
    
    else:  # assistant
        col1, col2 = st.columns([4, 1])
        with col1:
            # Thinking process
            if message.get('has_thinking') and message.get('thinking'):
                with st.expander("💭 Thinking Process", expanded=False):
                    st.markdown(f"<div class='thinking-box'>{message['thinking']}</div>", unsafe_allow_html=True)
            
            # Main answer
            st.markdown(f"""
            <div class="assistant-message">
                {content}
            </div>
            """, unsafe_allow_html=True)
            
            # Sources
            if message.get('sources'):
                with st.expander(f"📚 Sources ({len(message['sources'])} chunks)", expanded=False):
                    for idx, result in enumerate(message['sources']):
                        source_name = result['metadata'].get('source', 'Unknown')
                        chunk_idx = result['metadata'].get('chunk_index', 'N/A')
                        similarity = result['similarity_score']
                        chunk_content = result['content']
                        
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

# Load shared resources (cached globally)
rag_system = get_shared_rag_system()
llm_model = get_shared_llm_model(st.session_state.model_name)

# Check if resources loaded successfully
if not rag_system:
    st.error("❌ Failed to load RAG system")
    st.stop()

if not llm_model:
    st.error("❌ Failed to load language model")
    st.stop()

# Header
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown("""
    <div class="main-header">
        <h1>💬 AUDI RAG Chat</h1>
        <p style="color: #8b949e; font-size: 1.1rem;">Ask anything about your documents</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Clear Chat Button
    if st.button("🗑️ Clear", use_container_width=True, key="clear_chat"):
        st.session_state.chat_history = []
        st.session_state.uploaded_images = []
        st.rerun()

# Chat Container
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        display_message(message)

# Input area
st.markdown("<br><br>", unsafe_allow_html=True)

input_container = st.container()

with input_container:
    # User input area and send button removed ctrll+enter for simplicity
    user_input = st.text_area(
        "Type your message...",
        value="",
        height=100,
        max_chars=1000,
        on_change=None,  # "" to reset input after sending but it rolls back to the top of the
        placeholder="Ask me anything about documents...",
        key="message_input",
        disabled=False,
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([5, 1])
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)

# Handle send button
if send_button and user_input.strip():
    # Add user message
    user_message = {
        'role': 'user',
        'content': user_input,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.chat_history.append(user_message)
    
    # Create a placeholder for status messages
    status_placeholder = st.empty()
    
    # Generate response
    try:
        # Show initial processing status
        status_placeholder.info("🔍 **Searching knowledge base...**")
        
        # Search for relevant chunks
        results = rag_system.search(
            query=user_input,
            top_k=st.session_state.top_k
        )
        
        # Filter by similarity threshold
        filtered_results = [r for r in results if r['similarity_score'] >= SIMILARITY_THRESHOLD]
        no_relevant_context = not filtered_results
        
        if no_relevant_context:
            context = ""
            status_placeholder.warning(f"⚠️ No relevant information found in knowledge base (threshold: {SIMILARITY_THRESHOLD}%). Answering with LLM only.")
            time.sleep(1)  # Brief pause so user can see the message
        else:
            context = rag_system.format_for_llm(filtered_results, max_chunks=None)
        
        # Generate response using shared LLM (with status updates)
        response = generate_llm_response(
            llm_model,
            user_input,
            context,
            st.session_state.chat_history[:-1] if len(st.session_state.chat_history) > 1 else [],
            st.session_state.max_tokens,
            st.session_state.temperature,
            status_placeholder=status_placeholder
        )
        
        # Parse response
        parsed_response = parse_thinking_response(response)
        
        # Add warning if no context
        final_answer = parsed_response['answer']
        #if no_relevant_context:
            #final_answer = f"⚠️ **No relevant information found in knowledge base** - Answer based on general knowledge:\n\n{final_answer}"
        
        # Add assistant message
        assistant_message = {
            'role': 'assistant',
            'content': final_answer,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'sources': filtered_results,
            'has_thinking': parsed_response['has_thinking'],
            'thinking': parsed_response.get('thinking'),
            'no_context': no_relevant_context
        }
        st.session_state.chat_history.append(assistant_message)
        
        # Clear status and rerun
        status_placeholder.empty()
        st.rerun()
        
    except Exception as e:
        status_placeholder.empty()
        st.error(f"Error generating response: {str(e)}")

# Status bar
st.markdown(f"""
<div style="text-align: center; padding: 20px; color: #8b949e; font-size: 0.9rem;">
    ✅ {st.session_state.model_name} ready | 💬 {len(st.session_state.chat_history)//2} exchanges | 
    👤 Session: {st.session_state.session_id[:]}
</div>
""", unsafe_allow_html=True)
