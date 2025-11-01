"""
Modern Chat Interface for Hybrid RAG System - Multi-User Support
================================================================
Shared model resources with separate user sessions

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
SIMILARITY_THRESHOLD = 40.0

AVAILABLE_MODELS = ["InternVL3_5-2B-Q6_K"]

MODEL_CONFIG = {
    "InternVL3_5-2B-Q6_K": {
        "path": "models/llamacpp/InternVL3_5-2B-Q6_K.gguf",
        "key": "llm_internvl",
        "n_ctx": 40960,
        "supports_vision": True
    }
}

# =============================================================================
# SHARED RESOURCES (CACHED GLOBALLY ACROSS ALL USERS)
# =============================================================================

@st.cache_resource(show_spinner="Loading RAG system...")
def get_shared_rag_system():
    """Load RAG system once and share across all users."""
    db_path = project_root / DEFAULT_DB_PATH
    
    if not db_path.exists():
        st.error(f"Database not found: {db_path}")
        return None
    
    try:
        rag = HybridRAGQwen3_Module(
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
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
    """Load LLM once and share across all users."""
    if model_name not in MODEL_CONFIG:
        st.error(f"Unknown model: {model_name}")
        return None
    
    try:
        from llama_cpp import Llama
        
        config = MODEL_CONFIG[model_name]
        model_path = project_root / config["path"]
        
        if not model_path.exists():
            st.error(f"Model not found: {model_path}")
            return None
        
        llm = Llama(
            model_path=str(model_path),
            n_ctx=config["n_ctx"],
            n_gpu_layers=-1,
            temperature=0.7,
            verbose=False
        )
        return llm
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
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
        st.session_state.model_name = 'InternVL3_5-2B-Q6_K'
    
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = 4096
    
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 25

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
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
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
    max_tokens: int = 4096,
    temperature: float = 0.7
) -> str:
    """Generate LLM response using retrieved context and chat history."""
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
    
    # Build prompt
    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n"
    
    # Add chat history
    if chat_history and len(chat_history) > 0:
        recent_history = chat_history[-10:]
        for msg in recent_history:
            if msg['role'] == 'user':
                prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg['role'] == 'assistant':
                prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
    
    # Handle context
    if context and context.strip():
        prompt += f"""<|im_start|>user
Context from knowledge base:
{context}

Question: {query}<|im_end|>
<|im_start|>assistant
"""
    else:
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
            temperature=temperature,
            top_p=0.9,
            echo=False
        )
        return output["choices"][0]["text"]
    except Exception as e:
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
    user_input = st.text_area(
        "Type your message...",
        value="",
        height=100,
        placeholder="Ask me anything about your documents...",
        key="message_input",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([5, 1])
    with col2:
        send_button = st.button("📤 Send", type="primary", use_container_width=True)

# Handle send button
if send_button and user_input.strip():
    # Add user message
    user_message = {
        'role': 'user',
        'content': user_input,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.chat_history.append(user_message)
    
    # Generate response
    with st.spinner("🤔 Thinking..."):
        try:
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
                st.warning(f"⚠️ No relevant information found in knowledge base (threshold: {SIMILARITY_THRESHOLD}%). Answering with LLM only.")
            else:
                context = rag_system.format_for_llm(filtered_results, max_chunks=None)
            
            # Generate response using shared LLM
            response = generate_llm_response(
                llm_model,
                user_input,
                context,
                st.session_state.chat_history[:-1] if len(st.session_state.chat_history) > 1 else [],
                st.session_state.max_tokens,
                st.session_state.temperature
            )
            
            # Parse response
            parsed_response = parse_thinking_response(response)
            
            # Add warning if no context
            final_answer = parsed_response['answer']
            if no_relevant_context:
                final_answer = f"⚠️ **No relevant information found in knowledge base** - Answer based on general knowledge:\n\n{final_answer}"
            
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
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Status bar
st.markdown(f"""
<div style="text-align: center; padding: 20px; color: #8b949e; font-size: 0.9rem;">
    ✅ {st.session_state.model_name} ready | 💬 {len(st.session_state.chat_history)//2} exchanges | 
    👤 Session: {st.session_state.session_id[:8]}...
</div>
""", unsafe_allow_html=True)
