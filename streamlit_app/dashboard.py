import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import json
from src.qwen2_5.Qwen2_5 import llm as llm_qwen
from src.lfm2.LFM2 import llm as llm_lfm

st.set_page_config(page_title="Chatbot Dashboard", layout="wide")

st.title("Chatbot Dashboard")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Model Selection")
    # Choose a model from model_paths.json
    with open("./model_paths.json") as f:
        model_paths = json.load(f)
    model_option = st.selectbox(
        "Choose a model:",
        tuple(model_paths.keys())
    )
    
    st.header("Parameters")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 4096, 512, 100)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.1)
    
    user_input = st.text_area("User Input", height=150)
    system_message = st.text_area("System Message (Optional)", height=100)
    
    if st.button("Generate Response"):
        if model_option == "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL":
            llm = llm_qwen
        else:
            llm = llm_lfm
        
        # Build ChatML-style prompt
        prompt = ""
        if system_message:
            prompt += f"<|im_start|>system\n{system_message}<|im_end|>\n"
        
        prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        with st.spinner("Generating response..."):
            output = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False
            )
            response_text = output["choices"][0]["text"]
            st.session_state['response'] = response_text