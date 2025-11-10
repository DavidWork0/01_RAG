from llama_cpp import Llama


# Load the model
llm = Llama(
    model_path=".//models/llamacpp/LFM2-2.6B-Q8_0.gguf",  # Path to your GGUF model
    n_ctx=2048,  # Context window size
    n_gpu_layers=-1,  # Use 0 for CPU, >0 for GPU offloading
    temperature=0.7,  # Sampling temperature
    verbose=False  # Enable verbose output
)

# Format with proper chat template (ChatML-style)
prompt = """<|im_start|>user
What is speed of light?<|im_end|>
<|im_start|>assistant
"""
# Function to check token information
def check_token_info(llm):
    bos_id = llm.token_bos()
    eos_id = llm.token_eos()
    nl_id = llm.token_nl()
    bos_text = llm._model.token_get_text(bos_id)
    eos_text = llm._model.token_get_text(eos_id)
    nl_text = llm._model.token_get_text(nl_id)
    adds_bos = llm._model.add_bos_token()

    print(f"BOS token ID: {bos_id}, text: '{bos_text}'")
    print(f"EOS token ID: {eos_id}, text: '{eos_text}'")
    print(f"NL token ID: {nl_id}, text: '{nl_text}'")
    print(f"Model adds BOS automatically: {adds_bos}")

check_token_info(llm)

# Generate text
output = llm(
    prompt,
    max_tokens=4096,  # 512, 4096
    temperature=0.7,
    top_p=0.9,
    echo=False
)

print(output["choices"][0]["text"])
