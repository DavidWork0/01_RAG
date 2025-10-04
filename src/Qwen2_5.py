from llama_cpp import Llama

# Load the model
llm = Llama(
    model_path="./models/llamacpp/Qwen2.5-VL-7B-Instruct-Q8_0.gguf",  # Path to your GGUF model
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

# Check token information
print(f"BOS token ID: {llm.token_bos()}")
print(f"EOS token ID: {llm.token_eos()}")
print(f"BOS token text: {llm._model.token_get_text(llm.token_bos())}")
print(f"EOS token text: {llm._model.token_get_text(llm.token_eos())}")

# Check if model adds BOS automatically
print(f"Model adds BOS: {llm._model.add_bos_token()}")


# Generate text
output = llm(
    prompt,
    max_tokens=4096,  # 512, 4096
    temperature=0.7,
    top_p=0.9,
    echo=False
)

print(output["choices"][0]["text"])
