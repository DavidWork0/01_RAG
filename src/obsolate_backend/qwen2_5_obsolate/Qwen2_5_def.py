from llama_cpp import Llama

def load_model(model_path=".//models/llamacpp/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf", n_ctx=2048, n_gpu_layers=-1, temperature=0.7, test_mode=False):

    if test_mode==True:
        verbose=True  # Enable verbose output
    else:
        verbose=False  # Enable verbose output

    # Load the model
    llm = Llama(
        model_path=model_path,  # Path to your GGUF model alternative
        n_ctx=n_ctx,  # Context window size
        n_gpu_layers=n_gpu_layers,  # Use 0 for CPU, <0 for GPU offloading
        temperature=temperature,
        verbose=verbose  # Enable verbose output
    )

    if test_mode==True:
        check_token_info(llm)
    else:
        pass
    
    return llm

def run_model(llm):
    # Format with proper chat template (ChatML-style)
    system_message = "You are a helpful assistant."
    user_message = "What is speed of light?"
    # Build ChatML-style prompt
    prompt = ""
    if system_message:
        prompt += f"<|im_start|>system\n{system_message}<|im_end|>\n"
            
        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
    # Function to check token information

    # Generate text
    output = llm(
        prompt,
        max_tokens=4096,  # 512, 4096
        temperature=0.7,
        top_p=0.9,
        echo=False
    )

    return output["choices"][0]["text"]


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

if __name__ == "__main__":
    llm = load_model(test_mode=True)
    response = run_model(llm)
    print(response)