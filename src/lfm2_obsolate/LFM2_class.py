from llama_cpp import Llama
from typing import Optional, Dict, Any
import json


class LlamaInference:
    """
    A wrapper class for llama-cpp-python inference.
    Provides easy-to-use methods for model loading and text generation.
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        temperature: float = 0.7,
        verbose: bool = False
    ):
        """
        Initialize the Llama model.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 for all, 0 for CPU only)
            temperature: Sampling temperature
            verbose: Enable verbose output
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.temperature = temperature
        
        # Load the model
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            temperature=temperature,
            verbose=verbose
        )
    
    def check_token_info(self) -> Dict[str, Any]:
        """
        Get token information from the model.
        
        Returns:
            Dictionary containing BOS, EOS, and NL token information
        """
        bos_id = self.llm.token_bos()
        eos_id = self.llm.token_eos()
        nl_id = self.llm.token_nl()
        
        bos_text = self.llm._model.token_get_text(bos_id)
        eos_text = self.llm._model.token_get_text(eos_id)
        nl_text = self.llm._model.token_get_text(nl_id)
        
        adds_bos = self.llm._model.add_bos_token()
        
        token_info = {
            'bos': {'id': bos_id, 'text': bos_text},
            'eos': {'id': eos_id, 'text': eos_text},
            'nl': {'id': nl_id, 'text': nl_text},
            'adds_bos_automatically': adds_bos
        }
        
        return token_info
    
    def print_token_info(self):
        """Print token information in a readable format."""
        info = self.check_token_info()
        print(f"BOS token ID: {info['bos']['id']}, text: '{info['bos']['text']}'")
        print(f"EOS token ID: {info['eos']['id']}, text: '{info['eos']['text']}'")
        print(f"NL token ID: {info['nl']['id']}, text: '{info['nl']['text']}'")
        print(f"Model adds BOS automatically: {info['adds_bos_automatically']}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        echo: bool = False
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (uses instance default if None)
            top_p: Top-p sampling parameter
            echo: Whether to include the prompt in the output
            
        Returns:
            Generated text string
        """
        if temperature is None:
            temperature = self.temperature
        
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=echo
        )
        
        return output["choices"][0]["text"]
    
    def generate_with_chat_template(
        self,
        user_message: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate text using ChatML-style template.
        
        Args:
            user_message: User's input message
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            system_message: Optional system message
            
        Returns:
            Generated response text
        """
        # Build ChatML-style prompt
        prompt = ""
        if system_message:
            prompt += f"<|im_start|>system\n{system_message}<|im_end|>\n"
        
        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False
        )
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Allow the instance to be called directly like a function.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional arguments passed to generate()
            
        Returns:
            Generated text string
        """
        return self.generate(prompt, **kwargs)


# Example usage
if __name__ == "__main__":
    # Initialize the inference engine
    inference = LlamaInference(
        model_path="./models/llamacpp/LFM2-2.6B-Q8_0.gguf",
        n_ctx=2048,
        n_gpu_layers=-1,
        temperature=0.7,
        verbose=False
    )
    
    # Check token information
    inference.print_token_info()
    
    # Generate text with chat template
    response = inference.generate_with_chat_template(
        user_message="What is speed of light?",
        max_tokens=4096
    )
    
    print(response)
