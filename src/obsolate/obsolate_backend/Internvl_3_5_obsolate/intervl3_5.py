import os
import base64
from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
from PIL import Image
import json

def image_to_base64_data_uri(file_path):
    """Convert image file to base64 data URI"""
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        # Determine image format from extension
        ext = Path(file_path).suffix.lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }.get(ext, 'image/jpeg')
        return f"data:{mime_type};base64,{base64_data}"

def process_images_in_folder(
    image_folder,
    model_path,
    mmproj_path,
    output_file="image_descriptions.json",
    prompt="Describe this image in detail."
):
    """
    Process all images in a folder and generate descriptions
    
    Args:
        image_folder: Path to folder containing images
        model_path: Path to the GGUF model file
        mmproj_path: Path to the mmproj file
        output_file: Path to save the results
        prompt: The prompt to use for each image
    """
    
    # Initialize the model with vision support
    print("Loading model...")

    try:
        # Try loading with explicit chat handler if available
        llm = Llama(
            model_path=model_path,
            chat_format="chatml",  
            n_ctx=40960,
            n_gpu_layers=-1,  # Use -1 to offload all layers to GPU
            verbose=False
        )
    except:
        # Fallback: Load without specific chat format
        llm = Llama(
            model_path=model_path,
            n_ctx=40960,
            n_gpu_layers=-1,
            verbose=False
        )
    
    print(f"Model loaded from {model_path}")
    
    # Get all image files from the folder
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    image_files = [
        f for f in Path(image_folder).iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    print(f"Found {len(image_files)} images to process")
    
    results = []
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        print(f"\nProcessing {idx}/{len(image_files)}: {image_path.name}")
        
        try:
            # Convert image to base64 data URI
            image = Image.open(image_path)
            messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What's in this image?"},
        ],
    },
]

            
            # Generate description
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=8096,
                temperature=0.7
            )
            
            description = response['choices'][0]['message']['content']
            
            # Store result
            result = {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "description": description,
                "prompt": prompt
            }
            results.append(result)
            
            print(f"Description: {description[:10000]}")
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")
            results.append({
                "image_path": str(image_path),
                "image_name": image_path.name,
                "error": str(e)
            })
    
    # Save results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Processing complete! Results saved to {output_file}")
    return results

# Example usage
if __name__ == "__main__":
    # Configure paths
    IMAGE_FOLDER = ".//data/images"  # Your folder with images
    MODEL_PATH = ".//models/llamacpp/internvl3_5-8b-q4_k_m.gguf"
    MMPROJ_PATH = ".//models/mmproj-F16.gguf"  # May be needed
    OUTPUT_FILE = "descriptions.json"
    
    # Optional: Custom prompt
    CUSTOM_PROMPT = "Provide a detailed description of this image, including objects, colors, and scene composition."
    
    # Process images
    results = process_images_in_folder(
        image_folder=IMAGE_FOLDER,
        model_path=MODEL_PATH,
        mmproj_path=MMPROJ_PATH,
        output_file=OUTPUT_FILE,
        prompt=CUSTOM_PROMPT
    )
    
    # Print summary
    print(f"\nProcessed {len(results)} images")
    successful = sum(1 for r in results if 'description' in r)
    print(f"Successful: {successful}")
    print(f"Errors: {len(results) - successful}")
