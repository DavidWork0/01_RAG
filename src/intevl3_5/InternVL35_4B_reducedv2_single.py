from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import gc
import os


LOCAL_MODEL_PATH = ".//models/huggingface/InternVL3_5-4B"

def initialize_model():
    print ("Init model func")
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=torch.bfloat16,  # Changed from 'dtype' to 'torch_dtype'
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH, 
        trust_remote_code=True
    )

    return model, device, tokenizer


def inference_internvl3_5_4b_picture_path(model, device, tokenizer, picture_path=None):

    # Image preprocessing functions
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)


    def build_transform(input_size):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform


    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n + 1) for j in range(1, n + 1) 
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        best_ratio = min(target_ratios, 
                        key=lambda ratio: abs(aspect_ratio - ratio[0]/ratio[1]))
        
        target_width = image_size * best_ratio[0]
        target_height = image_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]
        
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        
        return processed_images


    def load_image(image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, 
                                use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values


    # Load and process image with HIGHER RESOLUTION
    # Increased max_num from 12 to 36 for more detail (can go up to 128)
    question = "Describe the image in detail."


    print(f"Processing image : {picture_path}")
    pixel_values = load_image(
        picture_path,
        input_size=448,  # Keep at 448 (standard tile size)
        max_num=12       # Increased from 12 to 36 for more tiles --> Increased from 12 to 36 for more tiles AFTER TESTING WITH 36 WITH A LOT OF PICTURES WAS OK BUT WITH SOME IT STARTED TO CONSUME 60+ GB RAM+VRAM !!DIPLOMA 
    ).to(torch.bfloat16).to(device)

    ############### INFERENCE ###############
    # Generation config
    generation_config = dict(max_new_tokens=8192, do_sample=False)

    # Use chat method instead of generate
    response = model.chat(tokenizer, pixel_values, question, generation_config)

    # Return the processed image name and response
    image_name = os.path.basename(picture_path)
    print(f"Completed processing for image: {image_name}")
    

    return response
        


if __name__ == "__main__":
    import time
    start_time = time.time()
    picture_path = ".//data/images/barchart.png"  # Path to a single image
    model, device, tokenizer = initialize_model()
    description = inference_internvl3_5_4b_picture_path(model, device, tokenizer, picture_path)
    print("\n\nFinal Description:")
    print(description)
    end_time = time.time()
    print(f"\n-----------------------------------------------------\n\n Total image processing time: {end_time - start_time:.2f} seconds")

