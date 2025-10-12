#handling data operations

"""Module for handling data operations.
This module provides functions to load, preprocess, and save data.
It supports pdf, jpg, png file formats.

Author: Ats David

Functions:
- load_data(file_path): Load data from a specified file path.
- depending on file type, process_data(data, file_type): Preprocess data based on its type (e.g., pdf, jpg, png).
-- extract_text_pictures_tables_from_pdf(pdf_path): Extract content from a PDF file, add signature identifier in the position of the images for later reconstruction with image description (text). return text, images{id, position}, ?tables {dataframe, position}?
- picture description function for pdf images - describe_image(image_path): Generate a description for an image file.
--Text + text from pictures reconstruction.
- Create vector database from processed data.
- save_data(data, file_path): Save processed data to a specified file path
"""

import time
import fitz
import os
import re
import gc, torch
from intevl3_5.InternVL35_4B_reducedv2_single import initialize_model, inference_internvl3_5_4b_picture_path
from typing import List, Dict, Tuple

DUPLICATE_DETECTION_FOR_IMAGES = False  # Set to True to enable duplicate image detection Currently off as it's not reconstructing duplicates correctly


def extract_text_with_image_placeholders(pdf_path: str, images_folder: str, 
                                         output_text: str) -> Dict:
    """
    Extract text from PDF and insert image placeholders at correct positions.
    Save images separately.
    
    Returns metadata dict with image info.
    """
    os.makedirs(images_folder, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Store all content in reading order
    full_text_parts = []
    images_metadata = {}
    image_counter = 0
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Get images list for this page with xrefs
        page_images = page.get_images(full=True)
        
        # Create a mapping of image positions to xrefs
        image_xrefs = {}
        for img_tuple in page_images:
            xref = img_tuple[0]
            # Get image rectangles (positions) for this xref
            rects = page.get_image_rects(xref)
            for rect in rects:
                # Use bbox as key (rounded to avoid floating point issues)
                bbox_key = tuple(round(x, 2) for x in (rect.x0, rect.y0, rect.x1, rect.y1))
                image_xrefs[bbox_key] = xref
        
        # Get all blocks (text and images) in reading order
        blocks = page.get_text("dict", sort=True)["blocks"]
        
        for block in blocks:
            if block["type"] == 0:  # Text block
                # Extract text from block
                block_text = ""
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    block_text += line_text + "\n"
                
                full_text_parts.append(block_text.strip())
            
            elif block["type"] == 1:  # Image block
                try:
                    # Image blocks contain the image data directly, not xref
                    bbox = block["bbox"]
                    bbox_key = tuple(round(x, 2) for x in bbox)
                    
                    # Try to get xref from our mapping
                    xref = image_xrefs.get(bbox_key)
                    
                    # Generate unique image ID
                    image_id = f"{pdf_basename}_page{page_num:04d}_img{image_counter:03d}"
                    
                    # Extract image - image blocks have the data directly
                    if "image" in block and block["image"]:
                        # Image data is directly in the block
                        image_bytes = block["image"]
                        ext = block.get("ext", "png")
                    elif xref:
                        # Fallback: extract using xref
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        ext = base_image["ext"]
                    else:
                        print(f"Warning: Could not extract image on page {page_num}")
                        continue
                    
                    # Save image with unique name
                    image_filename = f"{image_id}.{ext}"
                    image_path = os.path.join(images_folder, image_filename)
                    
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    # Insert placeholder in text
                    placeholder = f"[IMAGE:{image_id}]"
                    full_text_parts.append(placeholder)
                    
                    # Store metadata
                    images_metadata[image_id] = {
                        'filename': image_filename,
                        'filepath': image_path,
                        'page': page_num,
                        'bbox': bbox,
                        'placeholder': placeholder
                    }
                    
                    print(f"Extracted: {image_filename} -> {placeholder}")
                    image_counter += 1
                    
                except Exception as e:
                    print(f"Error extracting image on page {page_num}: {e}")
    
    # Combine all text parts
    full_text = "\n\n".join(full_text_parts)
    
    # Save text with placeholders
    with open(output_text, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    doc.close()
    
    print(f"\n✓ Saved text with placeholders: {output_text}")
    print(f"✓ Extracted {len(images_metadata)} images to: {images_folder}")
    
    return images_metadata

def create_descriptions_file_template(images_folder: str, output_file: str):
    """
    Create a template file for image descriptions.
    User will fill this with their vision-language model output.
    
    Format:
    image_id_1
    Description of image 1 goes here...
    
    image_id_2
    Description of image 2 goes here...
    """
    image_files = sorted([f for f in os.listdir(images_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for img_file in image_files:
            # Remove extension to get image_id
            image_id = os.path.splitext(img_file)[0]
            
            f.write(f"{image_id}\n")
            f.write(f"[TODO: Add description for {img_file}]\n\n")
    
    print(f"✓ Created descriptions template: {output_file}")
    print(f"  Process images and fill in descriptions!")

def process_images_with_vl_model(images_folder: str, output_descriptions: str, model, device, tokenizer):
    """
    Process all images with your vision-language model and save descriptions.
    Replace this with your actual VL model (InternVL, LLaVA, etc.)
    
    Output format:
    image_id
    Description text...
    
    next_image_id
    Description text...
    """

    image_files = sorted([f for f in os.listdir(images_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        print(f"No images found in {images_folder} to process.")
        return
    

    print(f"Processing {len(image_files)} images with vision-language model...")
    with open(output_descriptions, 'w', encoding='utf-8') as f:
        for img_file in image_files:
            image_id = os.path.splitext(img_file)[0]
            image_path = os.path.join(images_folder, img_file)
            
            # Call your inference function
            response = inference_internvl3_5_4b_picture_path(model, device, tokenizer, image_path)

            # Save to file
            f.write(f"[IMAGE:{image_id}]\n")
            f.write(f"{response}\n\n") #If something more is needed then response end can be signed as well
            
            print(f"Processed: {img_file}\n")
            
            # Clear GPU memory
            gc.collect()
            torch.cuda.empty_cache()

    
    print(f"\n✓ Saved descriptions: {output_descriptions}")

def parse_descriptions_file(descriptions_file: str) -> Dict[str, str]:
    """
    Parse the descriptions file.
    
    Expected format:
    image_id
    Description text (can be multiple lines)
    
    next_image_id
    Next description...
    """
    descriptions = {}
    
    with open(descriptions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_id = None
    current_description = []
    
    for line in lines:
        line_stripped = line.strip()

        # Next placeholder indicates the end of current description
        if line_stripped.startswith("[IMAGE:") and line_stripped != current_id:
            #Save descriptions with current_id if exists and with content
            if current_id and current_description:
                descriptions[current_id] = "\n".join(current_description).strip()
            current_id = line_stripped
            current_description = []
        elif not line_stripped.startswith("[IMAGE:"):
            #Removing blank rows (\n) from description but keeping intentional blank rows
            if line_stripped is not None and not line_stripped == "":
                current_description.append(line_stripped)
            else:
                current_description.append('')

    # Save last description
    if current_id and current_description:
        descriptions[current_id] = "\n".join(current_description).strip()
    return descriptions

def merge_text_with_descriptions(text_with_placeholders: str, 
                                 descriptions_file: str,
                                 output_merged: str):
    """
    Merge the text (with placeholders) and descriptions file.
    Replace all [IMAGE:id] placeholders with actual descriptions.
    """
    # Load text with placeholders
    with open(text_with_placeholders, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Parse descriptions
    descriptions = parse_descriptions_file(descriptions_file)
    
    print(f"Loaded {len(descriptions)} image descriptions")
    
    # Replace all placeholders with descriptions
    replaced_count = 0
    for image_id, description in descriptions.items():
        placeholder = image_id
        if placeholder in text:
            #Replace text till the next placeholder not just the first row
            text = text.replace(placeholder, description)
            replaced_count += 1
            print(f"Replaced: {placeholder}")
    
    # Save merged text
    with open(output_merged, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"\n✓ Merged text saved: {output_merged}")
    print(f"✓ Replaced {replaced_count} image placeholders")
    
    # Check for unreplaced placeholders
    remaining = re.findall(r'\[IMAGE:[^\]]+\]', text)
    if remaining:
        print(f"⚠ Warning: {len(remaining)} placeholders not replaced:")
        for p in remaining[:5]:  # Show first 5
            print(f"  {p}")

def detect_duplicate_images(images_folder: str):
    """
    Detect duplicate images in the folder using file size and hash.
    """
    import hashlib
    from collections import defaultdict
    
    def hash_file(filepath):
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    
    image_files = [f for f in os.listdir(images_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    hash_dict = defaultdict(list)
    
    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        file_hash = hash_file(img_path)
        hash_dict[file_hash].append(img_file)
    
    duplicates = {h: files for h, files in hash_dict.items() if len(files) > 1}
    
    if duplicates:
        print(f"Detected {len(duplicates)} sets of duplicate images:")
        for files in duplicates.values():
            print(f"  Duplicates: {', '.join(files)}")
        return duplicates
    else:
        print("No duplicate images detected.")


def rename_pdf_file_names(input_dir: str):
    """
    Rename PDF files in the input directory to avoid later processing problems with special characters. Replace "." and " " and "-" with "_" and add sequential numbering.
    """
    pdf_files = [f for f in os.listdir(input_dir) 
                 if f.lower().endswith('.pdf')]
    
    for i, pdf_file in enumerate(pdf_files, 1):
        new_name = re.sub(r'[.\s-]+', '_', os.path.splitext(pdf_file)[0])
        new_name = f"{new_name}.pdf"
        
        src_path = os.path.join(input_dir, pdf_file)
        dst_path = os.path.join(input_dir, new_name)
        
        if src_path != dst_path:
            os.rename(src_path, dst_path)
            print(f"Renamed: {pdf_file} -> {new_name}")
    
    print("✓ Renaming complete.")

def batch_process_pdfs_complete_workflow(input_dir: str, output_dir: str):
    """
    Complete workflow for all PDFs in a directory.
    """

    # Setup directories
    images_dir = os.path.join(output_dir, "extracted_images")
    texts_dir = os.path.join(output_dir, "texts_with_placeholders")
    descriptions_dir = os.path.join(output_dir, "image_descriptions")
    merged_dir = os.path.join(output_dir, "final_merged")

    #init model
    #Only init once  and check if initialized then skip
    model, device, tokenizer = initialize_model()

    rename_pdf_file_names(input_dir)
    
    for d in [images_dir, texts_dir, descriptions_dir, merged_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Get all PDFs
    pdf_files = [f for f in os.listdir(input_dir) 
                 if f.lower().endswith('.pdf')]
    
    print(f"Found {len(pdf_files)} PDF files\n")
    print("="*60)
    
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(input_dir, pdf_file)
        pdf_name = os.path.splitext(pdf_file)[0]
        
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file}")
        print("-"*60)


        # Create separate folder for each PDF's images
        pdf_images_dir = os.path.join(images_dir, pdf_name)


        # Step 1: Extract text with placeholders
        print("\nStep 1: Extracting text with image placeholders...")
        text_output = os.path.join(texts_dir, f"{pdf_name}.txt")
        
        try:
            metadata = extract_text_with_image_placeholders(
                pdf_path, pdf_images_dir, text_output
            )
            if DUPLICATE_DETECTION_FOR_IMAGES:
                # step 1.1: Detect duplicate images
                duplicates = detect_duplicate_images(pdf_images_dir)
                # TO MAKE THIS RIGHT I NEED TO PREPROCESS THE FILE NAMES TO BE ABLE TO GROUP THEM -> DONE
                # step 1.2 Move duplicates to a specific folder. Each duplicate type set in its own subfolder 
                if duplicates:
                    duplicates_dir = os.path.join(pdf_images_dir, "duplicates")
                    os.makedirs(duplicates_dir, exist_ok=True)
                    for dup_files in duplicates.values():
                        dup_set_dir = os.path.join(duplicates_dir, dup_files[0].split('.')[0])
                        os.makedirs(dup_set_dir, exist_ok=True)
                        for f in dup_files:
                            src_path = os.path.join(pdf_images_dir, f)
                            dst_path = os.path.join(dup_set_dir, f)
                            os.rename(src_path, dst_path)
                            # Remove duplicate from main folder
                    print(f"Moved duplicate images to: {duplicates_dir}")
            else:
                print("Duplicate detection skipped.")
            
            # Step 2: Process images with VL model
            print("\nStep 2: Processing images with vision model...")
            #making the files for the descriptions
            descriptions_output = os.path.join(descriptions_dir, f"{pdf_name}_descriptions.txt")

            process_images_with_vl_model(pdf_images_dir, descriptions_output, model, device, tokenizer)

            # ---------------------------------0----------------------------------

            # Step 3: Merge
            print("\nStep 3: Merging text and descriptions...")
          
            merged_output = os.path.join(merged_dir, f"{pdf_name}_final.txt")
            merge_text_with_descriptions(text_output, descriptions_output, merged_output)
            
            print(f"\n✓ Completed: {pdf_file}")
            
        except Exception as e:
            print(f"\n✗ Failed: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"All processing complete!")
    print(f"Final merged files: {merged_dir}")


if __name__ == "__main__":

    import time

    #Clear output directory before running
    output_dir = "./data/output"
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"Cleared existing output directory: {output_dir}")
    # Example usage of the complete workflow
    # Adjust input_dir and output_dir as needed
    input_dir = "./data/pdfs"
    output_dir = "./data/output"
    

    #measure time
    try:
        start_time = time.time()
    except:
        i=1

    batch_process_pdfs_complete_workflow(input_dir, output_dir)

    try:
        end_time = time.time()
    except:
        i=1
    
    print(f"Total processing time: {end_time - start_time:.2f} seconds")


"""
# Example: Process a single PDF
pdf_path = "document.pdf"

# Step 1: Extract text with placeholders and images
metadata = extract_text_with_image_placeholders(
    pdf_path=pdf_path,
    images_folder="extracted_images",
    output_text="text_with_placeholders.txt"
)

# Step 2: Process images with YOUR vision model
# Replace process_images_with_vl_model with your actual model
process_images_with_vl_model(
    images_folder="extracted_images",
    output_descriptions="image_descriptions.txt"
)

# Step 3: Merge the two text files
merge_text_with_descriptions(
    text_with_placeholders="text_with_placeholders.txt",
    descriptions_file="image_descriptions.txt",
    output_merged="final_document.txt"
)
"""