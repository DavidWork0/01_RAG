# RAG system

**THE SYSTEM IS NOT A STANDALONE. THE MODULES ARE CONFIGURED TO FIND LOCAL OFFLINE MODELS. IF THEY ARE NOT PRESENT THE SYSTEM NOT ABLE TO WORK (MOST PROPABLY)** see placeholder.txt in models/huggingface folder

## System usage:

Currently the POC is done and working. **The performace is below the expected**. 

### Working principle:
- Add pdf files to .\data\pdfs
- run data_pipeline_pdf.py
- run chunk_qwen4_0_6B.py
- run streamlit_modern_muliuser.py

## Current capabilities
    **data_pipeline_pdf.py:** 
      *INPUTS: multiple pdf file
        -Extract text from pdf
        -Extract image from pdf
        -Transform image to text and place back where the image was (Creating text from images)
        -Connects proper positions and create final .txt file with text only.
      *OUTPUTS: multiple folder with different content: extracted images, image_descriptions,text_with_placeholders -> final_merged/....final.txts

    **chunk_qwen4_0_6B.py:** 
      *INPUTS: text_with_placeholders -> final_merged/....final.txts
        -Chunks text (multiple parametrization possible)
        -Created vectorDB (ChromaDB) -> .\data\outputs\chroma_db_...PARAMETERS
      *OUTPUTS: (ChromaDB) -> .\data\outputs\chroma_db_...PARAMETERS

    **streamlit_modern_multiuser.py:** 
      *INPUTS: Chroma_DB and hybrid_RAG system
        -Doing it's done in a fancy way (of course this line and decription is not final :) )
        -
        -
        -
      *OUTPUTS: localhost with inference window

----FOR LATER MODULES---
    **.py:** 
      *INPUTS: 
        -Ex
        -
        -
        -
      *OUTPUTS: 