# Hybrid RAG System with Multi-Modal Support

A Retrieval-Augmented Generation (RAG) system that processes PDF documents with text and images, creates vector embeddings, and provides an interactive chat interface for querying the content. The system uses local offline models for complete privacy and control.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Check%20license%20folder-green.svg)](license/)

---

## ⚠️ Important: Not a Standalone System

**This repository does NOT include the required AI models.** The system is configured to use local offline models that must be downloaded and placed in the correct directories before use.

### Required Models (Not Included)

Due to their large size (10-30+ GB), models are excluded via `.gitignore` and must be downloaded separately:

**Location:** `models/huggingface/` and `models/llamacpp/`

**Required Models:**
1. **Embedding Model** (HuggingFace format):
   - `Qwen/Qwen3-Embedding-0.6B` → Place in `models/huggingface/models--Qwen--Qwen3-Embedding-0.6B/`
   - Download from: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

2. **Vision Model** (HuggingFace format):
   - `OpenGVLab/InternVL3_5-4B` → Place in `models/huggingface/InternVL3_5-4B/`
   - Download from: https://huggingface.co/OpenGVLab/InternVL3_5-4B

3. **LLM Inference Models** (GGUF format for llama.cpp):
   - Place `.gguf` files in `models/llamacpp/`
   - Recommended: `InternVL3_5-2B-Q6_K.gguf`, `Qwen3-4B-Instruct-2507-Q8_0.gguf`, `Qwen3-8B-Q4_K_M.gguf`
   - Download from: https://huggingface.co/bartowski (community GGUF conversions)

**Without these models, the system will not function.**

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.11 or higher
- **CUDA**: 11.8+ (for GPU acceleration)
- **VRAM**: Minimum 8GB recommended (depends on model size)
- **Storage**: 50GB+ for models and vector databases
- **OS**: Windows (PowerShell), Linux, or macOS

### Installation

#### Option 1: Local Installation (Recommended for Development)

1. **Clone the repository:**
   ```powershell
   git clone <repository-url>
   cd 01_RAG
   ```

2. **Create and activate virtual environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements_project.txt
   ```

4. **Download and place models** (see "Required Models" section above)

5. **Configure environment variables** (optional):
   ```powershell
   $env:RAG_MODEL_CACHE_DIR = "C:\path\to\your\models\huggingface"
   $env:CUDA_VISIBLE_DEVICES = "0"
   ```

#### Option 2: Docker Installation (Production/Isolated Environment)

1. **Prerequisites:**
   - Docker Desktop with WSL2 backend (Windows) or Docker Engine (Linux)
   - NVIDIA Container Toolkit for GPU support
   - Models downloaded and placed in `models/` directory

2. **Verify Docker GPU support:**
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
   ```

3. **Build Docker image:**
   ```powershell
   # Build from Dockerfile_multi (includes CUDA 12.8, Python 3.11.9, llama-cpp-python)
   docker-compose -f docker-compose.yml build
   
   # Or build manually
   docker build -f Dockerfile_multi -t linux_with_cuda_complete:latest .
   ```
   
   **Build time:** ~30-80 minutes (compiles Python, PyTorch, llama-cpp-python with CUDA support)

4. **Run container:**
   ```powershell
   # Using docker-compose (recommended)
   docker-compose -f docker-compose.yml up -d
   
   ```

5. **Access container:**
   ```powershell
   # Interactive shell
   docker exec -it linux_with_cuda-complete-container bash
   
   # Run dashboard inside container
   streamlit run /app/01_RAG/src/dashboard.py --server.port=8501 --server.address=0.0.0.0
   ```

6. **View logs:**
   ```powershell
   docker logs -f linux_with_cuda-complete-container
   ```

7. **Stop and remove container:**
   ```powershell
   docker-compose -f docker-compose.yml down
   # or
   docker stop linux_with_cuda-complete-container
   docker rm linux_with_cuda-complete-container
   ```

**Docker Notes:**
- Models, data, and code are **copied into the image** during build (no volume mounts by default)
- To update code/models, rebuild the image or use `docker cp` to copy files into running container:
  ```powershell
  # Copy updated code into container
  docker cp src/dashboard.py linux_with_cuda-complete-container:/app/01_RAG/src/dashboard.py
  
  # Copy models into container
  docker cp models/llamacpp/new_model.gguf linux_with_cuda-complete-container:/app/01_RAG/models/llamacpp/
  
  # Copy entire directory
  docker cp data/pdfs linux_with_cuda-complete-container:/app/01_RAG/data/
  ```
- For development with live code changes, add volume mounts to `docker-compose.yml`
- Container runs with `tail -f /dev/null` by default (stays alive for manual commands)
- GPU architecture is set to `CUDA_DOCKER_ARCH=89` (RTX 4060 Ti) - adjust in `Dockerfile_multi` for other GPUs

---

## 📖 System Architecture

### Pipeline Overview

```
PDF Documents → Text/Image Extraction → Image Captioning → Text Chunking → Vector Embeddings → RAG Query System
```

### Detailed Workflow Diagram (Mermaid)

```mermaid
flowchart TB
   %% ==============================
   %% Hybrid RAG: end-to-end workflow
   %% ==============================

   %% ---------- Actors ----------
   user([User])
   dev([Developer])
   jenkins([Jenkins Pipeline])

   %% ---------- Storage / Artifacts ----------
   subgraph artifacts["Artifacts / Folders"]
      pdfs[(data/pdfs (PDF files))]
      extracted_images[(data/output/extracted_images)]
      image_desc[(data/output/image_descriptions)]
      merged_text[(data/output/final_merged)]
      merged_clean[(data/output/final_merged/cleaned)]
      chroma_db[(data/output/chroma_db_*)]
      test_logs[(tests/logs)]
      session_logs[(tests/logs/sessions)]
   end

   %% ---------- Models ----------
   subgraph models["Local Offline Models (not in git)"]
      hf_embed[(models/huggingface/...\nQwen/Qwen3-Embedding-0.6B)]
      hf_vlm[(models/huggingface/...\nOpenGVLab/InternVL3_5-4B)]
      gguf_llm[(models/llamacpp (GGUF models)\nllama.cpp backend)]
   end

   %% ---------- Config ----------
   subgraph config["Configuration"]
      rag_cfg[[src/rag_config.py\nchunking + retrieval params]]
      model_cfg[[src/model_config.py\nLLM model list + prompt templates]]
      env_vars[[Env vars\nHF_HOME / TRANSFORMERS_CACHE\nRAG_MODEL_CACHE_DIR\nCUDA_VISIBLE_DEVICES]]
   end

   %% ---------- 1) Ingestion / Extraction ----------
   subgraph ingest["1) PDF Ingestion & Multi-Modal Extraction"]
      dp_pdf[[src/data_pipeline_pdf.py]]
      fitz[(PyMuPDF / fitz)]
      placeholder_text[[Text with IMAGE placeholders]]
      internvl[[InternVL3.5 inference\n(intevl3_5 module)]]
      merge_step[[Merge text + captions\n(placeholder replacement)]]
   end

   %% ---------- 2) Cleaning ----------
   subgraph clean["2) Text Cleaning"]
      pre_chunk[[src/pre_chunking.py]]
      cleaner[[Remove PDF artifacts\n(page nums / dot lines / spacing)]]
   end

   %% ---------- 3) Chunking + Embeddings + DB ----------
   subgraph index["3) Chunking, Embeddings, Vector DB Build"]
      chunker[[src/chunk_qwen3_0_6B.py]]
      strategy{CHUNK_STRATEGY}
      fixed[[fixed_size\nchunk_size + overlap]]
      bysent[[by_sentence\nNLTK sentence tokenize]]
      qwen_embed[[Qwen3EmbeddingFunction\nHF AutoTokenizer/AutoModel]]
      chroma[[ChromaDB PersistentClient]]
      collection[(Collection: docs_*\n(metadata: source, chunk_index))]
   end

   %% ---------- 4) Serving / RAG Query ----------
   subgraph serve["4) Interactive RAG Query (Streamlit)"]
      ui[[src/dashboard.py\nStreamlit multi-user UI]]
      shared_lock[[Shared LLM lock\n(st.cache_resource)]]
      rag_core[[src/hybrid_rag_module_qwen3.py\nHybridRAGQwen3_Module]]
      q_embed[[Embed user query\n(Qwen3-Embedding)]]
      retrieve[[Chroma similarity search\ninitial_k = top_k * multiplier (cap)]]
      keyword[[Keyword extraction + scoring\n(stop words, min length)]]
      rerank[[Hybrid rerank\nsemantic_weight + keyword_weight]]
      threshold{Min similarity threshold}
      prompt[[Prompt assembly\n(PROMPT_TEMPLATE(_WITH_HISTORY))]]
      llm[[llama-cpp-python\nGGUF LLM inference]]
      answer[[Final answer + citations-by-text\n(chat history stored per session)]]
   end

   %% ---------- 5) Testing / Evaluation / Experiment Tracking ----------
   subgraph eval["5) Testing, Evaluation, Neptune Upload"]
      pytest[[pytest tests]]
      test_infer[[tests/test_inference.py\nrun questions + log sessions]]
      infer_logger[[src/inference_logger.py\nwrite JSONL/logs]]
      ans_eval[[src/answer_evaluator.py\nROUGE/BLEU/semantic/TF-IDF]]
      nept_up[[src/neptune_uploader.py\nupload session logs]]
      neptune[(Neptune.ai\noptional cloud tracking)]
   end

   %% ---------- 6) Docker / CI ----------
   subgraph cicd["6) Docker + CI/CD"]
      compose[[docker-compose.yml\nlinux_with_cuda_complete]]
      image[[Dockerfile_multi\nCUDA + Python + deps]]
      jfile[[Jenkinsfile_docker_pipeline\n(run pipeline steps 1-3)]]
      jfiles[[Other Jenkinsfiles\n(inference tests / top-k sweeps / neptune)]]
   end

   %% ==============================
   %% Edges: Data pipeline
   %% ==============================

   dev -->|drops PDFs| pdfs
   pdfs --> dp_pdf
   dp_pdf -->|extract text blocks| fitz
   dp_pdf --> extracted_images
   dp_pdf -->|write| placeholder_text
   extracted_images --> internvl
   hf_vlm --> internvl
   internvl -->|write| image_desc
   placeholder_text --> merge_step
   image_desc --> merge_step
   merge_step -->|write| merged_text

   %% Cleaning
   merged_text --> pre_chunk
   pre_chunk --> cleaner
   cleaner -->|write| merged_clean

   %% Chunking/index build
   merged_clean --> chunker
   rag_cfg --> chunker
   env_vars --> chunker
   hf_embed --> qwen_embed
   hf_embed --> qwen_embed
   chunker --> strategy
   strategy -->|fixed_size| fixed
   strategy -->|by_sentence| bysent
   fixed --> qwen_embed
   bysent --> qwen_embed
   qwen_embed --> chroma
   chroma --> collection
   collection --> chroma_db

   %% ==============================
   %% Edges: RAG serving
   %% ==============================

   user --> ui
   ui -->|load configs| model_cfg
   ui -->|load retrieval params| rag_cfg
   ui -->|load shared RAG system| rag_core
   ui -->|load shared LLM| llm
   gguf_llm --> llm
   ui --> shared_lock

   rag_core -->|connect| chroma_db
   ui -->|question + history| q_embed
   q_embed --> retrieve
   retrieve --> keyword
   keyword --> rerank
   rerank --> threshold
   threshold -->|pass| prompt
   threshold -->|filter out low scores| retrieve
   prompt -->|sequential access| shared_lock
   shared_lock --> llm
   llm --> answer
   answer --> ui

   %% ==============================
   %% Edges: tests/eval
   %% ==============================

   dev --> pytest
   pytest --> test_infer
   test_infer --> infer_logger
   infer_logger --> test_logs
   infer_logger --> session_logs
   session_logs --> ans_eval
   session_logs --> nept_up
   nept_up --> neptune

   %% ==============================
   %% Edges: CI/CD
   %% ==============================

   image --> compose
   jenkins --> jfile
   jenkins --> jfiles
   jfile -->|python3 src/data_pipeline_pdf.py\npython3 src/pre_chunking.py\npython3 src/chunk_qwen3_0_6B.py| chunker
   jfiles -->|run tests + upload| eval

```

### Core Modules

#### 1. **`data_pipeline_pdf.py`** - PDF Processing
Extracts and processes content from PDF documents.

**Inputs:**
- PDF files in `data/pdfs/`

**Process:**
- Extracts text content from PDFs
- Extracts embedded images
- Generates text descriptions for images using vision models (InternVL3.5)
- Merges text and image descriptions maintaining document structure
- Cleans and formats output

**Outputs:**
- `data/output/extracted_images/` - Extracted images
- `data/output/image_descriptions/` - VLM generated image captions
- `data/output/final_merged/` - Combined text files with image descriptions

**Usage:**
```powershell
python src/data_pipeline_pdf.py

# Or run the complete pipeline (Steps 1-3)
src/runner_pipeline.bat   # Windows
src/runner_pipeline.sh    # Linux/macOS
```

---

#### 2. **`chunk_qwen3_0_6B.py`** - Text Chunking & Vector Database Creation
Creates vector embeddings and builds ChromaDB database.

**Inputs:**
- Merged text files from `data/output/final_merged/cleaned/`

**Process:**
- **Step 2 (pre_chunking.py):** Text preprocessing and cleaning
- **Step 3 (chunk_qwen3_0_6B.py):** Chunks text using configurable parameters (chunk_strategy, chunk_size, overlap, embedding_dimension)
- Generates embeddings using Qwen3-Embedding-0.6B
- Stores vectors in ChromaDB with metadata
- Supports multiple chunking strategies (fixed_size, by_sentence)

**Outputs:**
- ChromaDB database in `data/output/chroma_db_{chunk_strategy}_Qwen_Qwen3-Embedding-0.6B_{params}/`
- Database naming convention: `chroma_db_{chunk_strategy}_Qwen_Qwen3-Embedding-0.6B_{embed_dim}_{chunk_size}_{overlap}_{cleaned|uncleaned}`

**Usage:**
```powershell
# Run preprocessing + chunking separately
python src/pre_chunking.py
python src/chunk_qwen3_0_6B.py

# Or run the complete pipeline (Steps 1-3)
src/runner_pipeline.bat   # Windows
src/runner_pipeline.sh    # Linux/macOS
```

**Configuration:** Edit `src/rag_config.py` to adjust chunking parameters.

---

#### 3. **`hybrid_rag_module_qwen3.py`** - RAG Core Engine
Implements the retrieval and generation logic.

**Features:**
- Hybrid search combining vector similarity and simple keyword matching
- Single-stage reranking based on weighted semantic and keyword scores
- Similarity threshold filtering
- Configurable prompt templates
- Support for conversation history

**Key Functions:**
- `retrieve()` - Retrieves relevant document chunks
- `generate_response()` - Generates LLM responses with context
- `hybrid_search()` - Combines multiple retrieval strategies

---

---

#### 4. **`dashboard.py`** - Interactive Chat Interface
Multi-user Streamlit application for querying the RAG system.

**Features:**
- Multi-user support with session management
- Shared model resources (optimized VRAM usage)
- Real-time chat interface
- Configurable retrieval parameters (top-k, similarity threshold)
- Model selection (supports multiple LLM backends)
- Conversation history tracking

**Architecture:**
- Single shared LLM instance with thread-safe locking
- Independent user sessions with isolated conversation contexts
- Hybrid retrieval: Vector similarity + keyword search + reranking

**Usage:**
```powershell
python -m streamlit run src/dashboard.py
# or
src/runner_dashboard.bat
```

**Access:** http://localhost:8501



## 🔧 Configuration

### Model Configuration (`src/model_config.py`)

Configure LLM models, inference parameters, and paths:

```python
DEFAULT_DB_PATH = "data/output/chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024_1000_250_cleaned"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
TOP_K_RESULTS = 20
SIMILARITY_THRESHOLD = 50.0  # Percentage
```

### RAG Configuration (`src/rag_config.py`)

Adjust chunking and embedding parameters:

```python
# Chunking strategy: "fixed_size" or "by_sentence"
CHUNK_STRATEGY = "fixed_size"

# Fixed-size chunking parameters
FIXED_SIZE_CHUNK_SIZE = 1000  # Characters per chunk
FIXED_SIZE_OVERLAP = 250      # Character overlap

# Sentence-based chunking parameters
CHUNK_SIZE_MAX_BY_SENTENCE = 1000  # Max characters per chunk

# Embedding configuration
EMBEDDING_DIMENSION = 1024 #Configurable BUT with the only model currently (Qwen/Qwen3-Embedding-0.6B) 1024 must be used!
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
```

---

## 🧪 Testing

The project includes comprehensive test suites:

```powershell
# Run all tests
python -m pytest tests/

# Specific test suites
python tests/test_inference.py --model InternVL3_5-2B-Q6_K --mode quick
python tests/test_inference.py --model Qwen3-4B-Instruct-2507-Q8_0 --mode all --include-environment
python tests/test_chunk_extraction.py
python tests/test_reranking.py
python tests/test_full.py

# Encoding fix for Windows
$env:PYTHONIOENCODING="utf-8" ; python tests/test_inference.py
```

**test_inference.py Arguments:**
- `--model` - LLM model name (e.g., InternVL3_5-2B-Q6_K, Qwen3-4B-Instruct-2507-Q8_0) - VALIDATED
- `--mode` - Test mode: `single` (one question), `all` (all questions), `quick` (5 questions) - VALIDATED
- `--question-id` - Specific question ID to test (for single mode) - TO BE TESTED
- `--max-tokens` - Maximum tokens for LLM response (default: from model_config.py) - TO BE TESTED
- `--db-path` - Path to ChromaDB database (default: from model_config.py) - TO BE TESTED
- `--include-environment` - Include environment variables and Python environment in session log - VALIDATED
- `--show-stats` - Show statistics from logged tests - TO BE TESTED
- `--export-report` - Export test results to Excel report - TO BE TESTED

### Test Coverage
- `test_basics.py` - Basic functionality
- `test_chunk_extraction.py` - Chunking strategies
- `test_inference.py` - LLM inference with multiple models
- `test_reranking.py` - Reranking algorithms
- `test_full.py` - End-to-end pipeline tests

---

## 🐳 Docker Support

### Build and Run with Docker Compose

```powershell
# Build image
docker-compose -f docker-compose.yml build

# Run container
docker-compose -f docker-compose.yml up -d

# View logs
docker logs -f linux_with_cuda-complete-container
```

**Note:** Docker deployment requires models to be copied into the image during build or mounted at runtime.

### Docker Configuration
- `Dockerfile_multi` - Multi-stage build
- `docker-compose.yml` - Standard deployment
- `docker-compose-jenkins.yml` - CI/CD integration with Jenkins

---

## 🔄 CI/CD with Jenkins

The project includes Jenkins integration for automated testing and hyperparameter tuning.

### Jenkins Setup

**Prerequisites:**
- Docker and Docker Compose installed
- Project Docker image built (`linux_with_cuda_complete:latest`)
- GPU access configured (NVIDIA Container Toolkit)

**Start Jenkins Server:**
```powershell
# Build and start Jenkins container
docker-compose -f docker-compose-jenkins.yml up -d

# View logs
docker logs jenkins_server

# Get initial admin password
docker logs jenkins_server | Select-String "password"
```

**Initial Configuration:**
1. Open Jenkins at http://localhost:8080
2. Enter the initial admin password from logs
3. Install suggested plugins
4. Install **Docker Pipeline** plugin (Manage Jenkins → Plugins → Available)
5. Add Neptune.ai API token as credential (ID: `neptune-api-token`) as written in Thesis (secret)

**EXAMPLE: Create Pipeline Job:**
1. Click "New Item" → Enter name → Select "Pipeline"
2. Under Pipeline section:
   - **Definition**: Pipeline script from SCM
   - **SCM**: Git
   - **Repository URL**: `/var/jenkins_home/local_repo`
   - **Script Path**: `Jenkinsfile_docker_all_neptune_common_multiple_models_mult_topk`
3. Click "Save"

**Alternative (Direct Script):**
- Change "Definition" to "Pipeline script"
- Copy content from Jenkinsfile directly into text area

### Available Jenkinsfiles

- `Jenkinsfile_docker` - Basic inference testing with single model
- `Jenkinsfile_docker_all_neptune` - Multi-model testing with Neptune.ai logging
- `Jenkinsfile_docker_all_neptune_common_multiple_models_mult_topk` - Comprehensive testing with TOP_K sweeps across multiple models
- `Jenkinsfile_docker_pipeline` - Full pipeline execution
- `Jenkinsfile_hyperparameter_tuning` - Hyperparameter optimization runs

### Neptune.ai Integration

Configure Neptune.ai credentials in Jenkins:
1. Go to "Manage Jenkins" → "Credentials"
2. Add "Secret text" credential
3. ID: `neptune-api-token`
4. Secret: Your Neptune.ai API token
5. Update `NEPTUNE_PROJECT` in Jenkinsfile with your project name

**Jenkinsfile Environment Variables:**
```groovy
environment {
    NEPTUNE_API_TOKEN = credentials('neptune-api-token')
    NEPTUNE_PROJECT = 'your-workspace/project-name'
    NEPTUNE_UPLOAD_MODE = 'latest'
    RAG_MODEL_CACHE_DIR = '/app/01_RAG/models/huggingface'
}
```

### Run Tests

Click "Build Now" in Jenkins UI to start automated testing. Results are logged to:
- Jenkins console output
- Neptune.ai dashboard (if configured)
- Local log files in `tests/logs/`

---

## 📊 Hyperparameter Tuning

The system includes Neptune.ai integration for experiment tracking:

```powershell
python src/hyperparameter_tuner_neptune.py
```

**Features:**
- Automated parameter search across chunk sizes, overlaps, top-k values
- Performance metrics tracking
- Experiment comparison
- Results logged to Neptune.ai dashboard

**Configuration:** `src/hyperparameter_tuner_neptune.py` and `neptune_uploader.py`

---

## 📁 Project Structure

```
01_RAG/
├── src/                          # Source code
│   ├── data_pipeline_pdf.py      # PDF processing
│   ├── chunk_qwen3_0_6B.py       # Chunking & embedding
│   ├── dashboard.py              # Streamlit UI
│   ├── hybrid_rag_module_qwen3.py # RAG engine
│   ├── model_config.py           # Model configuration
│   ├── rag_config.py             # RAG configuration
│   └── ...
├── tests/                        # Test suites
├── data/                         # Data directory (gitignored)
│   ├── pdfs/                     # Input PDF files
│   ├── output/                   # Processed outputs
│   │   ├── extracted_images/
│   │   ├── image_descriptions/
│   │   ├── final_merged/
│   │   └── chroma_db_*/          # Vector databases
├── models/                       # Model directory (gitignored)
│   ├── huggingface/              # HuggingFace models
│   └── llamacpp/                 # GGUF models for llama.cpp
├── docs/                         # Documentation
├── license/                      # License information
├── requirements_project.txt      # Python dependencies
├── docker-compose.yml            # Docker configuration
└── README.md                     # This file
```

---

## 🔒 Privacy & Offline Operation

This system is designed for **complete offline operation**:
- ✅   All models run locally
- ✅   No external API calls (except optional Neptune.ai logging)
- ✅❓ Data never leaves your machine (sort of..) not perfectly (NLTK etc.)
- ✅  Suitable for sensitive/proprietary documents

---

## ⚙️ Environment Variables

Optional environment variables for advanced configuration:

```powershell
# Model cache directory (overrides default)
$env:RAG_MODEL_CACHE_DIR = "C:\custom\path\models\huggingface"

# GPU selection
$env:CUDA_VISIBLE_DEVICES = "0"

# Python encoding (Windows)
$env:PYTHONIOENCODING = "utf-8"

# HuggingFace cache
$env:HF_HOME = "C:\custom\path\models"

# PyTorch cache
$env:TORCH_HOME = "C:\custom\path\models"
```

---

## 📝 License - AI Generated, not validated

See the `license/` folder for detailed license information for all dependencies.

- `license_modules_map_generated.md` - Module license mapping
- `package_licenses_generated.csv` - Package license summary
- `package_licenses_report_generated.md` - Full license report

---

## 🤝 Contributing

This is a research/thesis project. For issues or questions, please open an issue on GitHub.

---

## 📚 References

- **Qwen Models**: https://huggingface.co/Qwen
- **InternVL**: https://huggingface.co/OpenGVLab
- **ChromaDB**: https://www.trychroma.com/
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Streamlit**: https://streamlit.io/

---

## 🔍 Troubleshooting

### Models Not Found
- Verify models are in `models/huggingface/` and `models/llamacpp/`
- Check paths in `src/model_config.py`
- Set `RAG_MODEL_CACHE_DIR` environment variable if using custom location

### CUDA/GPU Issues
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Set `CUDA_VISIBLE_DEVICES` to select GPU

### Encoding Errors (Windows)
- Set encoding: `$env:PYTHONIOENCODING="utf-8"`
- Use PowerShell (not CMD)

### Out of Memory (OOM)
- Reduce model size (use Q4/Q5 quantization instead of Q8)
- Reduce `n_ctx` in model config
- Reduce load_image(max_num=12) in InternVL 35 modules's function arg further if VRAM is an issue during image understanding or use 2B instead
    in data_pipeline_pdf.py from intevl3_5.InternVL35_4B_reducedv2_single -> from intevl3_5.InternVL35_2B_reducedv2_single
- Reduce batch size (default now 10) or top_k results

---

**Last Updated:** December 2025  
**Branch:** release-thesis_final 