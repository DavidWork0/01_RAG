#!/bin/bash
export HF_HOME='/app/01_RAG/models'
/app/01_RAG/.venv/bin/python tests/test_inference.py --model Qwen3-8B-Q5_K_M --mode all
