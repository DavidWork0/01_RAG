#!/bin/bash
# Batch Database Evaluation - Multiple Models

python3 batch_db_evaluation.py --model Qwen3-8B-Q5_K_M
python3 batch_db_evaluation.py --model InternVL3_5-8B-Q4_K_M
python3 batch_db_evaluation.py --model InternVL3-2B-Instruct-Q5_K_M
python3 batch_db_evaluation.py --model InternVL3_5-2B-Q8_0
python3 batch_db_evaluation.py --model OpenGVLab_InternVL3_5-8B-Q6_K
