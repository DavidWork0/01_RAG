#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Set Python
PYTHON="../.venv/bin/python"

# Capture start time (seconds since epoch)
START_TIME=$(date +%s)
START_TIME_DISPLAY=$(date '+%H:%M:%S')

echo "===================================="
echo "Starting Complete RAG Pipeline"
echo "===================================="

echo ""
echo "Step 1: Running data_pipeline_pdf.py"
echo "------------------------------------"
$PYTHON data_pipeline_pdf.py
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: data_pipeline_pdf.py failed with exit code $EXIT_CODE"
    read -p "Press Enter to continue..."
    exit $EXIT_CODE
fi
echo "Step 1 completed successfully."

echo ""
echo "Step 2: Running pre_chunking.py"
echo "------------------------------------"
$PYTHON pre_chunking.py
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: pre_chunking.py failed with exit code $EXIT_CODE"
    read -p "Press Enter to continue..."
    exit $EXIT_CODE
fi
echo "Step 2 completed successfully."

echo ""
echo "Step 3: Running chunk_qwen3_0_6B.py"
echo "------------------------------------"
$PYTHON chunk_qwen3_0_6B.py
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: chunk_qwen3_0_6B.py failed with exit code $EXIT_CODE"
    read -p "Press Enter to continue..."
    exit $EXIT_CODE
fi
echo "Step 3 completed successfully."

echo ""
echo "===================================="
echo "Pipeline Complete"
echo "===================================="

# Calculate and display total time
END_TIME=$(date +%s)
END_TIME_DISPLAY=$(date '+%H:%M:%S')

ELAPSED_SECONDS=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED_SECONDS / 3600))
ELAPSED_MINUTES=$(((ELAPSED_SECONDS % 3600) / 60))
ELAPSED_SECS=$((ELAPSED_SECONDS % 60))

echo ""
echo "Start Time: $START_TIME_DISPLAY"
echo "End Time:   $END_TIME_DISPLAY"
echo "Total Time: ${ELAPSED_HOURS}h ${ELAPSED_MINUTES}m ${ELAPSED_SECS}s"
echo ""

read -p "Press Enter to continue..."
