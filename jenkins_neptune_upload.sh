#!/bin/bash
#
# Jenkins Neptune Upload Script
# ==============================
# Automatically upload test results to Neptune.ai after Jenkins test execution.
# This script should be run as a post-build step in your Jenkinsfile.
#
# Usage:
#   bash jenkins_neptune_upload.sh
#
# Environment Variables (set in Jenkins):
#   NEPTUNE_API_TOKEN: Your Neptune.ai API token (required)
#   NEPTUNE_PROJECT: Your Neptune.ai project name (required)
#   NEPTUNE_UPLOAD_MODE: Upload mode - latest|all|inference (default: latest)
#   NEPTUNE_TAGS: Comma-separated tags (optional)
#   BUILD_NUMBER: Jenkins build number (automatically set by Jenkins)
#   JOB_NAME: Jenkins job name (automatically set by Jenkins)
#

set -e  # Exit on error

echo "=============================================================================="
echo "Neptune.ai Upload Script"
echo "=============================================================================="
echo ""

# Check if Neptune credentials are set
if [ -z "$NEPTUNE_API_TOKEN" ]; then
    echo "❌ ERROR: NEPTUNE_API_TOKEN environment variable is not set"
    echo "   Please configure it in Jenkins credentials or environment variables"
    exit 1
fi

if [ -z "$NEPTUNE_PROJECT" ]; then
    echo "❌ ERROR: NEPTUNE_PROJECT environment variable is not set"
    echo "   Example: username/project-name"
    exit 1
fi

echo "✅ Neptune credentials configured"
echo "   Project: $NEPTUNE_PROJECT"
echo ""

# Set default upload mode
UPLOAD_MODE="${NEPTUNE_UPLOAD_MODE:-latest}"
echo "📤 Upload mode: $UPLOAD_MODE"

# Prepare tags
TAGS="jenkins"
if [ ! -z "$JOB_NAME" ]; then
    TAGS="$TAGS jenkins-$JOB_NAME"
fi
if [ ! -z "$BUILD_NUMBER" ]; then
    TAGS="$TAGS build-$BUILD_NUMBER"
fi
if [ ! -z "$NEPTUNE_TAGS" ]; then
    TAGS="$TAGS $NEPTUNE_TAGS"
fi

echo "🏷️  Tags: $TAGS"
echo ""

# Change to project root
cd /app/01_RAG

# Install neptune if not already installed
echo "📦 Checking neptune installation..."
if ! /app/01_RAG/.venv/bin/python -c "import neptune" 2>/dev/null; then
    echo "   Installing neptune..."
    /app/01_RAG/.venv/bin/pip install neptune --quiet
    echo "   ✅ Neptune installed"
else
    echo "   ✅ Neptune already installed"
fi
echo ""

# Run the uploader
echo "🚀 Starting upload to Neptune.ai..."
echo ""

case "$UPLOAD_MODE" in
    latest)
        /app/01_RAG/.venv/bin/python src/neptune_uploader.py \
            --upload-latest \
            --tags $TAGS
        ;;
    all)
        /app/01_RAG/.venv/bin/python src/neptune_uploader.py \
            --upload-all \
            --limit 10 \
            --tags $TAGS
        ;;
    inference)
        /app/01_RAG/.venv/bin/python src/neptune_uploader.py \
            --upload-inference-logs \
            --tags $TAGS
        ;;
    *)
        echo "❌ ERROR: Invalid NEPTUNE_UPLOAD_MODE: $UPLOAD_MODE"
        echo "   Valid options: latest, all, inference"
        exit 1
        ;;
esac

echo ""
echo "=============================================================================="
echo "✅ Neptune upload completed successfully!"
echo "=============================================================================="
