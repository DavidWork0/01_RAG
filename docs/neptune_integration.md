# Neptune.ai Integration for RAG Test Results

This document explains how to upload your Jenkins test results to Neptune.ai for advanced data visualization, tracking, and evaluation.

## Overview

Neptune.ai is a metadata store for MLOps that allows you to:
- Track and compare test runs
- Visualize performance metrics over time
- Monitor model performance across different configurations
- Share results with team members
- Generate reports and insights

## Setup

### 1. Install Neptune

```bash
pip install neptune
```

Or add to your requirements:
```
neptune>=1.0.0
```

### 2. Get Neptune.ai Credentials

1. Sign up for a free account at [neptune.ai](https://neptune.ai/)
2. Create a new project (e.g., `username/rag-tests`)
3. Get your API token from: https://app.neptune.ai/ → Settings → API Tokens

### 3. Configure Jenkins Environment Variables

In your Jenkins configuration, add the following environment variables:

#### Required:
- `NEPTUNE_API_TOKEN`: Your Neptune.ai API token
- `NEPTUNE_PROJECT`: Your project name (e.g., `username/rag-tests`)

#### Optional:
- `NEPTUNE_UPLOAD_MODE`: Upload mode - `latest` (default), `all`, or `inference`
- `NEPTUNE_TAGS`: Additional tags for runs (space/comma separated)

**Security Note**: Store `NEPTUNE_API_TOKEN` as a Jenkins secret credential!

## Usage

### From Command Line

#### Upload Latest Test Session
```bash
python src/neptune_uploader.py --upload-latest
```

#### Upload Specific Session
```bash
python src/neptune_uploader.py --session-log tests/logs/sessions/test_session_Qwen3-8B_20251123.log
```

#### Upload All Sessions (limit to 10 most recent)
```bash
python src/neptune_uploader.py --upload-all --limit 10
```

#### Upload Inference Logs
```bash
python src/neptune_uploader.py --upload-inference-logs --model Qwen3-8B-Q5_K_M
```

#### With Custom Tags
```bash
python src/neptune_uploader.py --upload-latest --tags jenkins ci quick-test
```

### From Jenkins

#### Option 1: Add Post-Build Step to Jenkinsfile

Add this stage to your `Jenkinsfile_docker_quick`:

```groovy
stage('Upload to Neptune') {
    steps {
        script {
            echo 'Uploading results to Neptune.ai...'
            
            // Set environment for this stage
            withEnv([
                "NEPTUNE_UPLOAD_MODE=latest"
            ]) {
                // Run upload script
                sh 'bash jenkins_neptune_upload.sh'
            }
        }
    }
}
```

#### Option 2: Run Upload Script Directly

In your Jenkinsfile, add:

```groovy
stage('Upload to Neptune') {
    steps {
        script {
            sh '''
                /app/01_RAG/.venv/bin/python src/neptune_uploader.py \
                    --upload-latest \
                    --tags jenkins build-${BUILD_NUMBER} ${JOB_NAME}
            '''
        }
    }
}
```

#### Option 3: Use Post Actions (Always Run, Even on Failure)

```groovy
post {
    always {
        script {
            echo 'Uploading test results to Neptune.ai...'
            try {
                sh 'bash jenkins_neptune_upload.sh'
            } catch (Exception e) {
                echo "Warning: Neptune upload failed: ${e.message}"
            }
        }
    }
}
```

### From Docker Container

If running inside a Docker container:

```bash
# Inside container
cd /app/01_RAG
export NEPTUNE_API_TOKEN="your-token-here"
export NEPTUNE_PROJECT="username/rag-tests"

# Upload latest session
/app/01_RAG/.venv/bin/python src/neptune_uploader.py --upload-latest
```

## What Gets Uploaded

The uploader automatically extracts and uploads:

### Session Metadata
- Session name and timestamp
- Test mode (single/quick/all)
- Selected model name

### Hardware Information
- Operating system details
- CPU information
- RAM capacity and usage
- GPU details (name, CUDA version, memory)

### Model Configuration
- Context size
- Temperature settings
- GPU layers
- All model parameters

### RAG Configuration
- Embedding model
- Top-K settings
- Semantic/keyword weights
- Retrieval parameters

### Test Results
For each question:
- Question ID and text
- Response time
- Number of chunks retrieved
- Answer length
- Success/failure status

### Summary Statistics
- Total tests run
- Success rate
- Average response time
- Average chunks retrieved
- Min/max response times

### Files
- Complete session log file
- Environment report (if available)
- CSV/JSONL inference logs

## Viewing Results in Neptune

After uploading, you'll get a URL to view your results:

```
🔗 View in Neptune: https://app.neptune.ai/username/rag-tests/e/RAG-123
```

### Navigate Neptune UI:

1. **Runs Table**: Compare multiple test runs side-by-side
2. **Charts**: Visualize metrics over time
3. **Metadata**: Browse all logged parameters
4. **Files**: Download uploaded log files
5. **Compare**: Select multiple runs to compare

### Useful Queries in Neptune:

- Filter runs by tags: `jenkins`, `build-123`, `Qwen3-8B-Q5_K_M`
- Compare models: Group by `metadata/model_name`
- Track performance: Chart `summary/avg_response_time` over time
- Analyze failures: Filter by `summary/success_rate < 100`

## Advanced Usage

### Custom Metadata

You can extend the uploader to add custom metadata:

```python
from src.neptune_uploader import NeptuneUploader

uploader = NeptuneUploader(
    api_token="your-token",
    project="username/project"
)

# Upload with custom tags and description
uploader.upload_session(
    session_log_path=Path("tests/logs/sessions/test_session_xyz.log"),
    tags=["experiment", "custom-config", "v2"],
    description="Testing new RAG configuration with updated weights"
)
```

### Batch Upload Multiple Sessions

```python
# Upload last 5 sessions
uploader.upload_all_sessions(
    tags=["batch-upload"],
    limit=5
)
```

### Integration with CI/CD

Set environment variables in your CI/CD pipeline:

**GitHub Actions:**
```yaml
env:
  NEPTUNE_API_TOKEN: ${{ secrets.NEPTUNE_API_TOKEN }}
  NEPTUNE_PROJECT: "username/rag-tests"
```

**Jenkins:**
```groovy
environment {
    NEPTUNE_API_TOKEN = credentials('neptune-api-token')
    NEPTUNE_PROJECT = 'username/rag-tests'
}
```

## Troubleshooting

### "neptune package not installed"
```bash
pip install neptune
```

### "NEPTUNE_API_TOKEN not set"
Set environment variable:
```bash
export NEPTUNE_API_TOKEN="your-token-here"
```

### "Project not found"
Verify project name format: `username/project-name`
Check you have access to the project in Neptune UI.

### Upload fails with connection error
- Check internet connectivity
- Verify API token is correct
- Check Neptune.ai status page

### Large log files
Neptune has file size limits. For very large logs:
- Use `--limit` to upload fewer sessions
- Compress log files before uploading
- Use inference logs mode for summarized data

## Performance Considerations

- **Upload time**: ~5-30 seconds per session depending on log size
- **Storage**: Free tier includes 100GB storage
- **Network**: Requires internet connection from Jenkins container

## Security Best Practices

1. **Never commit API tokens** to git repositories
2. **Use Jenkins credentials** to store `NEPTUNE_API_TOKEN`
3. **Restrict project access** in Neptune UI settings
4. **Use environment-specific projects** (dev/staging/prod)

## Example Jenkinsfile Integration

Here's a complete example:

```groovy
pipeline {
    agent {
        docker {
            image 'linux_with_cuda_complete:latest'
            args '--gpus all --entrypoint=""'
            reuseNode true
        }
    }
    
    environment {
        NEPTUNE_API_TOKEN = credentials('neptune-api-token')
        NEPTUNE_PROJECT = 'username/rag-tests'
        NEPTUNE_UPLOAD_MODE = 'latest'
    }
    
    stages {
        stage('Run Tests') {
            steps {
                script {
                    sh '/app/01_RAG/.venv/bin/python tests/test_inference.py --model Qwen3-8B-Q5_K_M --mode quick'
                }
            }
        }
        
        stage('Upload to Neptune') {
            steps {
                script {
                    sh 'bash jenkins_neptune_upload.sh'
                }
            }
        }
    }
    
    post {
        always {
            script {
                // Fallback: try to upload even if tests failed
                try {
                    sh '''
                        /app/01_RAG/.venv/bin/python src/neptune_uploader.py \
                            --upload-latest \
                            --tags jenkins build-${BUILD_NUMBER} ${JOB_NAME} failed
                    '''
                } catch (Exception e) {
                    echo "Neptune upload failed: ${e.message}"
                }
            }
        }
    }
}
```

## Resources

- [Neptune.ai Documentation](https://docs.neptune.ai/)
- [Neptune Python API](https://docs.neptune.ai/api/neptune/)
- [Integrations](https://docs.neptune.ai/integrations/)
- [Best Practices](https://docs.neptune.ai/logging/best_practices/)

## Support

For issues with Neptune integration:
1. Check this documentation
2. Review Neptune.ai docs
3. Check Jenkins logs for error details
4. Contact Neptune support at support@neptune.ai
