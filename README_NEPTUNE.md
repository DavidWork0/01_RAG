# Neptune Uploader - README

## Overview

The Neptune uploader automatically uploads your Jenkins test logs to Neptune.ai for visualization and tracking.

## Files Created

```
src/
  └── neptune_uploader.py          # Main uploader script
tests/
  └── test_neptune_integration.py  # Integration test script
docs/
  ├── neptune_integration.md       # Full documentation
  └── neptune_quick_setup.md       # 5-minute setup guide
jenkins_neptune_upload.sh          # Jenkins upload script (Linux)
jenkins_neptune_upload.bat         # Jenkins upload script (Windows)
Jenkinsfile_docker_quick           # Updated with Neptune stage
requirements_project.txt           # Added neptune package
```

## What Gets Uploaded

✅ Session metadata (model, test mode, timestamp)
✅ Hardware information (CPU, GPU, RAM, CUDA)
✅ Model configuration (parameters, settings)
✅ RAG configuration (embeddings, retrieval settings)
✅ Test results (response times, chunks, success rates)
✅ Complete log files
✅ Summary statistics and charts

## Usage Examples

### 1. Upload Latest Session
```bash
python src/neptune_uploader.py --upload-latest
```

### 2. Upload Specific Session
```bash
python src/neptune_uploader.py --session-log tests/logs/sessions/test_session_Qwen3-8B_20251123.log
```

### 3. Upload All Recent Sessions
```bash
python src/neptune_uploader.py --upload-all --limit 10
```

### 4. In Jenkins Pipeline
```bash
bash jenkins_neptune_upload.sh
```

### 5. With Custom Tags
```bash
python src/neptune_uploader.py --upload-latest --tags experiment production v2
```

## Environment Variables

Required:
- `NEPTUNE_API_TOKEN`: Your Neptune API token
- `NEPTUNE_PROJECT`: Your project name (username/project-name)

Optional:
- `NEPTUNE_UPLOAD_MODE`: Upload mode (latest|all|inference)
- `NEPTUNE_TAGS`: Additional tags for runs

## Jenkins Integration

### Current Setup
Your `Jenkinsfile_docker_quick` now includes:

1. **Environment variables** section with Neptune config (commented)
2. **Upload to Neptune** stage that runs after tests
3. **Post actions** that archive logs

### To Enable

1. **Uncomment** Neptune environment variables in `Jenkinsfile_docker_quick`
2. **Configure** `NEPTUNE_API_TOKEN` as Jenkins secret credential
3. **Set** `NEPTUNE_PROJECT` to your project name
4. **Run** your Jenkins pipeline

### Jenkins Credentials Setup

1. Go to: Jenkins → Manage Jenkins → Credentials
2. Click: Add Credentials
3. Select: "Secret text"
4. ID: `neptune-api-token`
5. Secret: Paste your Neptune API token
6. Save

## Testing

Run the integration test to verify everything works:

```bash
# Basic test (no upload)
python tests/test_neptune_integration.py

# Full test with credentials
export NEPTUNE_API_TOKEN="your-token"
export NEPTUNE_PROJECT="username/project"
python tests/test_neptune_integration.py

# Test with actual upload
python tests/test_neptune_integration.py --api-token YOUR_TOKEN --project YOUR_PROJECT
```

## Inside Jenkins Container

When running inside your Jenkins Docker container:

```bash
# Set environment
export NEPTUNE_API_TOKEN="your-token"
export NEPTUNE_PROJECT="username/rag-tests"

# Install neptune (if not in image)
/app/01_RAG/.venv/bin/pip install neptune

# Upload latest session
/app/01_RAG/.venv/bin/python src/neptune_uploader.py --upload-latest
```

## Log Files Location

Your Jenkins container generates logs here:
```
/app/01_RAG/tests/logs/sessions/
```

These are accessible from Jenkins at:
```
http://localhost:8080/job/RAG_Tests_Quick/6/execution/node/4/ws/tests/logs/sessions/
```

The uploader automatically finds and uploads these logs.

## Neptune.ai Dashboard

After uploading, view your results at:
```
https://app.neptune.ai/YOUR_USERNAME/rag-tests
```

### Key Features:
- **Runs Table**: Compare multiple test runs
- **Charts**: Visualize metrics over time
- **Metadata**: Browse all parameters
- **Files**: Download log files
- **Compare**: Side-by-side comparisons

## Common Use Cases

### 1. Track Model Performance Over Time
```bash
# Run tests regularly and upload
python tests/test_inference.py --model Qwen3-8B-Q5_K_M --mode quick
python src/neptune_uploader.py --upload-latest --tags weekly-benchmark
```

### 2. Compare Different Models
```bash
# Upload sessions from different models
python src/neptune_uploader.py --upload-all --limit 20
# Then use Neptune UI to filter and compare by model name
```

### 3. CI/CD Integration
```bash
# In your Jenkins pipeline
stage('Upload to Neptune') {
    sh 'bash jenkins_neptune_upload.sh'
}
```

### 4. Debug Failed Tests
```bash
# Upload failed test session
python src/neptune_uploader.py --session-log path/to/failed_session.log --tags failed debug
```

## Documentation

- **Quick Setup**: [docs/neptune_quick_setup.md](../docs/neptune_quick_setup.md)
- **Full Guide**: [docs/neptune_integration.md](../docs/neptune_integration.md)
- **Neptune Docs**: https://docs.neptune.ai/

## Support

For issues:
1. Check documentation in `docs/neptune_integration.md`
2. Run integration test: `python tests/test_neptune_integration.py`
3. Check Neptune.ai docs: https://docs.neptune.ai/
4. Review Jenkins logs for errors

## Security Notes

⚠️ **Important**:
- Never commit `NEPTUNE_API_TOKEN` to git
- Use Jenkins credentials for secure storage
- Use environment-specific projects (dev/staging/prod)
- Restrict Neptune project access in UI settings

## Next Steps

1. ✅ Read [docs/neptune_quick_setup.md](../docs/neptune_quick_setup.md)
2. ✅ Get Neptune.ai account and credentials
3. ✅ Run `python tests/test_neptune_integration.py`
4. ✅ Configure Jenkins credentials
5. ✅ Enable Neptune stage in Jenkinsfile
6. ✅ Run Jenkins pipeline and view results!
