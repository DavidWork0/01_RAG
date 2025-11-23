# Neptune.ai Quick Setup Guide

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Neptune
```bash
pip install neptune
```

### Step 2: Get Your Credentials

1. Go to [neptune.ai](https://neptune.ai/) and sign up (free)
2. Create a new project: **New Project** → Name it `rag-tests`
3. Get your API token: **Settings** → **API Tokens** → Copy token

### Step 3: Configure Jenkins

Add these environment variables in Jenkins:

```groovy
environment {
    NEPTUNE_API_TOKEN = credentials('neptune-api-token')  // Store as secret!
    NEPTUNE_PROJECT = 'your-username/rag-tests'
    NEPTUNE_UPLOAD_MODE = 'latest'
}
```

**Important**: Store your API token as a Jenkins credential:
- Jenkins → Manage Jenkins → Credentials → Add Credentials
- Kind: "Secret text"
- ID: `neptune-api-token`
- Secret: Paste your Neptune API token

### Step 4: Test the Integration

```bash
# Set environment variables
export NEPTUNE_API_TOKEN="your-token-here"
export NEPTUNE_PROJECT="your-username/rag-tests"

# Run the test
python tests/test_neptune_integration.py
```

### Step 5: Upload Your First Log

```bash
# Upload latest test session
python src/neptune_uploader.py --upload-latest

# Or use the Jenkins script
bash jenkins_neptune_upload.sh
```

---

## 📊 What You'll See in Neptune

After uploading, Neptune will show:

- **Hardware Info**: CPU, GPU, RAM details
- **Model Config**: All model parameters
- **Test Results**: Response times, success rates
- **Charts**: Performance over time
- **Comparisons**: Compare different models side-by-side

---

## 🔧 Jenkins Integration

### Option A: Automatic Upload After Tests

Edit your `Jenkinsfile_docker_quick`:

1. Uncomment Neptune environment variables
2. Configure credentials in Jenkins
3. The pipeline will automatically upload after tests

### Option B: Manual Upload

Run from Jenkins console or as separate job:
```bash
/app/01_RAG/.venv/bin/python src/neptune_uploader.py --upload-latest
```

---

## 📖 Full Documentation

See [docs/neptune_integration.md](neptune_integration.md) for complete documentation.

---

## ❓ Troubleshooting

**"neptune not installed"**
```bash
pip install neptune
```

**"NEPTUNE_API_TOKEN not set"**
```bash
export NEPTUNE_API_TOKEN="your-token-here"
```

**"Connection failed"**
- Check internet connection
- Verify API token is correct
- Check project name format: `username/project-name`

---

## 🎯 Next Steps

1. ✅ Set up Neptune credentials
2. ✅ Test integration
3. ✅ Upload a test session
4. ✅ View results in Neptune UI
5. ✅ Configure Jenkins pipeline
6. ✅ Share with your team!

---

**Need help?** See [neptune_integration.md](neptune_integration.md) or visit [docs.neptune.ai](https://docs.neptune.ai/)
