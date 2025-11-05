# License-to-Modules Mapping

## Quick Reference: Packages by License Type

### MIT License (13 packages)
Permissiveness: ⭐⭐⭐⭐⭐ | Requirements: Include license & copyright notice | Commercial Use: ✅ Free

**Packages:**
- PyYAML
- SQLAlchemy
- instructor
- jsonschema
- langchain
- llama_cpp_python
- onnxruntime
- onnxruntime-gpu
- openai
- pydantic
- pytest
- rich
- typer

---

### Apache 2.0 License (12 packages)
Permissiveness: ⭐⭐⭐⭐⭐ | Requirements: Include license, copyright, & document changes | Patent Grant: ✅ Yes | Commercial Use: ✅ Free

**Packages:**
- accelerate
- aiohttp
- chromadb
- datasets
- google-auth
- grpcio
- onnx
- ragas
- requests
- sentence-transformers
- streamlit
- transformers

---

### BSD 3-Clause License (11 packages)
Permissiveness: ⭐⭐⭐⭐⭐ | Requirements: Include license, copyright, & disclaimer | Name Restriction: Cannot endorse without permission | Commercial Use: ✅ Free

**Packages:**
- click
- httpx
- lxml
- numpy
- pandas
- protobuf
- scikit-learn
- scipy
- torch
- torchvision
- uvicorn

---

### BSD 2-Clause License (1 package)
Permissiveness: ⭐⭐⭐⭐⭐ | Requirements: Include license & copyright notice | Commercial Use: ✅ Free

**Packages:**
- torchaudio

---

### MIT-CMU License (1 package)
Permissiveness: ⭐⭐⭐⭐⭐ | Requirements: Include license & copyright notice | Commercial Use: ✅ Free

**Packages:**
- pillow

---

## License Characteristics Comparison

| Aspect | MIT | Apache 2.0 | BSD 3-Clause | BSD 2-Clause | MIT-CMU |
|---|---|---|---|---|---|
| **Permissiveness** | Highest | Highest | Highest | Highest | Highest |
| **Patent Grant** | No | Yes ✅ | No | No | No |
| **Trademark Rights** | No | Limited | No | No | No |
| **Requires License Notice** | Yes | Yes | Yes | Yes | Yes |
| **Requires Copyright Notice** | Yes | Yes | Yes | Yes | Yes |
| **Requires Source Disclosure** | No | No | No | No | No |
| **Modifications Must Be Disclosed** | No | Yes | No | No | No |
| **Can Be Used in Closed Source** | Yes | Yes | Yes | Yes | Yes |
| **Commercial Use** | Free | Free | Free | Free | Free |

---

## Enterprise Deployment Checklist

### For MIT-Licensed Packages (13 total)
- [ ] Include LICENSE file in distribution
- [ ] Include copyright notices
- [ ] No other documentation required
- ✅ Can be used in proprietary software

**Affected packages:** LangChain, OpenAI, Instructor, Pydantic, SQLAlchemy, Pytest, Rich, Typer, LLaMA CPP Python, ONNXRuntime, PyYAML, JSONSchema

### For Apache 2.0-Licensed Packages (12 total)
- [ ] Include LICENSE file in distribution
- [ ] Include copyright notices
- [ ] **Document any modifications** you make
- [ ] Include NOTICE file if provided
- ✅ Explicit patent grant included
- ✅ Can be used in proprietary software

**Affected packages:** Transformers, Accelerate, Sentence-Transformers, ChromaDB, Ragas, Streamlit, Requests, aiohttp, gRPC, ONNX, Google Auth, Datasets

### For BSD 3-Clause Licensed Packages (11 total)
- [ ] Include LICENSE file in distribution
- [ ] Include copyright notices
- [ ] Include full disclaimer
- [ ] Cannot use contributor names for endorsement
- ✅ Can be used in proprietary software

**Affected packages:** PyTorch, NumPy, Pandas, scikit-learn, SciPy, HTTPx, Click, Uvicorn, lxml, Protobuf

### For BSD 2-Clause Licensed Packages (1 total)
- [ ] Include LICENSE file in distribution
- [ ] Include copyright notices
- ✅ Can be used in proprietary software

**Affected packages:** TorchAudio

### For MIT-CMU Licensed Packages (1 total)
- [ ] Include LICENSE file in distribution
- [ ] Include copyright notices
- ✅ Can be used in proprietary software

**Affected packages:** Pillow

---

## By Use Case

### 🎯 AI/ML Core Stack
**License Distribution:**
- Torch ecosystem: BSD 3-Clause (torch, torchvision, torchaudio)
- Transformers: Apache 2.0 (transformers, accelerate, sentence-transformers)
- Data processing: BSD 3-Clause (numpy, pandas, scipy, scikit-learn)

### 🌐 Web & Networking
**License Distribution:**
- Apache 2.0 (requests, aiohttp, grpcio, streamlit)
- BSD 3-Clause (httpx, uvicorn, click)

### 💾 Data & Vector DBs
**License Distribution:**
- Apache 2.0 (chromadb, datasets, ragas)
- BSD 3-Clause (lxml, protobuf)

### 🤖 LLM & AI Tools
**License Distribution:**
- MIT (langchain, openai, instructor, llama_cpp_python, onnxruntime)
- Apache 2.0 (transformers, google-auth)

### ⚙️ Configuration & Utilities
**License Distribution:**
- MIT (pydantic, PyYAML, SQLAlchemy, jsonschema, pytest)
- BSD 3-Clause (click)

---

## Commercial Use Summary

| License Type | Can Sell Software | Can Modify | Can Relicense | Must Distribute Source |
|---|---|---|---|---|
| MIT | ✅ Yes | ✅ Yes | ✅ Yes (under same terms) | ❌ No |
| Apache 2.0 | ✅ Yes | ✅ Yes | ✅ Yes (under same terms) | ❌ No |
| BSD 3-Clause | ✅ Yes | ✅ Yes | ✅ Yes (under same terms) | ❌ No |
| BSD 2-Clause | ✅ Yes | ✅ Yes | ✅ Yes (under same terms) | ❌ No |
| MIT-CMU | ✅ Yes | ✅ Yes | ✅ Yes (under same terms) | ❌ No |

**Conclusion:** All your dependencies are fully compatible with commercial software development and deployment.

---

## CSV Files Generated

1. **license_summary_by_type.csv** - Summary of licenses with package counts
2. **packages_by_license_detailed.csv** - Detailed list of all packages organized by license

---

*Generated: November 4, 2025*