# Dockerfile for LLM Application with Python 3.11 and CUDA 13.0

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
	
# Install dependencies and Python 3.11 from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnsl-dev \
    libsqlite3-dev \
    libreadline-dev \
    libbz2-dev \
    libexpat1-dev \
    zlib1g-dev \
    liblzma-dev \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and compile Python 3.11.9
RUN wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz && \
    tar -xzf Python-3.11.9.tgz && \
    cd Python-3.11.9 && \
    ./configure --prefix=/usr/local --enable-optimizations && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf Python-3.11.9 Python-3.11.9.tgz

# Create symlink
RUN ln -sf /usr/local/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/local/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements_project_without_torch.txt ./   

# Install dependencies (this layer only rebuilds when requirements change)
RUN pip install -r requirements_project_without_torch.txt

# Copy the entire 01_RAG project structure
COPY 01_RAG/ ./

EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "src/streamlit_modern_multiuser.py", "--server.port=8501", "--server.address=0.0.0.0"]


