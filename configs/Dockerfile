# Start from NVIDIA CUDA base image
FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/miniconda3/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda with retry mechanism
RUN wget --retry-connrefused \
    --waitretry=1 \
    --read-timeout=20 \
    --timeout=15 --tries=3 \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O miniconda.sh && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm miniconda.sh

# Initialize conda and create environment
RUN conda init bash && \
    . /root/miniconda3/etc/profile.d/conda.sh && \
    conda create -n torch-env python=3.10 -y

# Install packages in smaller chunks with conda
SHELL ["/bin/bash", "-c"]
RUN source /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate torch-env && \
    conda install -y numpy=1.24.3 && \
    conda install -y -c pytorch -c nvidia \
        pytorch \
        torchvision \
        torchaudio && \
    conda clean -ya

# Install remaining packages with pip
RUN source /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate torch-env && \
    pip install --no-cache-dir \
        transformers \
        wandb \
        elasticsearch \
        sacrebleu \
        rouge-score \
        pyyaml \
        tiktoken \
        sentencepiece \
        peft \
        rouge

# Set working directory
WORKDIR /app

# Make sure the environment is activated by default
RUN echo "conda activate torch-env" >> ~/.bashrc

# Clone the GitHub repository
RUN git clone https://github.com/iarroyof/ov-llm-reasoning.git

# Copy the host's timezone files into the container
COPY /etc/localtime /etc/localtime
COPY /etc/timezone /etc/timezone

# Default command
CMD ["/bin/bash", "-l"]
