FROM pytorch/pytorch:latest
RUN apt update && \
    apt install -y bash \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    mkl
WORKDIR /workspace

RUN python3 -m pip install transformers==4.21.1
RUN python3 -m pip install datasets
RUN python3 -m pip install librosa
ENV HF_HOME="./tmp"
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                      libsndfile1 
ENV NUMBA_CACHE_DIR="./tmp"
COPY modeling_wav2vec2.py /opt/conda/lib/python3.7/site-packages/transformers/models/wav2vec2
CMD ["/bin/bash"]
