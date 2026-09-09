FROM python:3.11-slim

# kenlm ships no wheel for any version on any platform, so it compiles here. cmake and a
# compiler build the Python extension; the boost components are what its CMake needs for the
# `build_binary` and `lmplz` executables.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    cmake \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-test-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# kenlm 0.2.0's CMakeLists declares a pre-3.5 minimum, and CMake 4 -- which this base image
# ships -- removed compatibility with those outright. This env var is the escape hatch CMake
# provides for exactly that case.
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    XDG_CACHE_HOME=/opt/cache \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# CPU-only torch: the default PyPI wheel drags in ~2 GB of CUDA that nothing in this app can use.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the voice and the recogniser in so a cold container has nothing to download.
RUN python -c "from stable_twi_tts import StableTwiTTS; StableTwiTTS.from_pretrained()"
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qlerqly/griot-nano-1', local_dir='griot')"

# Copy the app. The KenLM binary + ASR + frontend.
COPY app.py asr.py ./
COPY assets/multilingual.bin ./assets/multilingual.bin
COPY frontend ./frontend

# HF Spaces sets $PORT and expects the Dockerfile to EXPOSE it.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
