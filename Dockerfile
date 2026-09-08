# CPU-only image. If your Transcribe module can select faster-whisper's
# int8 CPU mode, that will run noticeably faster than openai-whisper on CPU.
FROM python:3.11-slim

# ffmpeg is required by Whisper (and most audio/video extraction) at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so Docker caches this layer between code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project (main.py, server.py, static/, utils/, core/, ...)
COPY . .

# Force CPU-only: no CUDA device is present in this image, and this also
# stops libraries that auto-detect a GPU from trying to touch one.
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CPU="1"

# Hugging Face Spaces (Docker SDK) always talks to the container on 7860.
EXPOSE 7860

CMD ["python", "server.py"]
