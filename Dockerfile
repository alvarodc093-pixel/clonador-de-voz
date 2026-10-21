FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install torch==2.6.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements-space.txt .
RUN pip install -r requirements-space.txt

RUN git clone https://github.com/Plachtaa/seed-vc.git seed-vc \
    && cd seed-vc && git checkout 51383efd921027683c89e5348211d93ff12ac2a8

COPY . .

RUN python scripts/prefetch_modelos.py

EXPOSE 7860
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-7860} --server.address=0.0.0.0 --server.headless=true"]
