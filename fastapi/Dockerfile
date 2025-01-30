FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as builder

RUN apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install  --no-cache-dir --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3.10 python3.10-distutils && \
    useradd -m appuser && \
    mkdir -p /cache/huggingface && \
    chown -R appuser:appuser /cache && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /home/appuser/.local
COPY . /app

ENV PATH="/home/appuser/.local/bin:${PATH}" \
    PYTHONPATH="/app" \
    CUDA_VISIBLE_DEVICES="0" \
    TRANSFORMERS_CACHE="/cache/huggingface"

USER appuser

EXPOSE 8501
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8501"]
