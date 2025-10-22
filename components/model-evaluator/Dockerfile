FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements-gpu.txt .

RUN pip install --no-cache-dir -r requirements-gpu.txt

COPY component.py .

ENTRYPOINT ["python", "component.py"]