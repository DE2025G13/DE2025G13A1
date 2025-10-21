FROM python:3.9-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY component.py .

ENTRYPOINT ["python3", "component.py"]