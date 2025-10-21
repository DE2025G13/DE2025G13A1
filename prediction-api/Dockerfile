FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

ENV MODEL_BUCKET glasses-model

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]