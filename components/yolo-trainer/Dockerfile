FROM ultralytics/ultralytics:latest

RUN pip install google-cloud-storage

WORKDIR /app

COPY component.py .

ENTRYPOINT ["python", "component.py"]