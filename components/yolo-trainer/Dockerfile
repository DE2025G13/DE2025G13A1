FROM europe-west4-docker.pkg.dev/data-engineering-vm/ml-services/ultralytics:latest

RUN pip install google-cloud-storage

WORKDIR /app

COPY component.py .

ENTRYPOINT ["python", "component.py"]