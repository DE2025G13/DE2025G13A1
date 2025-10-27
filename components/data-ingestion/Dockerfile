# Use a slim Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the component script
COPY component.py .

# Define the entrypoint for the container
ENTRYPOINT ["python3", "component.py"]