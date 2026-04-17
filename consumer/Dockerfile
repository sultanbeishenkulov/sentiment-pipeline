FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model artifacts are mounted as a volume from the host,
# not baked into the image (too large, changes independently).
# Default path inside container: /app/model_artifacts
ENV MODEL_PATH=/app/model_artifacts

CMD ["python", "main.py"]
