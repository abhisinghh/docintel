FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin tesseract-ocr-script-deva \
      libreoffice-core libreoffice-writer libreoffice-calc libreoffice-impress \
      fonts-noto fonts-noto-cjk fonts-lohit-deva libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY docintel ./docintel
COPY ui ./ui
COPY configs ./configs
RUN pip install --no-cache-dir -e ".[eval]"

ENV DOCINTEL_DATA_DIR=/data DOCINTEL_CONFIG=/app/configs/offline.yaml
VOLUME ["/data"]
EXPOSE 8080
CMD ["docintel", "serve", "--host", "0.0.0.0", "--port", "8080"]
