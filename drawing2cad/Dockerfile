FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglu1-mesa libxrender1 libxext6 libx11-6 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY d2c ./d2c
COPY server.py run_specs.py ./
COPY specs ./specs
RUN useradd -m app && mkdir -p /app/out && chown app /app/out
USER app
ENV D2C_OUT_DIR=/app/out PYTHONUNBUFFERED=1
# stdio MCP server; run with:  docker run -i --rm --network none -v $PWD/out:/app/out drawing2cad
ENTRYPOINT ["python", "server.py"]
