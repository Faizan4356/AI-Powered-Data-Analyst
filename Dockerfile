FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt packages.txt ./
RUN apt-get update \
    && xargs -a packages.txt apt-get install -y \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
