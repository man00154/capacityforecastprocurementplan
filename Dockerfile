# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# system deps for building wheels (kept small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY . .

# Streamlit settings (port 8501)
ENV STREAMLIT_SERVER_RUN_ON_SAVE=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
