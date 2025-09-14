# angelia-backend/Dockerfile

# Use uma imagem base Python oficial com Debian Bullseye (mais recente e mantida)
FROM python:3.10-slim-bullseye

# 1. Instalar as dependências do sistema operacional
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libpq-dev \
        build-essential \
        pkg-config \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Defina o diretório de trabalho dentro do container
WORKDIR /app

# 3. Copie APENAS o requirements.txt primeiro
COPY requirements.txt .

# 4. Instale o pip em uma versão específica (um pouco mais antiga para evitar bugs recentes)
# E então instale as dependências Python, forçando o uso de wheels.
# --no-deps (muito agressivo, pode quebrar se houver muitas dependências compiladas)
# --only-binary :all: é a chave para evitar a execução de setup.py
RUN pip install --upgrade "pip<24.0" && \
    pip install --no-cache-dir --default-timeout=1000 --only-binary :all: -r requirements.txt

# 5. Copie todo o resto do seu código da API para o container
COPY . .

# 6. Exponha a porta que a aplicação FastAPI vai usar
EXPOSE 8000

# 7. Comando para iniciar a aplicação FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]