# angelia-backend/Dockerfile - Versão Final com Librosa

# Usar uma imagem base Python oficial com Debian Bullseye
FROM python:3.10-slim-bullseye

# Definição de variáveis de ambiente para otimização
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# 1. Instalar as dependências do sistema operacional
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libpq-dev \
        build-essential \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Defina o diretório de trabalho
WORKDIR /app

# 3. Copie o requirements.txt
COPY requirements.txt .

# 4. ATUALIZAÇÃO CRÍTICA: Atualizar as ferramentas de build
RUN pip install --upgrade pip setuptools wheel

# 5. Instale as dependências Python
RUN pip install -r requirements.txt

# 6. Copie o resto do código
COPY . .

# 7. Exponha a porta
EXPOSE 8000

# 8. Comando para iniciar a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]