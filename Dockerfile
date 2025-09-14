# angelia-backend/Dockerfile - Versão Final e Robusta com FFMPEG e todas as suas libs

# Use uma imagem base Python oficial com Debian Bullseye (mais recente e mantida)
FROM python:3.10-slim-bullseye

# Definição de variáveis de ambiente para otimização do Python e PIP
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# 1. Instalar as dependências do sistema operacional necessárias
#    Inclui ffmpeg para processamento de áudio, libpq-dev para PostgreSQL e build-essential para compilação.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libpq-dev \
        build-essential \
        pkg-config \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/*

# 2. Defina o diretório de trabalho dentro do container
WORKDIR /app

# 3. Copie APENAS o requirements.txt primeiro
COPY requirements.txt .

# 4. ATUALIZAÇÃO CRÍTICA: Atualizar as ferramentas de build ANTES de instalar os pacotes
#    Isso é FUNDAMENTAL para resolver o erro 'use_2to3'.
#    setuptools e wheel são tão importantes quanto pip para lidar com setup.py.
RUN pip install --upgrade pip setuptools wheel

# 5. Instale as dependências Python a partir do requirements.txt
#    Permitimos compilação, pois parselmouth e outras libs precisam dela.
RUN pip install -r requirements.txt

# 6. Copie todo o resto do seu código da API para o container
COPY . .

# 7. Exponha a porta que a aplicação FastAPI vai usar
EXPOSE 8000

# 8. Comando para iniciar a aplicação FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]