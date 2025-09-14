# angelia-backend/Dockerfi# angelia-backend/Dockerfile

# Use uma imagem base Python oficial com Debian (mais completa e compatível)
# 'slim-buster' é Debian 10, leve mas completa
FROM python:3.10-slim-buster

# 1. Configurar repositórios Debian para incluir 'contrib' e 'non-free'
# Isso garante que codecs e outras dependências do ffmpeg, e talvez build-essential, sejam encontrados.
# E então, atualize a lista de pacotes.
RUN echo "deb http://deb.debian.org/debian buster main contrib non-free" > /etc/apt/sources.list.d/buster_main_contrib_non_free.list && \
    echo "deb http://deb.debian.org/debian buster-updates main contrib non-free" >> /etc/apt/sources.list.d/buster_main_contrib_non_free.list && \
    echo "deb http://security.debian.org/debian-security buster/updates main contrib non-free" >> /etc/apt/sources.list.d/buster_main_contrib_non_free.list && \
    apt-get update

# 2. Instalar as dependências do sistema operacional
# libavformat-dev, libavcodec-dev, libavdevice-dev são comuns para ffmpeg-dev
# ffmpeg é o pacote binário
# libpq-dev é para PostgreSQL (psycopg2-binary)
# build-essential é para compilação
RUN apt-get install -y --no-install-recommends \
        ffmpeg \
        libpq-dev \
        build-essential \
        # Bibliotecas adicionais que podem ser necessárias para compilação robusta de pacotes Python
        pkg-config \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Defina o diretório de trabalho dentro do container
WORKDIR /app

# 4. Copie APENAS o requirements.txt primeiro
COPY requirements.txt .

# 5. Instale as dependências Python *antes* de copiar o resto do código
RUN pip install --upgrade pip --default-timeout=1000 && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 6. Copie todo o resto do seu código da API para o container
COPY . .

# 7. Exponha a porta que a aplicação FastAPI vai usar
EXPOSE 8000

# 8. Comando para iniciar a aplicação FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]