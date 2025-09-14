# Use uma imagem base Python oficial com Alpine (leve e boa para deploys)
FROM python:3.10-alpine

# 1. Instale as dependências do sistema operacional (para FFmpeg e PostgreSQL)
# Certifique-se de que o ffmpeg e os headers do postgresql-dev estejam instalados
RUN apk add --no-cache ffmpeg build-base postgresql-dev

# 2. Defina o diretório de trabalho dentro do container
WORKDIR /app

# 3. Copie APENAS o requirements.txt primeiro
COPY requirements.txt .

# 4. Instale as dependências Python *antes* de copiar o resto do código
# Use --upgrade pip para garantir a versão mais recente e evitar problemas
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copie todo o resto do seu código da API para o container
# Isso deve ser feito DEPOIS da instalação das dependências
# para que o cache do Docker seja otimizado.
COPY . .

# 6. Exponha a porta que a aplicação FastAPI vai usar
EXPOSE 8000

# 7. Comando para iniciar a aplicação FastAPI
# Assegure que uvicorn esteja disponível e main seja o nome do seu arquivo principal
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]