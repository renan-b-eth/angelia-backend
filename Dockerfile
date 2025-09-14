# angelia-backend/Dockerfile

# 1. Comece com uma imagem base oficial e leve do Python
FROM python:3.11-slim

# 2. Defina o diretório de trabalho dentro do container
WORKDIR /app

# 3. ATUALIZAÇÃO: Instale o ffmpeg e outras dependências do sistema AQUI
#    Esta etapa é executada durante a construção da imagem, onde temos permissão de escrita.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 4. Copie o arquivo de dependências Python para o container
COPY requirements.txt .

# 5. Instale as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copie todo o resto do seu código da API para o container
COPY . .

# 7. Exponha a porta que a aplicação vai usar
EXPOSE 8000

# 8. Defina o comando para iniciar a API quando o container rodar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]