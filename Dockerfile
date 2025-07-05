FROM python:3.11-slim

# Instala git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Crea carpeta de trabajo
WORKDIR /app

# Copia todos los archivos del proyecto
COPY . /app

# Actualiza pip e instala dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expone el puerto
EXPOSE 8080

# Comando por defecto
CMD ["python3.11", "app.py"]


