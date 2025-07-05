FROM python:3.10-slim

# Instala git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Crea carpeta de trabajo
WORKDIR /app

# Copia todos los archivos del proyecto
COPY . /app

# Actualiza pip e instala dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Puerto que usará la aplicación (Render inyecta PORT)
ENV PORT ${PORT:-8050}

# Expone el puerto (opcional, Render no lo necesita explícito)
EXPOSE ${PORT}

# Arranca con Gunicorn (app:server es tu Dash app)
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:${PORT}", "--log-file", "-"]



