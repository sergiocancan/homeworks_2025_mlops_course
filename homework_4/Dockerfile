FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Instala las dependencias necesarias
RUN pip install pandas scikit-learn pyarrow

# Copia el script
COPY hw_4.py .

# Ejecuta el script con argumentos
ENTRYPOINT ["python", "hw_4.py"]
