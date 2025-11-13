FROM python:3.9-slim

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app
COPY data/ ./data

EXPOSE 8000

CMD ["python", "app/app.py"]