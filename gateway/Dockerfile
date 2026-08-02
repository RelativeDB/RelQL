FROM python:3.12-slim

RUN useradd --create-home --uid 10001 gateway
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py server.py ./
USER gateway
ENV PORT=8080 PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "server.py"]
