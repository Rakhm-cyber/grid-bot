FROM python:3.11-slim AS base

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Порт для веб-графика бота
EXPOSE 8099

# По умолчанию запускаем пайплайн
CMD ["python", "pipeline.py", "--full", "--dry-run"]
