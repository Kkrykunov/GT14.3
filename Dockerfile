# Використовуємо Python з базового образу
FROM python:3.9

# Задаємо робочу папку всередині контейнера
WORKDIR /app

# Копіюємо файли з вашої папки до контейнера
COPY . /app

# Встановлюємо залежності, якщо є requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || echo "No requirements.txt found"

# Команда за замовчуванням для запуску проекту
CMD ["python", "main.py"]  # Замініть main.py на ваш стартовий файл
