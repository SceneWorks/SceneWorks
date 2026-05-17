FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY apps/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/api ./apps/api
COPY packages/shared ./packages/shared
ENV PYTHONPATH=/app/apps/api:/app/packages/shared

EXPOSE 8000

CMD ["python", "-m", "sceneworks_api"]
