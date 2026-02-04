FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app


# System deps for SHAP / numba
RUN apt-get update && apt-get install -y \
    build-essential \
    llvm \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
