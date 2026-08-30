# 1. Use an official lightweight Python base image
FROM python:3.11-slim

# 2. Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. Set the working directory
WORKDIR /app

# 4. Install uv for fast package management
RUN pip install --no-cache-dir uv

# 5. Copy project definition files and install dependencies
COPY pyproject.toml uv.lock* ./
RUN uv pip install --system --no-cache -r pyproject.toml || uv pip install --system --no-cache pytest fastapi uvicorn streamlit requests joblib scikit-learn numpy structlog pydantic

# 6. Copy the rest of the application code
COPY . .

# 7. Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# 8. Start FastAPI backend server in background and Streamlit frontend
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]