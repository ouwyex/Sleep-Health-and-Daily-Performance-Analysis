# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim
 
# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app
 
# ── Install dependencies first (layer-cache friendly) ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# ── Copy project files ────────────────────────────────────────────────────────
COPY train.py .
COPY main.py  .
COPY app.py   .
 
# ── Train the model + log to local MLflow at build time ───────────────────────
RUN python train.py
 
# ── Expose ports: 8000 = FastAPI, 8501 = Streamlit ───────────────────────────
EXPOSE 8000
EXPOSE 8501
 
# ── Run the FastAPI app with Uvicorn (default entrypoint) ─────────────────────
# To run Streamlit instead: docker run ... streamlit run app.py --server.port 8501
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
 