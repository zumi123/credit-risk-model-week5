FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip \
 && pip install -r requirements.txt

ENV MLFLOW_TRACKING_URI=file:/app/mlruns
# supply the best run id at build time:  --build-arg RUN_ID=abc123
ARG RUN_ID
ENV BEST_MODEL_RUN_ID=${RUN_ID}

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
