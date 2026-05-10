FROM python:3.11-slim

WORKDIR /app

COPY model/ ./model/
COPY risk_engine.py .

RUN pip install fastapi uvicorn xgboost scikit-learn joblib numpy

EXPOSE 8000

CMD ["uvicorn", "risk_engine:app", "--host", "0.0.0.0", "--port", "8000"]