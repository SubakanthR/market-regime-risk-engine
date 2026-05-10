# Cloud-Native Market Regime & Risk Engine

A production-ready ML system that classifies Nifty 50 market conditions into **Normal**, **Volatile**, or **Crisis** regimes and serves probabilistic risk scores via a live REST API.

## Live API
- **Docs & Testing:** https://market-regime-risk-engine.onrender.com/docs
- **Endpoint:** https://market-regime-risk-engine.onrender.com/predict (POST only)

## Sample Request
```bash
curl -X POST "https://market-regime-risk-engine.onrender.com/predict" \
-H "Content-Type: application/json" \
-d '{"RSI": 35.5, "Volatility": 0.018, "ATR": 150.0, "MACD": -20.5}'
```

## Sample Response
```json
{
  "predicted_regime": "Volatile",
  "crisis_risk_score": 0.4222,
  "probabilities": {
    "Crisis": 0.4222,
    "Normal": 0.0080,
    "Volatile": 0.5698
  }
}
```

## Model Performance
| Metric | Score |
|--------|-------|
| Macro F1 | 0.69 |
| Crisis Recall | 0.57 |
| Normal F1 | 0.93 |
| Volatile F1 | 0.82 |

## Features Used
| Feature | Description |
|---------|-------------|
| RSI | Relative Strength Index — momentum indicator |
| Volatility | Rolling 10-day standard deviation of returns |
| ATR | Average True Range — measures market volatility |
| MACD | Moving Average Convergence Divergence — trend indicator |

## Tech Stack
- **ML:** XGBoost + SMOTE
- **API:** FastAPI + Uvicorn
- **Container:** Docker
- **Registry:** AWS ECR
- **Deployment:** Render
- **CI/CD:** GitHub Actions

## Project Structure
```
market-regime-risk-engine/
├── data/
│   └── nifty_50.csv
├── model/
│   ├── regime_model.pkl
│   └── label_encoder.pkl
├── .github/
│   └── workflows/
│       └── deploy.yml
├── marketregime.py
├── risk_engine.py
└── Dockerfile
```

## Run Locally with Docker
```bash
git clone https://github.com/SubakanthR/market-regime-risk-engine.git
cd market-regime-risk-engine
docker build -t risk-engine .
docker run -p 8000:8000 risk-engine
```
Then open: http://127.0.0.1:8000/docs

## Note on AWS
Docker image is pushed to AWS ECR. S3 model storage, IAM roles, and App Runner deployment are in progress pending AWS account activation.
