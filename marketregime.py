import pandas as pd
import numpy as np
from collections import Counter

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

df = pd.read_csv("/Users/subakanth/Documents/Market_Regime/data/nifty_50.csv", index_col=0)
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)
print("\n========== DATA LOADED ========")
print(df.head())
df['Returns']   = df['Close'].pct_change()
df['Volatility'] = df['Returns'].rolling(10).std()
df['RSI']       = RSIIndicator(close=df['Close']).rsi()
df['SMA_20']    = SMAIndicator(close=df['Close'], window=20).sma_indicator()
df['EMA_20']    = EMAIndicator(close=df['Close'], window=20).ema_indicator()
df['MACD']      = MACD(close=df['Close']).macd()
df['ATR']       = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
df.dropna(inplace=True)
def classify_regime(row):
    if row['Returns'] < -0.010 and row['Volatility'] > 0.012:  
        return "Crisis"
    elif row['Volatility'] > 0.008:                             
        return "Volatile"
    else:
        return "Normal"

df['Regime'] = df.apply(classify_regime, axis=1)

print("\n========= REGIME DISTRIBUTION =======")
print(df['Regime'].value_counts())
df['Target'] = df['Regime'].shift(-1)
df.dropna(inplace=True)
encoder = LabelEncoder()
df['Target'] = encoder.fit_transform(df['Target'])
print("\n======= LABEL MAPPING ========")
for i, cls in enumerate(encoder.classes_):
    print(cls, "→", i)
features = ['RSI', 'Volatility', 'ATR', 'MACD']
X = df[features]
y = df['Target']
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("\nTest class distribution:", Counter(y_test))
print("\nBefore SMOTE:", Counter(y_train))
sm = SMOTE(random_state=42, k_neighbors=3)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
print("After SMOTE: ", Counter(y_train_bal))
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    gamma=0.2,
    random_state=42,
    eval_metric='mlogloss',
    objective='multi:softprob'
)
model.fit(X_train_bal, y_train_bal)
y_proba = model.predict_proba(X_test)
y_pred  = np.argmax(y_proba, axis=1)
print("\n======= CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))
print("\n========== CONFUSION MATRIX =========")
print(confusion_matrix(y_test, y_pred))
print("\nMacro F1:", f1_score(y_test, y_pred, average='macro'))
import joblib
import os
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/regime_model.pkl")
joblib.dump(encoder, "model/label_encoder.pkl")
print("\n Model saved to model/regime_model.pkl")
print(" Encoder saved to model/label_encoder.pkl")