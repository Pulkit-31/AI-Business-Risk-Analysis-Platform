from sklearn.ensemble import IsolationForest
import pandas as pd

expenses = pd.read_csv("expenses.csv")

model = IsolationForest(contamination=0.05)
expenses['anomaly'] = model.fit_predict(expenses[['amount']])

anomalies = expenses[expenses['anomaly'] == -1]