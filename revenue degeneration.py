import pandas as pd
from prophet import Prophet

# Load sales data
df = pd.read_csv("sales.csv")
df = df.rename(columns={"date": "ds", "revenue": "y"})

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()