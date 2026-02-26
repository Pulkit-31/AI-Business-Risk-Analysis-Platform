import xgboost as xgb
from sklearn.model_selection import train_test_split

X = df[['recency','frequency','monetary']]
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)