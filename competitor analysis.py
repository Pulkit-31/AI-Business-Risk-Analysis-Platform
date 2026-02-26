from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis")

reviews = [
    "Great product but delivery slow",
    "Price too high compared to others"
]

results = sentiment_model(reviews)
print(results)