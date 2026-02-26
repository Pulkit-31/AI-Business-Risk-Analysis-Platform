from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

docs = ["delivery late", "app crashes often", "bad customer service"]

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(docs)

lda = LatentDirichletAllocation(n_components=2)
lda.fit(X)