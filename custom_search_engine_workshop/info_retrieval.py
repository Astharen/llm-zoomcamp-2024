from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


class DataFrameInfoRetrieval:

    def __init__(self, documents, vect_model, trainable) -> None:
        self.documents = documents
        self.trainable = trainable
        self.vect_model = vect_model.train(self.documents) if trainable else vect_model
        self.vect_docs = self.vect_model.vectorize(documents)

    def keyword_filtering(self, keyword, value):
        return self.vect_docs.query(f'{keyword} == {value}')

    def search_most_similar(self, query):
        vect_query = self.vect_model.vectorize(query)
        