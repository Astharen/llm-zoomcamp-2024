import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from custom_search_engine_workshop.vectorizer_models.base import VectorizerModel


class SklearnVectorizerModel(VectorizerModel):

    def train(self, docs):
        self.model.fit(docs)
    
    def clean_output_to_df(self, output, is_df):
        df_docs = output.toarray().T
        if is_df:
            names = self.model.get_feature_names_out()
            df_docs = pd.DataFrame(df_docs, index=names)
        return df_docs

    def convert_to_vector(self, docs):
        vect_docs = self.model.transform(docs)
        vect_docs = self.clean_output_to_df(vect_docs, isinstance(docs, pd.DataFrame))
        return vect_docs


class CountVectorizerModel(SklearnVectorizerModel):

    def __init__(self, **kwargs):
        self.model = CountVectorizer(stop_words='english', **kwargs)


class TfidfVectorizerModel(SklearnVectorizerModel):

    def __init__(self, **kwargs):
        self.model = CountVectorizer(stop_words='english', **kwargs)
