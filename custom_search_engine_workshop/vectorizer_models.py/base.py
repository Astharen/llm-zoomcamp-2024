from abc import ABC, abstractmethod


class VectorizerModel(ABC):

    @property
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def train(self, docs):
        raise NotImplementedError
    
    @abstractmethod
    def convert_to_vector(self, docs):
        raise NotImplementedError
    
    def train_vectorize(self, docs):
        self.train(docs)
        return self.convert_to_vector(docs)