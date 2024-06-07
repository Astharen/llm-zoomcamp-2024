from abc import abstractmethod, ABC


class RAGWorkflow:

    def __init__(self, vector_db_client, embedding_model, chat_model) -> None:
        self.vector_db_client = vector_db_client
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.extra_vars = {}

    def load_docs_into_vector_db(self, documents):
        chunks = self.document_splitter(documents)
        emb_vectors = self.convert_chunks_into_vectors(chunks)
        self.load_vectors_into_db(emb_vectors)

    @abstractmethod
    def document_splitter(self, documents) -> list:
        pass

    @abstractmethod
    def convert_chunks_into_vectors(self, chunks):
        pass

    @abstractmethod
    def load_vectors_into_db(self, emb_vectors):
        pass

    def chat_workflow(self, query):
        nearest_vects = self.search_nearest_vector(query)
        context = self.build_context(nearest_vects)
        prompt = self.prepare_prompt(context)
        response = self.chat_response(prompt)
        return response

    @abstractmethod
    def search_nearest_vector(self, query):
        pass
    
    @abstractmethod
    def build_context(self, nearest_vects):
        pass
        
    @abstractmethod
    def prepare_prompt(self, context):
        pass

    @abstractmethod
    def chat_response(self, prompt):
        pass
