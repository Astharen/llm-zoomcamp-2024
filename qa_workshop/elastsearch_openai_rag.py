from tools.elastic_search.elastic_search_client import ElasticSearchClient
from tools.elastic_search.index_settings import ElasticSearchIndexSettings
from tools.llms.openai.base import BaseOpenAIClient
from tools.rag.rag import RAGWorkflow


class ElastSearchOpenAIRAG(RAGWorkflow):

    def __init__(self, vector_db_client: ElasticSearchClient, 
                 embedding_model: BaseOpenAIClient, chat_model: BaseOpenAIClient) -> None:
        super().__init__(vector_db_client, embedding_model, chat_model)
        self.extra_vars['index_name'] = "course-questions"

    def document_splitter(self, docs) -> list:
        documents = []

        for course in docs:
            course_name = course['course']

            for doc in course['documents']:
                doc['course'] = course_name
                documents.append(doc)
        return documents

    def convert_chunks_into_vectors(self, chunks):
        return chunks

    def load_vectors_into_db(self, emb_vectors):
        properties = {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"}
            }
        index_name = self.extra_vars['index_name']
        es_index = ElasticSearchIndexSettings(name=index_name, properties=properties)
        self.vector_db_client.create_index(es_index)
        self.vector_db_client.index_documents(index_name=index_name, documents=emb_vectors)

    def search_nearest_vector(self, query):
        search_query = {
            "size": 5,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["question^3", "text", "section"],
                            "type": "best_fields"
                        }
                    },
                    "filter": {
                        "term": {
                            "course": "data-engineering-zoomcamp"
                        }
                    }
                }
            }
        }
        index_name = self.extra_vars['index_name']
        
        response = self.vector_db_client.search(index_name=index_name, 
                                                search_query=search_query)

        results = self.vector_db_client.extract_info_from_search(response)
        return results
    
    def build_context(self, nearest_vects):
        context = ""

        for doc in nearest_vects:
            doc_str = f"Section: {doc['section']}\nQuestion: {doc['question']}\nAnswer: {doc['text']}\n\n"
            context += doc_str

        context = context.strip()
        return context
        
    def prepare_prompt(self, query, context):
        system_prompt = """You're a course teaching assistant. Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database. 
        Only use the facts from the CONTEXT. If the CONTEXT doesn't contan the answer, return "NONE"}]"""

        user_prompt = """
        QUESTION: {user_question}

        CONTEXT:

        {context}
        """.strip()

        system_prompt = system_prompt.format(user_question=query, context=context)

        messages=[{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt}]
        return messages

    def chat_response(self, prompt):
        return self.chat_model.chat(messages=prompt)