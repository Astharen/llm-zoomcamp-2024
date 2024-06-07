import logging
from typing import List

from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

from tools.elastic_search.index_settings import ElasticSearchIndexSettings

class ElasticSearchClient:

    def __init__(self, host='localhost', port='9200'):
        self.client = Elasticsearch(f"http://{host}:{port}")

        logging.info(self.client.info())

    def create_index(self, index: ElasticSearchIndexSettings):
        response = self.client.indices.create(index=index.name, body=index.index_settings)
        return response
    
    def index_documents(self, index_name, documents: List[dict]):
        for doc in tqdm(documents):
            self.client.index(index=index_name, document=doc)

    def search(self, index_name, search_query):
        """
        Query DSL info: https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html#query-dsl

        Example of query: here, the question match are consider above the rest.

        search_query = {
            "size": max_results,
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
        """
        return self.client.search(index=index_name, body=search_query)
    
    def extract_info_from_search(self, response):
        documents = [hit['_source'] for hit in response['hits']['hits']]
        return documents
    