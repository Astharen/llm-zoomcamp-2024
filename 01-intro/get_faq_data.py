import requests 
import tiktoken

from tools.elastic_search.elastic_search_client import ElasticSearchClient


doc_loaded_in_index = True
index_name = "llm-zoomcamp-docs"

max_results = 3

query = "How do I execute a command in a running docker container?"
search_query = {
    "size": max_results,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^4", "text"],
                    "type": "best_fields"
                }
            },
                        "filter": {
                            "term": {
                                "course": "machine-learning-zoomcamp"
                            }
                        }
        }
    }
}

if not doc_loaded_in_index:
    docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    documents = []

    for course in documents_raw:
        course_name = course['course']

        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)


    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"} 
            }
        }
    }

    es_client = ElasticSearchClient()

    es_client.create_index(index_name, index_settings)
    es_client.index_documents(index_name, documents)
else:
    es_client = ElasticSearchClient()

results = es_client.search(index_name, search_query)

raw_contexts = es_client.extract_info_from_search(results)

context_template = """
Q: {question}
A: {text}
""".strip()

context = "\n\n".join([context_template.format(question=raw_context['question'],
                                               text=raw_context['text']) 
                                               for raw_context in raw_contexts])


prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

prompt = prompt_template.format(question=query, context=context)

encoding = tiktoken.encoding_for_model("gpt-4o")

print(len(encoding.encode(prompt)))