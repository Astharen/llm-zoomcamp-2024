from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_dist():
    cosine_similarity()

def dot_product(doc_vect, query_vect):
    return doc_vect.dot(query_vect.T).toarray()