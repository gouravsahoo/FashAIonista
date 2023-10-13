# db_interface.py

from docarray.index import MilvusDocumentIndex
from pymilvus import MilvusClient
import numpy as np

class DatabaseInterface:
    def __init__(self, uri, token=None):
        self.client = MilvusClient(uri=uri, token=token)

    def index_documents(self, docs, index_name):
        doc_index = MilvusDocumentIndex(index_name=index_name)
        doc_index.index(docs)

    def search(self, query_text, processor, model):
        # Creating text embedding using FashionCLIP
        text_inputs = processor(text=query_text, return_tensors="pt", padding=True)
        text_embedding = model.get_text_features(**text_inputs)[0].detach().numpy()
        # Normalizing the embedding to unit norm (so that we can use dot product instead of cosine similarity to do comparisons)
        text_embedding = text_embedding / np.linalg.norm(text_embedding, ord=2)

        # Searching for documents using text embedding
        results = self.doc_index.find(text_embedding)
        return results