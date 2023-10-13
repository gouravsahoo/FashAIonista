import configparser
from pymilvus import connections
from docarray.index import MilvusDocumentIndex
from DocSchema3 import FashionDocList

# ------------------------------------------------------------------------------------------------ #
fashion_inventory = FashionDocList('Flipkart_Sarees_new2.csv')

fashion_docs = fashion_inventory.create_doclist()
fashion_schema = fashion_inventory.create_docschema()
# ------------------------------------------------------------------------------------------------ #

# Load configuration from ini file
cfp = configparser.RawConfigParser()
cfp.read('config_serverless.ini')
milvus_uri = cfp.get('example', 'uri')
token = cfp.get('example', 'token')

# Print configuration for debugging
print(f"Connecting to DB: {milvus_uri}")
print(f"Using Token: {token}")

# Connect to Milvus using the specified URI and token
connections.connect(alias="fashion_db", uri=milvus_uri, token=token)

# Indexing the documents using Milvus and DocArray
doc_index = MilvusDocumentIndex[fashion_schema](index_name="Product-Inventory")
doc_index.index(fashion_docs)

# Searching for documents using text query
query_text = input(str("Enter your Query. "))
results = doc_index.find(query_text)
