from milvus import default_server
from pymilvus import connections, utility
from docarray.index import MilvusDocumentIndex
from DocSchema3 import FashionDocList

# ------------------------------------------------------------------------------------------------ #
fashion_inventory = FashionDocList('Flipkart_Sarees_new2.csv')

fashion_docs = fashion_inventory.create_doclist()
fashion_schema = fashion_inventory.create_docschema()
# ------------------------------------------------------------------------------------------------ #

# Start your milvus server
default_server.start()

# Now you can connect with localhost and the given port
# Port is defined by default_server.listen_port
connections.connect(host='127.0.0.1', port=default_server.listen_port)

# Check if the server is ready.
print(utility.get_server_version())

# Indexing the documents using Milvus and DocArray
doc_index = MilvusDocumentIndex[fashion_schema](index_name="Product-Inventory")
doc_index.index(fashion_docs)

# Searching for documents using text query
query_text = input(str("Enter your Query. "))
results = doc_index.find(query_text)

# Stop your milvus server
default_server.stop()