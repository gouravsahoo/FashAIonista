import numpy as np
import pandas as pd
# import docarray

from docarray import BaseDoc, DocList
from docarray.typing import NdArray, ImageUrl
from pydantic import Field
from PIL import Image

# from fashion_clip.fashion_clip import FashionCLIP
from transformers import CLIPProcessor, CLIPModel

import requests
import re
import os
import tempfile

df = pd.read_csv("Flipkart_Sarees_new2.csv")
df = df.rename(columns={'Unnamed: 0': 'ID'})
# dropping items that have the same description
# df_subset = df.drop_duplicates("Description").copy()


# Defining a custom document schema
class FashionDocument(BaseDoc):
    article_id: str
    description: str
    price: int
    image_url: ImageUrl
    embedding: NdArray

# fclip = FashionCLIP('fashion-clip')
# Loading the FashionCLIP model and processor
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

# Loading the dataset
# articles_subset

# Creating a list to store the FashionDocument objects
fashion_docs = []

# Adding samples to the list
for index, row in df.iterrows():
    try:
        # Printing the index of the row being processed
        print(f"Processing row {index}")

        # Getting the image URL for the article from the Kaggle dataset
        image_url = row["Image_url"]

        # Loading the image from the URL

        # Downloading the image file
        response = requests.get(image_url)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(response.content)
        temp_file.close()

        # Opening the image file
        image = Image.open(temp_file.name)

        # Creating image embedding using FashionCLIP
        # image_embedding = fclip.encode_images([image], batch_size = 32)[0]
        image_inputs = processor(images=image, return_tensors="pt")
        # image_embedding = model.get_image_features(**image_inputs)[0]
        image_embedding = model.get_image_features(**image_inputs)[0].detach().numpy()
            
        # Normalizing the embedding to unit norm (so that we can use dot product instead of cosine similarity to do comparisons)
        image_embedding = image_embedding / np.linalg.norm(image_embedding, ord=2)

        # Deleting the temporary file
        os.unlink(temp_file.name)

        # headers = {'User-Agent': 'Mozilla/5.0'}
        # response = requests.get(image_url, headers=headers, stream=True)
        # image = Image.open(response.raw)
        # image = Image.open(requests.get(image_url, stream=True).raw)

        # Converting 'Price' string to integer
        string = row['Price']
        price = int(re.search(r'\d+', string).group())

        # Creating a FashionDocument object schema
        doc = FashionDocument(
            article_id=row['ID'],
            description=row['Colour'] + ' ' + row['Description'],
            price=price,
            image_url=image_url,
            embedding=image_embedding
        )

        # Appending the document to the list
        fashion_docs.append(doc)

        # Printing a success message
        print(f"Successfully appended document for row {index}")
    except Exception as e:
        print(f"Warning: An error occurred while processing row {index}: {e}")

# Creating a DocList from the list of FashionDocument objects
#docs = DocList(fashion_docs)
docs = DocList[FashionDocument](fashion_docs)
    
print(docs.summary())

# --------------------------------------------------------------------------------------------------------- #
# DB Agnostic. The db_interface.py file is an abstract interface for interacting with any database. Combined  
# with docarray schema, one can use this interface to connect to other databases, by creating a new class.
# --------------------------------------------------------------------------------------------------------- #

# Interfacing with Vector db (Milvus). 
from db_interface import Milvus_db

#db = DatabaseInterface(uri="milvus_uri", token="milvus_token")
db = Milvus_db(uri="https://in03-b6b813853136c82.api.gcp-us-west1.zillizcloud.com",
               token="eea5b4447fe7d400bd4643bccab53f081f0332f1072b02be1f81a202d0b4f1091e4e05083876e32ada12b035f7370e95a58aa2c3")
db.index_documents(docs, index_name='Product_Inventory')

query_text = input(str("Enter your Query: "))
results = db.search(query_text, processor, model)