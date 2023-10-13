import numpy as np
import pandas as pd
from docarray import BaseDoc, DocList
from docarray.typing import NdArray, ImageUrl
from pydantic import Field
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
import re
import os
import tempfile

class FashionDocList:
    class FashionDocument(BaseDoc):
        article_id: str
        description: str
        price: int
        image_url: ImageUrl
        embedding: NdArray

    def __init__(self, path: str):
        self.path = path
        self.df = pd.read_csv(path)
        self.df = self.df.rename(columns={'Unnamed: 0': 'ID'})
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.fashion_docs = []

    def create_docschema(self):
        return self.FashionDocument   

    def create_doclist(self):
        for index, row in self.df.iterrows():
            try:
                print(f"Processing row {index}")
                image_url = row["Image_url"]
                response = requests.get(image_url)
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(response.content)
                temp_file.close()
                image = Image.open(temp_file.name)
                image_inputs = self.processor(images=image, return_tensors="pt")
                image_embedding = self.model.get_image_features(**image_inputs)[0].detach().numpy()
                image_embedding = image_embedding / np.linalg.norm(image_embedding, ord=2)
                os.unlink(temp_file.name)
                string = row['Price']
                price = int(re.search(r'\d+', string).group())
                doc = self.FashionDocument(
                    article_id=row['ID'],
                    description=row['Colour'] + ' ' + row['Description'],
                    price=price,
                    image_url=image_url,
                    embedding=image_embedding
                )
                self.fashion_docs.append(doc)
                print(f"Successfully appended document for row {index}")
            except Exception as e:
                print(f"Warning: An error occurred while processing row {index}: {e}")
        docs = DocList[self.FashionDocument](self.fashion_docs)
        print(docs.summary())
        return docs

# Example usage:
# fdl = FashionDocList('path/to/csv')
# docs = fdl.create_doclist()
