# -*- coding: utf-8 -*-
'''
@File    :   VectorBase.py
@Time    :   2024/07/16 10:01:11
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''

from typing import List
from .Embeddings import BaseEmbeddings
from pymilvus import MilvusClient
from pydantic import BaseModel

class ItemModel(BaseModel):
    key: str
    value: str
    score: float

class ReturnModel(BaseModel):
    topK: List[ItemModel]

class VectorStore:
    def __init__(self, uri:str="http://192.168.4.177:19530") -> None:
        self.milvus_client = MilvusClient(uri=uri)

        
    def query(self, query: str, collection_name: str, EmbeddingModel: BaseEmbeddings, k: int = 1, threshold=0.5) -> List[str]:
        search_res = self.milvus_client.search(
            collection_name=collection_name,
            data=[
                EmbeddingModel.get_embedding(query)
            ],  # Use the `emb_text` function to convert the question to an embedding vector
            limit=k,  # Return top 3 results
            # search_params={"metric_type": "IP", "params": {}},
            search_params={"metric_type": "IP", "params": {'radius': threshold}}, # Inner product distance
            output_fields=["key", "value"],  # Return the text field
            
        )
        # return [(res["entity"]["value"], res["distance"]) for res in search_res[0]]
        return [ItemModel(key=res["entity"]["key"], value=res["entity"]["value"], score=res["distance"]) for res in search_res[0]]

if __name__ == '__main__':
    from Embeddings import BgeEmbedding
    emb_model = BgeEmbedding('/root/sunyd/model_hub/ZhipuAI/bge-large-zh-v1___5/')
    vector = VectorStore(uri='/root/sunyd/llms/TinyRAG-master/storage/milvus_law.db')
    prompt = '为这个句子生成表示以用于检索相关文章：'
    res = vector.query(f'{prompt}为了确保关键信息基础设施供应链安全', 'JudicialExamination', emb_model, 1, 0.3)
    print(res.topK)
    
    
