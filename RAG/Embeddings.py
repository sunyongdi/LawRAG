# -*- coding: utf-8 -*-
'''
@File    :   Embeddings.py
@Time    :   2024/07/16 09:39:18
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''


from typing import List
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, get_device_name


class BaseEmbeddings(ABC):
    """
    Base class for embeddings
    """
    @abstractmethod
    def get_embedding(self, text: str, model: str) -> List[float]:
        """"""

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        return cos_sim(vector1, vector2).numpy()



class BgeEmbedding(BaseEmbeddings):
    """
    class for BGE embeddings
    """

    def __init__(self, path: str = 'BAAI/bge-base-zh-v1.5') -> None:
        self._model = self.load_model(path)

    def get_embedding(self, text: str) -> List[float]:
        return self._model.encode(text)

    def load_model(self, path: str):
        model = SentenceTransformer(path, device=get_device_name())
        return model
    
if __name__ == "__main__":
    emb_model = BgeEmbedding('/root/sunyd/model_hub/ZhipuAI/bge-large-zh-v1___5/')
    emb1 = emb_model.get_embedding('孙永荻')
    emb2 = emb_model.get_embedding('吴娜')
    print(emb_model.cosine_similarity(emb1, emb2))
