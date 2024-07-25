#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   Reranker.py
@Time    :   2024/05/15 14:27:42
@Author  :   YueZhengMeng
@Version :   1.0
@Desc    :   None
'''

from typing import List
import numpy as np
from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """
    Base class for reranker
    """
    @abstractmethod
    def rerank(self, text: str, content: List[str], k: int) -> List[str]:
        """"""


class BgeReranker(BaseReranker):
    """
    class for Bge reranker
    """

    def __init__(self, path: str = 'BAAI/bge-reranker-base') -> None:
        self._model, self._tokenizer = self.load_model(path)

    def rerank(self, text: str, content: List[str], k: int) -> List[str]:
        import torch
        pairs = [(text, c) for c in content]
        with torch.no_grad():
            inputs = self._tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            scores = self._model(**inputs, return_dict=True).logits.view(-1, ).float()
            index = np.argsort(scores.tolist())[-k:][::-1]
        return [content[i] for i in index]

    def load_model(self, path: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
        model.eval()
        return model, tokenizer
    
if __name__ == '__main__':
    bge = BgeReranker(path='/root/sunyd/model_hub/Xorbits/bge-reranker-base')
    print(bge.rerank(text='这个电影很不错', content=['这个电影我看的都快睡着了', '什么垃圾电影', '明天吃什么', '你一定要看这个电影'], k=1))
