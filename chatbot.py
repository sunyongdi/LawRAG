# -*- coding: utf-8 -*-
'''
@File    :   chatbot.py
@Time    :   2024/07/25 15:58:29
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''

from RAG import *
from question_classifier import QuestionClassifier


'''问答类'''
class ChatBotGraph:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.llm = QwenModelChat('/root/sunyd/model_hub/qwen/Qwen2-7B-Instruct')
        self.embedding = BgeEmbedding('/root/sunyd/model_hub/ZhipuAI/bge-large-zh-v1___5/')
        self.vector = VectorStore(uri='/root/sunyd/llms/TinyRAG-master/storage/milvus_law.db')
        self.reranker = BgeReranker(path='/root/sunyd/model_hub/Xorbits/bge-reranker-base')

    def chat_main(self, sent):
        answer = '您好，我是小荻.请问有什么需要咨询的！'
        res_classify = self.classifier.classify(sent)
        if len(res_classify['kg_names'])==0:
            prompt = self.llm.generete_prompt(question, answer)
            return self.llm.chat(prompt)
        
        # 从向量数据库中查询出最相似的3个文档
        contents = []
        sim_query = []
        for collection_name in res_classify['kg_names']:
            for content in self.vector.query(question, collection_name=collection_name, EmbeddingModel=self.embedding, k=3):
                sim_query.append(content.key)
                contents.append(content.value)
        if len(contents) == 0:
            return self.llm.chat(sent)
        best_content = ''.join(contents)
        prompt = self.llm.generete_prompt(question, best_content)
        final_answers = self.llm.chat(prompt)
        print(f'相关问题：{sim_query}')
        if not final_answers:
            return answer
        else:
            return final_answers

if __name__ == '__main__':
    handler = ChatBotGraph()
    while 1:
        question = input('用户:')
        answer = handler.chat_main(question)
        print('小勇:', answer)