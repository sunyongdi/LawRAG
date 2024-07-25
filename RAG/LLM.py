# -*- coding: utf-8 -*-
'''
@File    :   LLM.py
@Time    :   2024/07/15 17:26:51
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''
from typing import Dict, List
from abc import ABC, abstractmethod

PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
    InternLM_PROMPT_TEMPALTE="""先对上下文进行内容总结,再使用上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
)


class BaseModel(ABC):
    @abstractmethod
    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        """"""

    def generete_prompt(self, quert: str, content: str) -> str:
        return PROMPT_TEMPLATE['InternLM_PROMPT_TEMPALTE'].format(question=quert, context=content)
    


class QwenModelChat(BaseModel):
    def __init__(self, path: str = '') -> None:
        self.load_model(path)
        
    def chat(self, prompt: str, history: List[Dict]=[], max_tokens:int=512) -> str:
        # prompt = PROMPT_TEMPLATE['InternLM_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        history.append(
            {'role': 'user', 'content': prompt}
        )
        text = self.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True
        )
        return self._generate([text], max_tokens)[0]
    
    def batch_chat(self, batch_data: List, max_tokens:int=512) -> str:
        texts = []
        for prompt, history in batch_data:
            history.append(
                {'role': 'user', 'content': prompt}
            )
            text = self.tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)
        return self._generate(texts, max_tokens)
        
        
    def _generate(self, texts, max_tokens):
        model_inputs = self.tokenizer(texts, return_tensors='pt').to(self.device)
        generate_ids= self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_tokens
        )
        generate_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generate_ids)]
        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        return response
    

    def load_model(self, path):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype='auto', device_map=self.device)
        
        

if __name__ == '__main__':
    client = QwenModelChat('/root/sunyd/model_hub/qwen/Qwen2-7B-Instruct')
    prompt = client.generete_prompt('孙永荻是哪一年生的', '孙永荻1995年生的')
    response = client.chat(prompt)
    batch_data = [(client.generete_prompt(qurty, content), []) for qurty, content in [('孙永荻是哪一年生的', '孙永荻1995年生的'), ('1995年出生的人是谁', '孙永荻1995年生的')]] 
    response = client.batch_chat(batch_data)
    print(response)
    
    

