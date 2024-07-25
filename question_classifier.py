
# -*- coding: utf-8 -*-
'''
@File    :   question_classifier.py
@Time    :   2024/07/25 16:56:00
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''


class QuestionClassifier:
    def __init__(self) -> None:
        self.intention_reg = {
            "legal_articles": ["法律","法条","判罚","罪名","刑期","本院认为","观点","条文","条款","案由","定罪","量刑","法院观点","法官意见","条例","规定","执法依据","法规","罪责","量刑","诉讼","立法"],
            "legal_books": ["依据","法律读物","法学著作","法律参考书","法典","法规","参考书","读本","丛书","法理","法学","法哲学","法学家","著作","文献","学说","学术"],
            "legal_templates": ["文书","起诉书","法律文书","起诉状","判决书","裁定书","答辩状","法律合同","协议","证据","证明","合同","格式","模板","样本","规范","范本"],
            "legal_cases": ["法律","判罚","事实","案例","罪名","刑期","本院认为","观点","法律案件","典型案例","案情","案由","定罪","量刑","证据","判例","裁决","仲裁","先例","判决","司法"],
            "JudicialExamination": ["选项","选择","A,B,C,D","ABCD","A,B,C和D","考试","题目","法考","法律考试","考题","选择题","判断题","多选题","单选题","填空题","辨析题","案例分析题","答案","试题","试卷","法学","考研","司法考试","律师考试"]
            }

    
    '''分类主函数'''
    def classify(self, question):
        data = {}
        kg_names = self.key_words_match_intention(question)
        data['kg_names'] = kg_names
        return data
    
    def key_words_match_intention(self, input):
        kg_names = set()
        for key,val in self.intention_reg.items():
            for el in val:
                if el in input:
                    kg_names.add(key)
        return kg_names