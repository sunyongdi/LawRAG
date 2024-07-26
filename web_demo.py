import json
import torch
import streamlit as st

from RAG import *
from question_classifier import QuestionClassifier
st.set_page_config(page_title="LawLLM")
st.title("LawBot🤖️")


@st.cache_resource
def init_model():
    llm = QwenModelChat('/root/sunyd/model_hub/qwen/Qwen2-7B-Instruct')
    embedding = BgeEmbedding('/root/sunyd/model_hub/ZhipuAI/bge-large-zh-v1___5/')
    vector = VectorStore(uri='/root/sunyd/llms/TinyRAG-master/storage/milvus_law.db')
    reranker = BgeReranker(path='/root/sunyd/model_hub/Xorbits/bge-reranker-base')
    classifier = QuestionClassifier()
    return llm, embedding, vector, reranker, classifier


def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("您好，我是小荻，很高兴为您服务💖")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = "🙋‍♂️" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    answer = '您好，我是小荻。一个法律AI私人助手，请问有什么需要咨询的！'
    llm, embedding, vector, reranker, classifier = init_model()
    messages = init_chat_history()
    if question := st.chat_input("Shift + Enter 换行，Enter 发送"):
        with st.chat_message("user", avatar="🙋‍♂️"):
            st.markdown(question)
        res_classify = classifier.classify(question)
        if len(res_classify['kg_names'])==0:
            prompt = llm.generete_prompt(question, answer)
            answer = llm.chat(prompt)
        else:
            contents = []
            sim_query = []
            for collection_name in res_classify['kg_names']:
                for content in vector.query(question, collection_name=collection_name, EmbeddingModel=embedding, k=3):
                    sim_query.append(content.key)
                    contents.append(content.value)
            if len(contents) == 0:
                prompt = llm.generete_prompt(question, answer)
                answer = llm.chat(prompt)
            else:
                best_content = "参考资料："
                for sq in contents:
                    best_content += f'\n\n- {sq}'
                prompt = llm.generete_prompt(question, best_content)
                answer = llm.chat(prompt)
                with st.chat_message("assistant", avatar="🤖"):
                    placeholder = st.empty()
                    placeholder.markdown(best_content) 

        messages.append({"role": "user", "content": question})
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            placeholder.markdown(answer)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": answer})
        print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()