# Hands on TinyRAG

## 什么是RAG？

LLM会产生误导性的 “幻觉”，依赖的信息可能过时，处理特定知识时效率不高，缺乏专业领域的深度洞察，同时在推理能力上也有所欠缺。

正是在这样的背景下，检索增强生成技术（Retrieval-Augmented Generation，RAG）应时而生，成为 AI 时代的一大趋势。

RAG 通过在语言模型生成答案之前，先从广泛的文档数据库中检索相关信息，然后利用这些信息来引导生成过程，极大地提升了内容的准确性和相关性。RAG 有效地缓解了幻觉问题，提高了知识更新的速度，并增强了内容生成的可追溯性，使得大型语言模型在实际应用中变得更加实用和可信。


以下为笔者所构思的RAG实现过程，这里面主要包括包括三个基本步骤：

1. 索引 — 将文档库分割成较短的 Chunk，并通过编码器构建向量索引。

2. 检索 — 根据问题和 chunks 的相似度检索相关文档片段。

3. 生成 — 以检索到的上下文为条件，生成问题的回答。



# 参考文献

| Name                                                         | Paper Link                                |
| ------------------------------------------------------------ | ----------------------------------------- |
| When Large Language Models Meet Vector Databases: A Survey   | [paper](http://arxiv.org/abs/2402.01763)  |
| Retrieval-Augmented Generation for Large Language Models: A Survey | [paper](https://arxiv.org/abs/2312.10997) |
| Learning to Filter Context for Retrieval-Augmented Generation | [paper](http://arxiv.org/abs/2311.08377)  |
| In-Context Retrieval-Augmented Language Models               | [paper](https://arxiv.org/abs/2302.00083) |
