from typing import List, Dict
from openai import OpenAI

client = OpenAI(api_key="your_api_key", base_url="http://localhost:11434/v1")


class SequentialRewriteRAG:
    """
    带 query 改写的 Sequential RAG
    适合 Bridge 型问题 (A → B → C)
    """
    def __init__(self, retriever, model="gpt-4o-mini"):
        self.retriever = retriever
        self.model = model

    def next_query(self, query: str, context: str) -> str:
        """调用 LLM 改写 query，基于已有上下文"""
        prompt = f"""
        原始问题：{query}
        已知信息：{context}

        请生成下一个需要检索的子问题。
        只输出子问题本身，不要解释。
        """
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()

    def answer(self, query: str, steps: int = 2) -> str:
        """执行顺序检索"""
        context = ""
        results = []

        for i in range(steps):
            subq = self.next_query(query, context)
            docs = self.retriever.search(subq, top_k=3)
            joined = " | ".join(d["text"] for d in docs)
            results.append((subq, joined))
            context += f"\n{subq}: {joined}"

        # 聚合
        agg_prompt = f"""
        原始问题：{query}
        我们依次检索到以下信息：
        {context}

        请基于这些信息，生成最终答案。
        """
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": agg_prompt}],
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
