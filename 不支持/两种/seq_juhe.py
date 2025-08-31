from typing import List, Dict
from openai import OpenAI

client = OpenAI(api_key="your_api_key", base_url="http://localhost:11434/v1")

class SequentialContextRAG:
    """
    不改写 query，只累积 context 的 Sequential RAG
    适合 Aggregation / Exploratory 型问题
    """
    def __init__(self, retriever, model="gpt-4o-mini"):
        self.retriever = retriever
        self.model = model

    def answer(self, query: str, steps: int = 3) -> str:
        """执行顺序检索"""
        context = ""
        results = []

        for i in range(steps):
            # 检索时把已有 context 加进去
            docs = self.retriever.search(query + " " + context, top_k=3)
            joined = " | ".join(d["text"] for d in docs)
            results.append((f"Step{i+1}", joined))
            context += " " + joined  # 累积上下文

        # 聚合
        agg_prompt = f"""
        原始问题：{query}
        我们逐步积累的上下文是：
        {context}

        请整合这些信息，生成最终答案。
        """
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": agg_prompt}],
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
