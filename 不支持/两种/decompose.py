import uuid
from typing import List, Dict
from openai import OpenAI

# ========= 初始化 =========
client = OpenAI(api_key="your_api_key", base_url="http://localhost:11434/v1")

class Retriever:
    """简化版检索器接口，可以换成 Milvus / Elastic / FAISS"""
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        # ⚠️ 这里模拟检索结果，你可以替换成真实的向量/稀疏检索
        return [{"text": f"检索结果({query})-1"}, {"text": f"检索结果({query})-2"}]

# ========= Decompose RAG 核心类 =========

class DecomposeRAG:
    def __init__(self, retriever: Retriever, model="gpt-4o-mini"):
        self.retriever = retriever
        self.model = model

    def decompose(self, query: str) -> List[str]:
        """调用 LLM，把复杂问题分解为多个子问题"""
        prompt = f"""
        你是一个任务分解助手。请把以下复杂问题拆分为若干个可以单独检索的子问题：
        问题：{query}

        输出格式：一行一个子问题，不要解释。
        """
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = resp.choices[0].message.content.strip()
        subqs = [line.strip("- ").strip() for line in content.split("\n") if line.strip()]
        return subqs

    def retrieve_parallel(self, sub_questions: List[str]) -> Dict[str, List[Dict]]:
        """对每个子问题并行检索（这里用简单 for，可替换 async 并发）"""
        results = {}
        for sq in sub_questions:
            docs = self.retriever.search(sq, top_k=3)
            results[sq] = docs
        return results

    def aggregate(self, query: str, sub_results: Dict[str, List[Dict]]) -> str:
        """调用 LLM 聚合多个子问题的结果，生成最终答案"""
        context = ""
        for sq, docs in sub_results.items():
            joined = " | ".join(d["text"] for d in docs)
            context += f"\n子问题: {sq}\n检索结果: {joined}\n"

        prompt = f"""
        原始问题：{query}
        我们已经将问题拆成子问题，并收集到如下检索结果：
        {context}

        请基于这些信息，整合出最终答案。
        """
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()

    def answer(self, query: str) -> str:
        """完整流程：任务分解 → 检索 → 聚合"""
        subqs = self.decompose(query)
        print("子问题分解：", subqs)

        sub_results = self.retrieve_parallel(subqs)
        final_answer = self.aggregate(query, sub_results)
        return final_answer

# ========= 示例运行 =========

if __name__ == "__main__":
    retriever = Retriever()
    rag = DecomposeRAG(retriever)

    query = "乔布斯和马斯克谁更早出生？"
    answer = rag.answer(query)

    print("\n最终答案：", answer)
