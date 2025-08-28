from typing import List
from dataclasses import dataclass


# ===== 子问题数据结构 =====
@dataclass
class SubQuery:
    query: str
    mode: str  # "summary" or "detail"


# ===== Mock 向量数据库检索接口 =====
def vector_search(query: str, mode: str, top_k: int = 3) -> List[str]:
    """
    模拟向量数据库检索，返回 chunks
    """
    return [f"[{mode.upper()}-CHUNK] 与 “{query}” 相关的文档片段 {i}" for i in range(1, top_k + 1)]


# ===== Decomposition-based RAG =====
class DecompositionRAG:
    def __init__(self, llm):
        self.llm = llm

    def decompose(self, question: str) -> List[SubQuery]:
        """
        Step1: 拆分复杂问题为子问题，并指定 summary/detail 模式
        """
        prompt = f"请将复杂问题拆分成若干子问题，并为每个子问题指定检索模式(summary 或 detail)。\n问题: {question}\n格式: 子问题 | 模式"
        raw = self.llm.generate(prompt)

        sub_queries: List[SubQuery] = []
        for line in raw.split("\n"):
            if "|" in line:
                q, mode = line.split("|")
                sub_queries.append(SubQuery(q.strip(), mode.strip()))
        return sub_queries

    def retrieve_and_answer(self, sub_query: SubQuery) -> str:
        """
        Step2 + Step3: 检索 chunks，并让 LLM 基于 chunks 回答子问题
        """
        chunks = vector_search(sub_query.query, sub_query.mode)
        evidence_text = "\n".join(chunks)

        prompt = f"子问题: {sub_query.query}\n相关文档:\n{evidence_text}\n请基于这些文档简洁回答该子问题："
        answer = self.llm.generate(prompt)
        return answer

    def answer(self, question: str) -> str:
        """
        总流程：拆分 → 检索 → 子问题回答 → 汇总
        """
        # Step1: 拆分
        sub_queries = self.decompose(question)
        print(f"[Decomposition] 子问题: {sub_queries}")

        # Step2 + Step3: 针对子问题检索 + 回答
        sub_answers = []
        for sq in sub_queries:
            ans = self.retrieve_and_answer(sq)
            sub_answers.append(f"{sq.query} → {ans}")

        # Step4: 汇总
        summary_prompt = f"原问题: {question}\n子问题回答:\n" + "\n".join(sub_answers) + "\n请综合生成最终答案："
        final_answer = self.llm.generate(summary_prompt)
        return final_answer


# ===== Mock LLM =====
class MockLLM:
    def generate(self, prompt: str) -> str:
        if "拆分" in prompt:
            return """乔布斯创办了哪家公司？ | summary
Apple 收购了哪家公司开发 iPod 的硬盘？ | detail"""
        elif "简洁回答" in prompt:
            if "乔布斯创办了哪家公司" in prompt:
                return "乔布斯创办了 Apple 公司。"
            elif "Apple 收购了哪家公司" in prompt:
                return "Apple 收购了 PortalPlayer，这家公司开发了 iPod 的硬盘。"
        else:
            return "乔布斯创办了 Apple 公司，并且 Apple 后来收购了 PortalPlayer，这家公司开发了 iPod 的硬盘。"


# ===== 测试运行 =====
if __name__ == "__main__":
    llm = MockLLM()
    rag = DecompositionRAG(llm)

    result = rag.answer("乔布斯创办的公司后来收购了哪家公司开发了 iPod 的硬盘？")
    print("\n[最终答案] ", result)
