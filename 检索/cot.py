from typing import List
from dataclasses import dataclass

# ===== 子问题数据结构 =====
@dataclass
class SubQuery:
    query: str
    mode: str  # "summary" or "detail"


# ===== Mock 向量数据库检索接口 =====
def vector_search(query: str, mode: str, top_k: int = 3) -> List[str]:
    return [f"[{mode.upper()}-CHUNK] 与 “{query}” 相关的文档片段 {i}" for i in range(1, top_k + 1)]


# ===== CoT-based RAG =====
class CoTRAG:
    def __init__(self, llm):
        self.llm = llm

    def generate_chain(self, question: str) -> List[SubQuery]:
        """
        Step1: 生成推理链（子问题 + 模式）
        """
        prompt = f"请逐步推理回答问题，每一步生成子问题，并指定检索模式(summary 或 detail)。\n问题: {question}\n格式: 子问题 | 模式"
        raw = self.llm.generate(prompt)

        steps: List[SubQuery] = []
        for line in raw.split("\n"):
            if "|" in line:
                q, mode = line.split("|")
                steps.append(SubQuery(q.strip(), mode.strip()))
        return steps

    def refine_query(self, prev_answer: str, current_query: str) -> str:
        """
        Step2: 用上一步答案改写当前子问题，使其更明确
        """
        prompt = f"已知上一步答案是“{prev_answer}”，请将当前问题“{current_query}”改写为更明确的查询："
        return self.llm.generate(prompt)

    def retrieve_and_answer(self, sub_query: SubQuery) -> str:
        """
        Step3: 检索 chunks → 用 LLM 回答子问题
        """
        chunks = vector_search(sub_query.query, sub_query.mode)
        evidence_text = "\n".join(chunks)

        prompt = f"子问题: {sub_query.query}\n相关文档:\n{evidence_text}\n请基于这些文档简洁回答该子问题："
        return self.llm.generate(prompt)

    def answer(self, question: str) -> str:
        """
        总流程：生成推理链 → 改写子问题 → 检索并回答 → 汇总
        """
        # Step1: 生成推理链
        chain = self.generate_chain(question)
        print(f"[CoT] 初始推理链: {chain}")

        step_answers = []
        for i, sq in enumerate(chain):
            # Step2: 如果有上一步答案，改写当前子问题
            if i > 0:
                refined_query = self.refine_query(step_answers[-1], sq.query)
                sq.query = refined_query
                print(f"[Refine] 改写后子问题: {sq.query}")

            # Step3: 检索并回答
            ans = self.retrieve_and_answer(sq)
            step_answers.append(ans)
            print(f"[Step {i+1}] {sq.query} → {ans}")

        # Step4: 汇总
        summary_prompt = f"原问题: {question}\n子问题回答:\n" + "\n".join(step_answers) + "\n请综合生成最终答案："
        final_answer = self.llm.generate(summary_prompt)
        return final_answer


# ===== Mock LLM =====
class MockLLM:
    def generate(self, prompt: str) -> str:
        if "逐步推理" in prompt:
            return """谁发现了 DNA 双螺旋结构？ | summary
他的导师是谁？ | detail
导师在哪所大学任职？ | summary"""
        elif "改写" in prompt:
            if "他的导师是谁" in prompt:
                return "James Watson 的导师是谁？"
            elif "导师在哪所大学" in prompt:
                return "Salvador Luria 在哪所大学任职？"
            else:
                return prompt
        elif "简洁回答" in prompt:
            if "DNA 双螺旋结构" in prompt:
                return "DNA 双螺旋结构由 James Watson 和 Francis Crick 发现。"
            elif "James Watson 的导师" in prompt:
                return "James Watson 的导师是 Salvador Luria。"
            elif "Salvador Luria 在哪所大学" in prompt:
                return "Salvador Luria 任职于 MIT。"
        else:
            return "James Watson 发现了 DNA 双螺旋结构，他的导师 Salvador Luria 任职于 MIT。"


# ===== 测试运行 =====
if __name__ == "__main__":
    llm = MockLLM()
    rag = CoTRAG(llm)

    result = rag.answer("哪位科学家发现了 DNA 双螺旋结构，他的导师在哪所大学任职？")
    print("\n[最终答案] ", result)
