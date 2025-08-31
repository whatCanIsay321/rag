# -*- coding: utf-8 -*-
import json
from typing import Dict, Any
from openai import OpenAI


class DecisionRouter:
    """
    根据分类器输出 (level + category)，自动选择对应的检索策略。
    内部集成了聚合型 (aggregation) 的二级分类 (quantitative / open)。
    """

    def __init__(self, retrievers: Dict[str, Any],
                 api_key: str,
                 base_url: str = None,
                 model: str = "deepseek-chat"):
        self.retrievers = retrievers
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        # 子分类提示词（system 部分，固定规则）
        self.agg_system_prompt = """你是一个子分类器，任务是进一步区分聚合型 (aggregation) 问题的两种模式：

1. quantitative（定量聚合）：问题要求一个确定数量的答案，或明确要求“列出 N 个”、“前 N 个”、“所有符合条件的”。
2. open（开放聚合）：问题要求尽可能多的答案，但没有固定数量限制，通常问“有哪些”、“贡献是什么”、“应用场景有哪些”。

【输出要求】
严格输出 JSON：
{"aggregation_type": "quantitative"} 或 {"aggregation_type": "open"}
"""

    def _classify_aggregation(self, query: str) -> str:
        """调用 LLM，区分 quantitative / open"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.agg_system_prompt},
                {"role": "user", "content": f"待分类问题：{query}"}
            ],
            temperature=0
        )
        txt = resp.choices[0].message.content.strip()
        try:
            return json.loads(txt)["aggregation_type"]
        except Exception:
            if "quant" in txt.lower():
                return "quantitative"
            return "open"

    def route(self, classification: Dict[str, Any], query: str):
        """
        输入分类结果，返回对应的检索器
        classification = {
            "level": "single-hop" 或 "multi-hop",
            "category": None 或 "bridge/comparison/aggregation/constraint/hybrid"
        }
        """
        level = classification.get("level")
        category = classification.get("category")

        if level == "single-hop":
            return self.retrievers["sequential_context"]

        if category == "bridge":
            return self.retrievers["sequential_bridge"]

        elif category == "comparison":
            return self.retrievers["decompose"]

        elif category == "aggregation":
            agg_type = self._classify_aggregation(query)
            if agg_type == "quantitative":
                return self.retrievers["decompose"]
            else:
                return self.retrievers["sequential_context"]

        elif category == "constraint":
            return self.retrievers["decompose_filter"]

        elif category == "hybrid":
            return self.retrievers["decompose"]

        else:
            return self.retrievers["sequential_context"]



if __name__ == "__main__":
    class MockRetriever:
        def __init__(self, name): self.name = name

        def search(self, q): return f"[{self.name}] 检索: {q}"


    retrievers = {
        "decompose": MockRetriever("Decompose RAG"),
        "sequential_bridge": MockRetriever("Sequential Bridge RAG"),
        "sequential_context": MockRetriever("Sequential Context RAG"),
        "decompose_filter": MockRetriever("Decompose+Filter RAG"),
    }

    router = DecisionRouter(retrievers, api_key="sk-17a8b42bfb644940807225f61811f750", base_url="https://api.deepseek.com/v1",model="deepseek-chat")

    # 示例分类结果（假设是从大模型主分类器拿到的）
    cases = [
        {"level": "single-hop", "category": None},
        {"level": "multi-hop", "category": "bridge"},
        {"level": "multi-hop", "category": "comparison"},
        {"level": "multi-hop", "category": "aggregation"},
        {"level": "multi-hop", "category": "constraint"},
        {"level": "multi-hop", "category": "hybrid"},
    ]

    for case in cases:
        retriever = router.route(case, "爱因斯坦在物理学上的贡献有哪些？")
        print(case, "=>", retriever.search("示例问题"))