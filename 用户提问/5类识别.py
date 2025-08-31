# -*- coding: utf-8 -*-
import json
from typing import Dict, Any
from openai import OpenAI


class QuestionTypeClassifier:
    """
    使用大模型识别问题属于 single-hop 还是 multi-hop，
    如果是 multi-hop，还要输出具体类别（bridge/comparison/aggregation/constraint/hybrid）。
    """

    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        # 系统提示词（带详细定义+特征+例子）
        self.system_prompt = """你是一个问题分类器，任务是识别用户输入的问题属于以下类型。

【问题类型】

1. single-hop（单跳问题）
定义：只需一次检索即可回答的问题。
特征：问题明确、直接，不需要推理链条。
例子：
- “爱因斯坦的国籍是什么？”
- “相对论的定义是什么？”

2. multi-hop（桥接型 bridge）
定义：答案需要经过中间实体或概念的推理链条才能得到。典型模式是 A → B → C，即先找到中间项 B，再基于 B 得到最终答案 C。
识别特征：
- 问题通常涉及两个以上实体或概念，但它们之间没有直接关系，需要中间关联。
- 常见问法：“X 的 Y 是什么？”、“与某对象相关的另一个对象的属性是什么？”
- 一般包含两步逻辑，例如：对象 → 找到对应实体 → 再查询其属性。
例子：
- “写《时间简史》的作者的国籍是什么？”（书 → 作者 → 国籍）
- “特斯拉公司创始人的出生地在哪里？”（公司 → 创始人 → 出生地）
❌ 非桥接： “爱因斯坦的国籍是什么？”（直接单跳，无中间实体）

3. multi-hop（比较型 comparison）
定义：答案需要对两个或多个对象进行比较、对照或排序。
识别特征：
- 常出现对比词：“谁更… / 哪个更… / 比较 A 和 B”。
- 涉及两个或多个对象，需要获取它们的属性或表现，再进行对照。
- 推理链：对象 A → 属性；对象 B → 属性；最后做对比。
例子：
- “牛顿和爱因斯坦谁更早提出引力理论？”
- “模型A和模型B在准确率上有什么不同？”
❌ 非比较： “牛顿提出引力理论的时间是什么？”（单对象）

4. multi-hop（聚合型 aggregation）
定义：答案需要收集或组合多个子问题的结果，再进行汇总、列表或总结。
识别特征：
- 常出现集合型问法：“有哪些 / 列出 / 总结 / 贡献 / 应用 / 优缺点”。
- 涉及一个主体，但答案是多个要点，不是单一事实。
- 推理链：多个事实/子答案 → 合并 → 完整答案。
例子：
- “爱因斯坦在物理学上的贡献有哪些？”（相对论 + 光电效应 + 布朗运动）
- “请列出三位获得诺贝尔物理学奖的美国科学家。”
❌ 非聚合： “爱因斯坦提出了相对论吗？”（单点事实）

5. multi-hop（条件型 constraint）
定义：答案必须同时满足多个约束条件，先从全集中筛选，再根据条件得到子集。
识别特征：
- 问题通常包含限制条件，如时间（某年以后）、地点（总部在某地）、属性（女性、美国籍）、数量（至少三位）。
- 逻辑模式是：全集 → 条件过滤 → 输出满足条件的子集。
- 往往涉及“满足…条件的有哪些”。
例子：
- “2000年以后创办、总部在美国的 AI 公司有哪些？”
- “获得诺贝尔文学奖的女性作家有哪些？”
❌ 非条件： “有哪些 AI 公司总部在美国？”（单条件聚合）

6. multi-hop（混合型 hybrid）
定义：问题同时包含多种推理方式（桥接 + 聚合 / 聚合 + 比较 / 桥接 + 条件等），需要组合才能回答。
识别特征：
- 问题结构复杂，往往同时包含集合、比较、桥接等模式。
- 如果拆解，会发现子问题属于不同类型。
- 最终答案需要经过“多步骤 + 多逻辑”组合。
例子：
- “比较两位诺贝尔物理学奖获得者在量子力学领域的贡献。”
  - 聚合：收集每位获奖者的贡献
  - 比较：再对比两者差异
- “某公司创始人和微软的创始人，谁在AI领域发表的论文更多？”
  - 桥接：找到两家公司的创始人
  - 聚合：收集他们在 AI 领域的论文
  - 比较：谁更多
❌ 非混合： “列出诺贝尔物理学奖得主的贡献。”（纯聚合）

【输出要求】
- 严格输出 JSON 格式：
  {
    "level": "single-hop" 或 "multi-hop",
    "category": null 或 "bridge/comparison/aggregation/constraint/hybrid"
  }
- 如果是单跳，category 填 null。
- 不要输出任何解释或多余文本。
"""

        # 用户提示词模板
        self.user_template = "请对以下问题进行分类：\n{query}"

    def classify(self, query: str) -> Dict[str, Any]:
        """调用大模型进行分类"""
        user_prompt = self.user_template.format(query=query)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

        txt = resp.choices[0].message.content.strip()
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            # 兜底策略
            if "multi" in txt.lower():
                for cat in ["bridge", "comparison", "aggregation", "constraint", "hybrid"]:
                    if cat in txt.lower():
                        return {"level": "multi-hop", "category": cat}
                return {"level": "multi-hop", "category": "hybrid"}
            else:
                return {"level": "single-hop", "category": None}


if __name__ == "__main__":
    clf = QuestionTypeClassifier(api_key="your_api_key", base_url="http://localhost:11434/v1")

    questions = [
        "爱因斯坦的国籍是什么？",
        "写《时间简史》的作者的国籍是什么？",
        "牛顿和爱因斯坦谁更早提出引力理论？",
        "爱因斯坦在物理学上的贡献有哪些？",
        "2000年以后创办、总部在美国的 AI 公司有哪些？",
        "比较两位诺贝尔物理学奖获得者在量子力学领域的贡献"
    ]

    for q in questions:
        result = clf.classify(q)
        print(q, "=>", result)
