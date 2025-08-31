# llm_intent_classifier.py
import json
from typing import Dict, Any
from openai import OpenAI


class LLMIntentClassifier:
    """
    用大模型来做 summary/detail 两类分类器
    """

    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        # 系统提示词：身份设定 + 分类规则
        self.system_prompt = """你是一个问题分类器。
请根据问题的语义意图，将其分类为以下两类之一：
- summary：问题希望得到整体性的、高层次的、概览性的回答。
- detail：问题希望得到具体的、操作性的、证据性的回答。

输出要求：
- 严格输出 JSON：
  {"intent": "summary"} 或 {"intent": "detail"}
- 不要输出任何解释或其他文本。"""

        # 用户提示词模板
        self.user_prompt_template = "用户问题：{query}\n请给出分类结果。"

    def classify(self, subq: str) -> Dict[str, Any]:
        user_prompt = self.user_prompt_template.format(query=subq)

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
            # fallback：如果模型没按要求返回，简单兜底
            if "summary" in txt.lower():
                return {"intent": "summary"}
            elif "detail" in txt.lower():
                return {"intent": "detail"}
            else:
                return {"intent": "summary"}
if __name__ == "__main__":
    # 初始化分类器
    clf = LLMIntentClassifier(
        api_key="your_api_key",
        base_url="http://localhost:11434/v1"  # 如果是本地/第三方部署，改成对应的地址；用官方OpenAI则可以不传
    )

    # 示例问题
    q1 = "请概述第3章的核心思想和主要贡献"
    q2 = "算法X在3.2节是如何实现的？请给出具体步骤和关键参数"

    # 调用分类方法
    res1 = clf.classify(q1)
    res2 = clf.classify(q2)
