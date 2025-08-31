# -*- coding: utf-8 -*-
# coverage_assessor.py
import re
from typing import List, Dict, Any, Set


class CoverageAssessor:
    """
    覆盖/质量后处理：
      - enough(): 证据点是否“够用”（去重后数量阈值）
      - dedup(): 去重
      - expand_terms(): 从已命中 detail 抽关键词，用于“二次扩展检索”
    """

    def __init__(self, min_unique_points: int = 3):
        self.min_unique_points = min_unique_points

    def enough(self, detail_hits: List[Dict[str, Any]]) -> bool:
        uniq = {(h.get("parent_id"), h.get("id")) for h in detail_hits}
        return len(uniq) >= min(self.min_unique_points, len(detail_hits))

    def dedup(self, detail_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Set[tuple] = set()
        out: List[Dict[str, Any]] = []
        for h in detail_hits:
            key = (h.get("parent_id"), h.get("id"))
            if key in seen:
                continue
            seen.add(key)
            out.append(h)
        return out

    def expand_terms(self, detail_hits: List[Dict[str, Any]], max_terms: int = 6) -> List[str]:
        texts = [h.get("text", "") for h in detail_hits[: min(10, len(detail_hits))]]
        candidates: Dict[str, int] = {}
        for t in texts:
            for w in re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,}", t):
                candidates[w] = candidates.get(w, 0) + 1
        terms = sorted(candidates.items(), key=lambda x: (-x[1], -len(x[0])))[:max_terms]
        return [t for t, _ in terms]
