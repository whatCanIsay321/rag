# -*- coding: utf-8 -*-
# tree_search_planner.py
from typing import List, Dict, Any, Optional


class TreeAwareRetriever:
    """
    抽象接口（请用你的 Milvus 检索实现掉这3个方法）：
      - search_summary(query_text, book_id, top_k=5, extra_filters=None) -> List[Dict]
      - search_detail_under(query_text, book_id, parent_ids, top_k=8) -> List[Dict]
      - search_detail_global(query_text, book_id, top_k=8) -> List[Dict]

    命中建议包含字段：
      id, parent_id(叶子), section_path(可选), text, score
    """
    def search_summary(self, query_text: str, book_id: str, top_k: int = 5,
                       extra_filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def search_detail_under(self, query_text: str, book_id: str, parent_ids: List[str],
                            top_k: int = 8) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def search_detail_global(self, query_text: str, book_id: str, top_k: int = 8
                             ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class TreeSearchPlanner:
    """
    动态反馈流程：
      - 对于 intent=summary：先 summary → 判断强弱 → 下钻 detail / 扩扇出 / 全局兜底
      - 对于 intent=detail：若有章节线索先锁定子树，否则直接全局 detail；弱命中再借 summary
    """

    def __init__(self, retriever: TreeAwareRetriever,
                 min_summary_score: float = 0.15,
                 parent_fanout_max: int = 3,
                 detail_topk: int = 8):
        self.ret = retriever
        self.min_summary_score = min_summary_score
        self.parent_fanout_max = parent_fanout_max
        self.detail_topk = detail_topk

    def run(self, subq: str, book_id: str, intent: str,
            section_hints: List[str]) -> Dict[str, Any]:
        """
        返回：
        {
          "mode": "<summary→detail | detail-only | detail@hint-subtree | ...>",
          "summary_hits": [...],
          "detail_hits": [...],
          "used_fallback": bool
        }
        """
        if intent == "summary":
            return self._summary_first(subq, book_id, section_hints)
        else:
            return self._detail_first(subq, book_id, section_hints)

    # -------- summary-first --------
    def _summary_first(self, subq: str, book_id: str, section_hints: List[str]) -> Dict[str, Any]:
        sum_hits = self.ret.search_summary(
            query_text=subq, book_id=book_id, top_k=self.parent_fanout_max,
            extra_filters={"section_hints": section_hints} if section_hints else None
        )
        if self._is_strong(sum_hits) or section_hints:
            parent_ids = [h["id"] for h in sum_hits[: self.parent_fanout_max]]
            det_hits = self.ret.search_detail_under(subq, book_id, parent_ids, top_k=self.detail_topk)
            if det_hits:
                return {"mode": "summary→detail", "summary_hits": sum_hits,
                        "detail_hits": det_hits, "used_fallback": False}
            # 扩父节点扇出再试
            if len(sum_hits) > 1:
                fan_ids = [h["id"] for h in sum_hits[: self.parent_fanout_max]]
                det2 = self.ret.search_detail_under(subq, book_id, fan_ids, top_k=max(self.detail_topk, 12))
                if det2:
                    return {"mode": "summary→detail(fanout)", "summary_hits": sum_hits,
                            "detail_hits": det2, "used_fallback": False}
        # 全局兜底
        det_global = self.ret.search_detail_global(subq, book_id, top_k=max(self.detail_topk, 12))
        return {"mode": "detail(fallback-global)", "summary_hits": sum_hits,
                "detail_hits": det_global, "used_fallback": True}

    # -------- detail-first --------
    def _detail_first(self, subq: str, book_id: str, section_hints: List[str]) -> Dict[str, Any]:
        # 有路径线索时，先用 summary 锁子树再 detail
        if section_hints:
            sum_hits = self.ret.search_summary(
                query_text=subq, book_id=book_id, top_k=self.parent_fanout_max,
                extra_filters={"section_hints": section_hints}
            )
            parent_ids = [h["id"] for h in sum_hits[: self.parent_fanout_max]] if sum_hits else []
            if parent_ids:
                det_hits = self.ret.search_detail_under(subq, book_id, parent_ids, top_k=self.detail_topk)
                if det_hits:
                    return {"mode": "detail@hint-subtree", "summary_hits": sum_hits,
                            "detail_hits": det_hits, "used_fallback": False}
        # 直接全局 detail
        det_hits = self.ret.search_detail_global(subq, book_id, top_k=self.detail_topk)
        if det_hits:
            return {"mode": "detail-only", "summary_hits": [], "detail_hits": det_hits, "used_fallback": False}
        # 再借 summary 导航
        sum_hits = self.ret.search_summary(subq, book_id, top_k=self.parent_fanout_max)
        parent_ids = [h["id"] for h in sum_hits[: self.parent_fanout_max]] if sum_hits else []
        det2 = self.ret.search_detail_under(subq, book_id, parent_ids, top_k=max(self.detail_topk, 12))
        return {"mode": "summary→detail(recovery)", "summary_hits": sum_hits,
                "detail_hits": det2, "used_fallback": True}

    def _is_strong(self, hits: List[Dict[str, Any]]) -> bool:
        return bool(hits) and hits[0].get("score", 0.0) >= self.min_summary_score
