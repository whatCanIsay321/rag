from typing import List, Dict, Any
from retrievers.milvus_hybrid import MilvusHybridRetriever


class TreeRetriever:
    """
    支持两种检索模式: summary / detail
    基于同一个 MilvusHybridRetriever, 通过 filter 参数控制层级
    """

    def __init__(self, uri: str, token: str, collection_name: str, top_k: int = 5):
        self.retriever = MilvusHybridRetriever(uri, token, collection_name, top_k)

    def search_summary(self, query_vec: List[float] = None, query_text: str = None,
                       mode: str = "dense", **kwargs) -> List[Dict[str, Any]]:
        """
        检索 summary 节点
        :param mode: "dense" | "bm25" | "hybrid"
        """
        if mode == "dense":
            return self.retriever.dense_search(query_vec, **kwargs)
        elif mode == "bm25":
            return self.retriever.bm25_search(query_text, **kwargs)
        elif mode == "hybrid":
            return self.retriever.hybrid_search(query_vec, query_text, **kwargs)
        else:
            raise ValueError("mode 必须是 dense | bm25 | hybrid")

    def search_detail(self, query_vec: List[float] = None, query_text: str = None,
                      mode: str = "dense", **kwargs) -> List[Dict[str, Any]]:
        """
        检索 detail 节点
        :param mode: "dense" | "bm25" | "hybrid"
        """
        if mode == "dense":
            return self.retriever.dense_search(query_vec, **kwargs)
        elif mode == "bm25":
            return self.retriever.bm25_search(query_text, **kwargs)
        elif mode == "hybrid":
            return self.retriever.hybrid_search(query_vec, query_text, **kwargs)
        else:
            raise ValueError("mode 必须是 dense | bm25 | hybrid")
