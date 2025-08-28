from pymilvus import MilvusClient
from typing import List, Dict, Any


class MilvusHybridRetriever:
    def __init__(self, uri: str, token: str, collection_name: str, top_k: int = 5):
        """
        Milvus 混合检索器 (Dense + BM25 + RRF/WeightedRank)
        :param uri: Milvus 服务地址 (http://localhost:19530)
        :param token: 认证信息 (root:Milvus)
        :param collection_name: 集合名
        :param top_k: 默认返回结果数
        """
        self.client = MilvusClient(uri=uri, token=token)
        self.collection_name = collection_name
        self.top_k = top_k

    # ===== Dense 向量检索 =====
    def dense_search(self, query_vec: List[float], top_k: int = None) -> List[Dict[str, Any]]:
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vec],
            limit=top_k or self.top_k,
            anns_field="embedding",
            output_fields=["id", "text"]
        )
        return self._format_results(results[0])

    # ===== BM25 检索 =====
    def bm25_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            data=[{"text": query}],   # Sparse 查询
            limit=top_k or self.top_k,
            sparse_fields=["text"],   # 基于 text 字段
            ranker="bm25",            # 指定 BM25
            output_fields=["id", "text"]
        )
        return self._format_results(results[0])

    # ===== Hybrid 检索 + RRF/WeightedRanker =====
    def hybrid_search(
        self,
        query_vec: List[float],
        query_text: str,
        top_k: int = None,
        method: str = "rrf",
        weights: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        :param method: "rrf" 或 "weighted"
        :param weights: weighted 模式下的权重 { "sparse":0.3, "dense":0.7 }
        """
        params = {
            "collection_name": self.collection_name,
            "limit": top_k or self.top_k,
            "anns_field": "embedding",
            "sparse_fields": ["text"],
            "output_fields": ["id", "text"],
        }

        if method == "rrf":
            params["ranker"] = "rrf"
        elif method == "weighted":
            params["ranker"] = {"type": "weighted", "weights": weights or {"dense": 0.5, "sparse": 0.5}}
        else:
            raise ValueError("method 必须是 'rrf' 或 'weighted'")

        results = self.client.hybrid_search(
            data=[{"dense": query_vec, "sparse": {"text": query_text}}],
            **params
        )
        return self._format_results(results[0])

    # ===== 结果格式化 =====
    def _format_results(self, hits) -> List[Dict[str, Any]]:
        return [
            {
                "id": h.get("id"),
                "text": h.get("entity").get("text"),
                "score": h.get("distance")
            }
            for h in hits
        ]
retriever = MilvusHybridRetriever(
    uri="http://localhost:19530",
    token="root:Milvus",
    collection_name="documents",
    top_k=5
)

# 示例 query
query_text = "合同A的违约条款"
query_vec = [0.1, 0.2, 0.3, ...]  # 维度要和 collection 一致

print("\n--- Dense ---")
print(retriever.dense_search(query_vec))

print("\n--- BM25 ---")
print(retriever.bm25_search(query_text))

print("\n--- Hybrid RRF ---")
print(retriever.hybrid_search(query_vec, query_text, method="rrf"))

print("\n--- Hybrid Weighted ---")
print(retriever.hybrid_search(query_vec, query_text, method="weighted", weights={"dense": 0.7, "sparse": 0.3}))
