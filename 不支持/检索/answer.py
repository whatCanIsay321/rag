f
class RAGAnswerModule:
    def __init__(self, retriever_summary, retriever_detail, classifier):
        self.retriever_summary = retriever_summary
        self.retriever_detail = retriever_detail
        self.classifier = classifier  # 识别任务类型 & 检索目标类型

    def answer(self, query):
        task_type = self.classifier.classify_task(query)

        if task_type == "single":
            retriever = self._choose_retriever(query)
            docs = retriever.search(query, top_k=5)
            return self._aggregate(docs)

        elif task_type == "multi-seq":
            sub_queries = self._decompose_sequential(query)
            results = []
            context = ""
            for sq in sub_queries:
                retriever = self._choose_retriever(sq)
                docs = retriever.search(sq, context=context, top_k=5)
                results.append(docs)
                context += self._summarize(docs)  # 用结果更新上下文
            return self._aggregate(results)

        elif task_type == "multi-parallel":
            sub_queries = self._decompose_parallel(query)
            results = []
            for sq in sub_queries:
                retriever = self._choose_retriever(sq)
                docs = retriever.search(sq, top_k=5)
                results.append(docs)
            return self._aggregate(results)

    def _choose_retriever(self, query):
        # 判定 summary / detail
        if self.classifier.is_summary_query(query):
            return self.retriever_summary
        else:
            return self.retriever_detail

    def _aggregate(self, docs):
        # 简单聚合，也可以加 reranker
        return docs
