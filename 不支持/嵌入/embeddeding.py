from openai import OpenAI

class OpenAIEmbedding:
    def __init__(self, api_base: str, api_key: str, model_name: str):
        self.client = OpenAI(
            base_url=api_base.rstrip("/"),
            api_key=api_key,
        )
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入多个文档"""
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """嵌入单个查询"""
        return self.embed_documents([text])[0]
