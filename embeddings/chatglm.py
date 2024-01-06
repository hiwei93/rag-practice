from typing import List

import zhipuai
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel


class ChatGLMEmbeddings(BaseModel, Embeddings):
    def _embed_text(self, text) -> List[float]:
        response = zhipuai.model_api.invoke(
            model="text_embedding",
            prompt=text,
        )
        data = response.get('data', {})
        return data.get('embedding', [])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            text = text.replace("\n", " ")
            results.append(self._embed_text(text))
        return results

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)
