import requests
from typing import List
from langchain.embeddings.base import Embeddings

class UnifiedEmbeddings:
    """统一的Embeddings实现,支持多种API Provider"""

    def __init__(
            self,
            provider: str = "siliconflow",
            api_key: str = None,
            api_base: str = None,
            model: str = None
    ):
        """
        初始化Embeddings

        Args:
            provider: API提供商 (siliconflow, openai, etc.)
            api_key: API密钥
            api_base: API base URL
            model: 模型名称
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.api_base = api_base
        self.model = model

        # 默认配置
        if self.provider == "siliconflow":
            self.api_base = self.api_base or "https://api.siliconflow.cn/v1"
            self.model = self.model or "BAAI/bge-large-zh-v1.5"
        elif self.provider == "openai":
            self.api_base = self.api_base or "https://api.openai.com/v1"
            self.model = self.model or "text-embedding-ada-002"

        if not self.api_key:
            raise ValueError(f"API key required for provider: {self.provider}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文档embeddings

        Args:
            texts: 文本列表

        Returns:
            embeddings列表
        """
        return self._call_embedding_api(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        生成单条查询embedding

        Args:
            text: 查询文本

        Returns:
            embedding向量
        """
        return self._call_embedding_api([text])[0]

    def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """
        调用embedding API

        Args:
            texts: 文本列表

        Returns:
            embeddings列表
        """
        url = f"{self.api_base}/embeddings"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "input": texts
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()

            # 提取embeddings
            embeddings = []
            for item in result["data"]:
                embeddings.append(item["embedding"])

            return embeddings

        except Exception as e:
            print(f"Embedding API error: {e}")
            raise


class EmbeddingsAdapter(Embeddings):
    """适配器：将UnifiedEmbeddings包装为LangChain兼容的Embeddings"""

    def __init__(self, unified_embeddings: UnifiedEmbeddings):
        self.unified_embeddings = unified_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档embedding"""
        return self.unified_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """生成单条查询embedding"""
        return self.unified_embeddings.embed_query(text)


def init_embeddings_from_config() -> Embeddings:
    """从Config初始化Embeddings"""


    unified_emb = UnifiedEmbeddings(
        provider="siliconflow",
        api_key="sk-adrknricirdyvtwkdbjqsftyllokwwccckvktypmrjjfoxgq",
        api_base= "https://api.siliconflow.cn/v1/",
        model= "Qwen/Qwen3-Embedding-8B"
    )

    return EmbeddingsAdapter(unified_emb)