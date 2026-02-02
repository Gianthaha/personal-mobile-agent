"""
RAG Module - 二级检索系统

模块组成:
- api_embeddings.py: Embeddings API封装
- rag_builder.py: RAG知识库构建器
- rag_index_builder.py: FAISS向量索引构建器
- rag_query.py: RAG查询引擎

使用方法:
    from rag import (
        init_embeddings_from_config,
        init_rag_builder,
        init_index_builder,
        init_query_engine
    )
"""

import os
from typing import Dict, Optional
from langchain.embeddings.base import Embeddings

# 从各子模块导入
from .api_embeddings import (
    UnifiedEmbeddings,
    EmbeddingsAdapter,
    init_embeddings_from_config
)

from .rag_builder import (
    RAGBuilder,
    init_rag_builder_from_config
)

from .rag_index_builder import (
    RAGIndexBuilder,
    init_index_builder_from_config
)

from .rag_query import (
    RAGQueryEngine,
    init_query_engine_from_config,
    distance_to_similarity
)


# ==================== 🔥 支持自定义目录的初始化函数 ====================

def init_rag_builder(
        data_dir: str,
        embeddings: Optional[Embeddings] = None,
        api_config: Optional[Dict] = None
) -> RAGBuilder:
    """
    初始化RAG Builder（支持自定义目录）

    Args:
        data_dir: RAG数据目录
        embeddings: Embeddings对象
        api_config: API配置

    Returns:
        RAGBuilder实例
    """
    return RAGBuilder(
        data_dir=data_dir,
        embeddings=embeddings,
        api_config=api_config
    )


def init_index_builder(
        data_dir: str,
        embeddings: Optional[Embeddings] = None
) -> RAGIndexBuilder:
    """
    初始化RAG Index Builder（支持自定义目录）

    Args:
        data_dir: RAG数据目录
        embeddings: Embeddings对象

    Returns:
        RAGIndexBuilder实例
    """
    return RAGIndexBuilder(
        data_dir=data_dir,
        index_dir=os.path.join(data_dir, "indices"),
        embeddings=embeddings
    )


def init_query_engine(
        data_dir: str,
        embeddings: Optional[Embeddings] = None
) -> RAGQueryEngine:
    """
    初始化RAG Query Engine（支持自定义目录）

    Args:
        data_dir: RAG数据目录
        embeddings: Embeddings对象

    Returns:
        RAGQueryEngine实例
    """
    return RAGQueryEngine(
        data_dir=data_dir,
        index_dir=os.path.join(data_dir, "indices"),
        embeddings=embeddings
    )


# 定义公开接口
__all__ = [
    # Embeddings
    'UnifiedEmbeddings',
    'EmbeddingsAdapter',
    'init_embeddings_from_config',

    # Builder (支持自定义目录)
    'RAGBuilder',
    'init_rag_builder',
    'init_rag_builder_from_config',

    # Index Builder (支持自定义目录)
    'RAGIndexBuilder',
    'init_index_builder',
    'init_index_builder_from_config',

    # Query Engine (支持自定义目录)
    'RAGQueryEngine',
    'init_query_engine',
    'init_query_engine_from_config',
    'distance_to_similarity',
]

# 版本信息
__version__ = "1.0.0"
