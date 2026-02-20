"""
RAG Query Module - 两级检索系统 (Multi-App支持版)
Level 1: 指令 -> Category -> 候选Apps
Level 2: App -> Workflows + Preferences + Usage

修复:
1. 支持multi_app任务的检索
2. 合并多个app的RAG知识
"""

import os
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings.base import Embeddings

from rag.api import  inference_chat
from rag.config import Config


def distance_to_similarity(distance: float, metric: str = 'l2') -> float:
    """将FAISS距离转换为相似度分数 [0, 1]"""
    try:
        if metric == 'l2':
            return math.exp(-distance * distance)
        elif metric == 'cosine':
            return (1 + distance) / 2
        else:
            return 1.0 / (1.0 + float(distance))
    except Exception:
        return 0.0


class RAGQueryEngine:
    """RAG查询引擎 - 两级检索 + Multi-App支持"""

    def __init__(
            self,
            data_dir: str = "./data",
            index_dir: str = "./data/indices",
            embeddings: Optional[Embeddings] = None
    ):
        """
        初始化查询引擎

        Args:
            data_dir: 数据根目录
            index_dir: 索引目录
            embeddings: Embeddings对象
        """
        self.data_dir = Path(data_dir)
        self.level1_dir = self.data_dir / "level1"
        self.level2_dir = self.data_dir / "level2"

        self.index_dir = Path(index_dir)
        self.level1_index_dir = self.index_dir / "level1"
        self.level2_index_dir = self.index_dir / "level2"

        self.embeddings = embeddings

        # 延迟加载
        self._level1_store = None
        self._level2_store = None

    def set_embeddings(self, embeddings: Embeddings):
        """设置embeddings对象"""
        self.embeddings = embeddings

    # ==================== 工具函数：解析apps ====================

    def _parse_apps(self, apps_field: Union[str, List[str]]) -> List[str]:
        """
        解析apps字段，支持多种格式
        """
        if isinstance(apps_field, str):
            apps = [app.strip() for app in apps_field.split(",")]
        elif isinstance(apps_field, list):
            apps = []
            for item in apps_field:
                if isinstance(item, str):
                    apps.extend([app.strip() for app in item.split(",")])
                else:
                    apps.append(str(item).strip())
        else:
            apps = [str(apps_field).strip()]

        return [app for app in apps if app]

    # ==================== Level 1: Category检索 ====================

    def query_category(
            self,
            instruction: str,
            top_k: int = 3,
            min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        一级检索: 指令 -> Category

        Args:
            instruction: 用户指令
            top_k: 返回top k个结果
            min_similarity: 最低相似度阈值

        Returns:
            匹配的categories列表
        """
        if self._level1_store is None:
            if not self.level1_index_dir.exists():
                print(" Level 1 index not found")
                return []

            if not self.embeddings:
                raise ValueError("Embeddings not set!")

            try:
                self._level1_store = FAISS.load_local(
                    str(self.level1_index_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Failed to load Level 1 index: {e}")
                return []

        hits = self._level1_store.similarity_search_with_score(instruction, k=top_k)

        results = []
        for doc, distance in hits:
            sim = distance_to_similarity(distance, metric='l2')

            if sim < min_similarity:
                continue

            result = {
                "category": doc.metadata["category"],
                "apps": doc.metadata["apps"],
                "score": float(sim),
                "text": doc.page_content
            }
            results.append(result)

        return results

    # ==================== Level 2: App检索 ====================

    def query_app(
            self,
            app_name: str,
            instruction: str,
            top_k: int = 5,
            min_similarity: float = 0.5,
            doc_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        二级检索: 在指定app内检索workflows

        Args:
            app_name: 目标app名称
            instruction: 用户指令
            top_k: 返回top k个结果
            min_similarity: 最低相似度阈值
            doc_types: 限制文档类型

        Returns:
            匹配的documents列表
        """
        if self._level2_store is None:
            if not self.level2_index_dir.exists():
                print(" Level 2 index not found")
                return []

            if not self.embeddings:
                raise ValueError("Embeddings not set!")

            try:
                self._level2_store = FAISS.load_local(
                    str(self.level2_index_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Failed to load Level 2 index: {e}")
                return []

        hits = self._level2_store.similarity_search_with_score(instruction, k=top_k * 10)

        results = []
        for doc, distance in hits:
            meta = doc.metadata or {}

            if meta.get("app_name", "").lower() != app_name.lower():
                continue

            if doc_types and meta.get("doc_type") not in doc_types:
                continue

            sim = distance_to_similarity(distance, metric='l2')

            if sim < min_similarity:
                continue

            result = {
                "app_name": meta.get("app_name"),
                "doc_type": meta.get("doc_type"),
                "score": float(sim),
                "text": doc.page_content,
                "metadata": meta
            }
            results.append(result)

            if len(results) >= top_k:
                break

        return results

    def query_multiple_apps(
            self,
            app_names: List[str],
            instruction: str,
            top_k_per_app: int = 3,
            min_similarity: float = 0.5
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """
        查询多个app的知识

        Args:
            app_names: app名称列表
            instruction: 用户指令
            top_k_per_app: 每个app返回的workflow数量
            min_similarity: 最低相似度阈值

        Returns:
            {app_name: {"workflows": [...], "preferences": [...], "usage": [...]}}
        """
        results = {}

        for app_name in app_names:
            app_results = {
                "workflows": self.query_app(
                    app_name, instruction,
                    top_k=top_k_per_app,
                    min_similarity=min_similarity,
                    doc_types=["workflow"]
                ),
                "preferences": self.query_app(
                    app_name, instruction,
                    top_k=1,
                    min_similarity=0.1,
                    doc_types=["preferences"]
                ),
                "usage": self.query_app(
                    app_name, instruction,
                    top_k=1,
                    min_similarity=0.0,
                    doc_types=["usage_stats"]
                )
            }
            results[app_name] = app_results

        return results

    # ==================== 完整的两级检索 ====================

    def two_level_retrieve(
            self,
            instruction: str,
            experiences: Dict[str, str] = None,
            top_k_workflows: int = 3
    ) -> Tuple[Optional[str], str]:
        """
        完整的两级检索流程（单app）

        Args:
            instruction: 用户指令
            experiences: 经验字典
            top_k_workflows: 返回top k个workflows

        Returns:
            (selected_app, rag_knowledge)
        """
        print("\n" + "=" * 70)
        print(" Two-Level RAG Retrieval")
        print("=" * 70)

        categories = self.query_category(instruction, top_k=1)

        if not categories:
            print(" No matching category found")
            return None, ""

        category_info = categories[0]
        category = category_info["category"]
        candidate_apps = category_info["apps"]

        print(f"\n Category: {category} (score: {category_info['score']:.3f})")
        print(f"   Candidates: {candidate_apps}")

        selected_app = self._select_app_by_preference(
            category, candidate_apps, experiences, instruction
        )
        print(f" Selected: {selected_app}")

        if not selected_app:
            return None, ""

        print(f"\n Searching in {selected_app}...")

        workflows = self.query_app(
            selected_app, instruction,
            top_k=top_k_workflows,
            min_similarity=0.3,
            doc_types=["workflow"]
        )

        preferences = self.query_app(
            selected_app, instruction,
            top_k=1,
            min_similarity=0.1,
            doc_types=["preferences"]
        )

        usage = self.query_app(
            selected_app, instruction,
            top_k=1,
            min_similarity=0.5,
            doc_types=["usage_stats"]
        )

        print(f"   Workflows: {len(workflows)} found")
        print(f"   Preferences: {len(preferences)} found")
        print(f"   Usage: {len(usage)} found")

        rag_knowledge = self._format_rag_knowledge(
            selected_app, category,
            workflows,
            preferences[0] if preferences else None,
            usage[0] if usage else None
        )

        print("\n" + "=" * 70)

        return selected_app, rag_knowledge

    def two_level_retrieve_multi_app(
            self,
            instruction: str,
            apps: Union[str, List[str]],
            experiences: Dict[str, str] = None,
            top_k_workflows: int = 2
    ) -> Tuple[List[str], str]:
        """
        Multi-App的两级检索流程

        Args:
            instruction: 用户指令
            apps: app列表或逗号分隔字符串
            experiences: 经验字典
            top_k_workflows: 每个app返回的workflow数量

        Returns:
            (app_list, combined_rag_knowledge)
        """
        print("\n" + "=" * 70)
        print(" Two-Level RAG Retrieval (Multi-App)")
        print("=" * 70)

        app_list = self._parse_apps(apps)
        print(f" Apps: {app_list}")

        if not app_list:
            print(" No apps specified")
            return [], ""

        # 查询所有app的知识
        all_app_results = self.query_multiple_apps(
            app_list, instruction,
            top_k_per_app=top_k_workflows,
            min_similarity=0.5
        )

        # 格式化多app的RAG知识
        rag_knowledge = self._format_multi_app_rag_knowledge(
            app_list, all_app_results
        )

        print("\n" + "=" * 70)

        return app_list, rag_knowledge

    def _select_app_by_preference(
            self,
            category: str,
            candidate_apps: List[str],
            experiences: Dict[str, str] = None,
            instruction: str = None
    ) -> str:
        """
        根据experience和LLM判断选择最佳app
        
        Args:
            category: 应用类别
            candidate_apps: 候选app列表
            experiences: 经验字典
            instruction: 用户指令（可选，用于LLM判断）
            
        Returns:
            选中的app名称
        """
        if not candidate_apps:
            return None

        # 只有一个候选时直接返回
        if len(candidate_apps) == 1:
            return candidate_apps[0]

        # 调用LLM进行选择
        try:
            selected = self._llm_select_app(
                category=category,
                candidate_apps=candidate_apps,
                experiences=experiences,
                instruction=instruction
            )
            if selected and selected in candidate_apps:
                print(f"  LLM selected app: {selected}")
                return selected
        except Exception as e:
            print(f"  LLM selection failed: {e}, falling back to default")

        # 回退：使用原有的规则匹配逻辑
        return self._rule_based_select_app(category, candidate_apps, experiences)

    def _llm_select_app(
            self,
            category: str,
            candidate_apps: List[str],
            experiences: Dict[str, str] = None,
            instruction: str = None
    ) -> Optional[str]:
        """
        使用LLM选择最佳app
        
        Args:
            category: 应用类别
            candidate_apps: 候选app列表
            experiences: 经验字典
            instruction: 用户指令
            
        Returns:
            LLM选择的app名称
        """
        # 构建prompt
        prompt = self._build_app_selection_prompt(
            category, candidate_apps, experiences, instruction
        )
        
        # 构建消息
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent assistant that helps select the most appropriate mobile app based on user needs and historical preferences. You must respond with ONLY the app name, nothing else."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            # 使用已导入的inference_chat函数调用API
            llm_response = inference_chat(
                messages,
                model=Config.MODEL,
                api_url=Config.API_URL,
                token=Config.API_TOKEN
            )
            
            llm_response = llm_response.strip()
            
            # 解析LLM响应，提取app名称
            selected_app = self._parse_llm_app_selection(llm_response, candidate_apps)
            
            return selected_app
            
        except Exception as e:
            print(f"  LLM API call failed: {e}")
            raise

    def _build_app_selection_prompt(
            self,
            category: str,
            candidate_apps: List[str],
            experiences: Dict[str, str] = None,
            instruction: str = None
    ) -> str:
        """
        构建app选择的prompt
        
        Args:
            category: 应用类别
            candidate_apps: 候选app列表
            experiences: 经验字典
            instruction: 用户指令
            
        Returns:
            构建好的prompt
        """
        prompt_parts = []
        
        prompt_parts.append(f"Task: Select the most appropriate app from the candidates for the '{category}' category.")
        prompt_parts.append("")
        
        # 添加用户指令（如果有）
        if instruction:
            prompt_parts.append(f"User Instruction: {instruction}")
            prompt_parts.append("")
        
        # 候选app列表
        prompt_parts.append("Candidate Apps:")
        for i, app in enumerate(candidate_apps, 1):
            prompt_parts.append(f"  {i}. {app}")
        prompt_parts.append("")
        
        # 添加历史经验（如果有）
        if experiences:
            prompt_parts.append("Historical Experience/Preferences:")
            for key, content in experiences.items():
                # 筛选相关的经验
                if category.lower() in content.lower() or any(app.lower() in content.lower() for app in candidate_apps):
                    # 截取相关部分，避免prompt过长
                    relevant_content = content[:500] if len(content) > 500 else content
                    prompt_parts.append(f"  - {relevant_content}")
            prompt_parts.append("")
        
        prompt_parts.append("Based on the above information, which app would be the best choice?")
        prompt_parts.append("Please respond with ONLY the exact app name from the candidate list, nothing else.")
        
        return "\n".join(prompt_parts)

    def _parse_llm_app_selection(
            self,
            llm_response: str,
            candidate_apps: List[str]
    ) -> Optional[str]:
        """
        解析LLM响应，提取选择的app名称
        
        Args:
            llm_response: LLM的原始响应
            candidate_apps: 候选app列表
            
        Returns:
            匹配到的app名称，如果无法匹配则返回None
        """
        response_clean = llm_response.strip().strip('"\'').strip()
        
        # 精确匹配
        for app in candidate_apps:
            if response_clean.lower() == app.lower():
                return app
        
        # 包含匹配
        for app in candidate_apps:
            if app.lower() in response_clean.lower():
                return app
        
        # 模糊匹配：检查响应中是否包含app名称的关键部分
        for app in candidate_apps:
            app_keywords = app.lower().replace("_", " ").replace("-", " ").split()
            if all(kw in response_clean.lower() for kw in app_keywords):
                return app
        
        return None

    def _rule_based_select_app(
            self,
            category: str,
            candidate_apps: List[str],
            experiences: Dict[str, str] = None
    ) -> str:
        """
        基于规则的app选择（回退方法）
        
        Args:
            category: 应用类别
            candidate_apps: 候选app列表
            experiences: 经验字典
            
        Returns:
            选中的app名称
        """
        if not experiences:
            return candidate_apps[0]

        app_scores = {}
        pattern = re.compile(r'([A-Za-z0-9\u4e00-\u9fa5\-\_ ]+?)\s*\(\s*(\d{1,3})\s*%\s*\)')

        for exp_content in experiences.values():
            if f"[App Preference - {category}]" in exp_content:
                matches = pattern.findall(exp_content)
                for app_name, percentage in matches:
                    app_name_clean = app_name.strip()
                    for candidate in candidate_apps:
                        if app_name_clean.lower() in candidate.lower():
                            app_scores[candidate] = max(
                                app_scores.get(candidate, 0),
                                int(percentage)
                            )
                            break

        if app_scores:
            selected = max(app_scores, key=app_scores.get)
            print(f"  App preferences found (rule-based): {app_scores}")
            return selected

        return candidate_apps[0]

    def _format_rag_knowledge(
            self,
            app_name: str,
            category: str,
            workflows: List[Dict],
            preference_doc: Optional[Dict],
            usage_doc: Optional[Dict]
    ) -> str:
        """格式化单app的RAG知识"""
        lines = ["### RAG Knowledge ###"]
        lines.append(f"App: {app_name} (Category: {category})")

        if usage_doc:
            lines.append("\n### Usage Statistics ###")
            lines.append(usage_doc["text"])

        if preference_doc:
            lines.append("\n### User Preferences ###")
            lines.append(preference_doc["text"])

        if workflows:
            lines.append("\n### Relevant Workflows ###")
            for i, wf in enumerate(workflows, 1):
                meta = wf.get("metadata", {})

                lines.append(f"\n{i}. Workflow (score: {wf['score']:.3f})")

                task = meta.get("task", "")
                task_summary = meta.get("task_summary", "")
                if task:
                    lines.append(f"Task: {task}")
                if task_summary:
                    lines.append(f"Summary: {task_summary}")

                steps = meta.get("steps", "")
                if steps:
                    lines.append(f"\nSteps:")
                    lines.append(steps)

                ui_elements = meta.get("ui_elements", "")
                if ui_elements:
                    lines.append(f"\nUI Elements:")
                    lines.append(ui_elements)

                success_count = meta.get("success_count", 0)
                if success_count > 0:
                    lines.append(f"Success Count: {success_count}")

                lines.append("")

        lines.append("\n### Important Notes ###")
        lines.append("- Workflow coordinates are for reference only")
        lines.append("- Always use current screen detection for accurate coordinates")

        return "\n".join(lines)

    def _format_multi_app_rag_knowledge(
            self,
            app_list: List[str],
            all_app_results: Dict[str, Dict[str, List[Dict]]]
    ) -> str:
        """
        格式化Multi-App的RAG知识

        Args:
            app_list: app列表
            all_app_results: {app_name: {"workflows": [...], "preferences": [...], "usage": [...]}}

        Returns:
            合并后的RAG知识文本
        """
        lines = ["### RAG Knowledge (Multi-App Task) ###"]
        lines.append(f"Apps involved: {', '.join(app_list)}")
        lines.append("")

        for app_name in app_list:
            app_data = all_app_results.get(app_name, {})

            lines.append(f"\n{'='*50}")
            lines.append(f"📱 App: {app_name}")
            lines.append(f"{'='*50}")

            # Usage stats
            usage = app_data.get("usage", [])
            if usage:
                lines.append("\n### Usage Statistics ###")
                lines.append(usage[0]["text"])

            # Preferences
            preferences = app_data.get("preferences", [])
            if preferences:
                lines.append("\n### User Preferences ###")
                lines.append(preferences[0]["text"])

            # Workflows
            workflows = app_data.get("workflows", [])
            if workflows:
                lines.append("\n### Relevant Workflows ###")
                for i, wf in enumerate(workflows, 1):
                    meta = wf.get("metadata", {})

                    lines.append(f"\n{i}. Workflow (score: {wf['score']:.3f})")

                    task = meta.get("task", "")
                    task_summary = meta.get("task_summary", "")
                    if task:
                        lines.append(f"   Task: {task}")
                    if task_summary:
                        lines.append(f"   Summary: {task_summary}")

                    steps = meta.get("steps", "")
                    if steps:
                        lines.append(f"   Steps: {steps[:200]}...")

            lines.append("")

        lines.append("\n### Multi-App Task Notes ###")
        lines.append("- This task involves multiple apps")
        lines.append("- Complete operations in each app sequentially")
        lines.append("- Use app switching when needed")

        return "\n".join(lines)

    # ==================== 直接加载App数据 ====================

    def load_app_data_directly(self, app_name: str) -> Optional[Dict[str, Any]]:
        """直接加载app数据(无向量检索)"""
        app_dir = self.level2_dir / app_name
        if not app_dir.exists():
            return None

        meta = self._load_json(app_dir / "meta.json")
        workflows = self._load_json(app_dir / "workflows.json")
        preferences = self._load_json(app_dir / "preferences.json")
        usage_stats = self._load_json(app_dir / "usage_stats.json")

        return {
            "meta": meta,
            "workflows": workflows,
            "preferences": preferences,
            "usage_stats": usage_stats
        }

    def load_multiple_apps_directly(self, app_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """直接加载多个app的数据"""
        results = {}
        for app_name in app_names:
            data = self.load_app_data_directly(app_name)
            if data:
                results[app_name] = data
        return results

    def _load_json(self, file_path: Path) -> Any:
        """加载JSON文件"""
        if not file_path.exists():
            return {} if file_path.name != "workflows.json" else []

        try:
            with file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {} if file_path.name != "workflows.json" else []

    # ==================== 统计信息 ====================

    def print_index_info(self):
        """打印索引信息"""
        print("\n" + "=" * 70)
        print("RAG Index Information")
        print("=" * 70)

        if self.level1_index_dir.exists():
            try:
                if self._level1_store is None and self.embeddings:
                    self._level1_store = FAISS.load_local(
                        str(self.level1_index_dir),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )

                if self._level1_store:
                    print(f"\n Level 1 Index:")
                    print(f"   Status: Loaded")
                    print(f"   Location: {self.level1_index_dir}")
            except Exception as e:
                print(f"\nLevel 1 Index:")
                print(f"   Status: Error: {e}")
        else:
            print(f"\n Level 1 Index:")
            print(f"   Status: Not found")

        if self.level2_index_dir.exists():
            try:
                if self._level2_store is None and self.embeddings:
                    self._level2_store = FAISS.load_local(
                        str(self.level2_index_dir),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )

                if self._level2_store:
                    print(f"\n Level 2 Index:")
                    print(f"   Status: Loaded")
                    print(f"   Location: {self.level2_index_dir}")
            except Exception as e:
                print(f"\nLevel 2 Index:")
                print(f"   Status: Error: {e}")
        else:
            print(f"\n Level 2 Index:")
            print(f"   Status: Not found")

        print("=" * 70)


def init_query_engine(
        data_dir: str = None,
        embeddings: Embeddings = None
) -> RAGQueryEngine:
    """
    初始化查询引擎（支持自定义目录）

    Args:
        data_dir: 数据目录（如果为None则使用Config）
        embeddings: Embeddings对象

    Returns:
        RAGQueryEngine实例
    """
    if data_dir is None:
        from config import Config
        data_dir = Config.RAG_DATA_DIR

    return RAGQueryEngine(
        data_dir=data_dir,
        index_dir=os.path.join(data_dir, "indices"),
        embeddings=embeddings
    )


def init_query_engine_from_config(embeddings: Embeddings = None) -> RAGQueryEngine:
    """从Config初始化查询引擎（向后兼容）"""
    return init_query_engine(data_dir=None, embeddings=embeddings)