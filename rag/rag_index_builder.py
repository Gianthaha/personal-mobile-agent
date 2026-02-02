"""
RAG Vector Index Builder - FAISS向量索引构建 (增强版)
为一级和二级知识库构建向量索引

修复:
1. 增强索引构建的健壮性
2. 支持增量更新多个app
3. 更好的错误处理
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings.base import Embeddings


class RAGIndexBuilder:
    """RAG向量索引构建器 - 增强版"""

    def __init__(
            self,
            data_dir: str = "./data",
            index_dir: str = "./data/indices",
            embeddings: Optional[Embeddings] = None
    ):
        """
        初始化索引构建器

        Args:
            data_dir: 数据根目录
            index_dir: 索引保存目录
            embeddings: Embeddings对象
        """
        self.data_dir = Path(data_dir)
        self.level1_dir = self.data_dir / "level1"
        self.level2_dir = self.data_dir / "level2"

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.level1_index_dir = self.index_dir / "level1"
        self.level2_index_dir = self.index_dir / "level2"

        self.embeddings = embeddings

    def set_embeddings(self, embeddings: Embeddings):
        """设置embeddings对象"""
        self.embeddings = embeddings

    # ==================== 一级索引构建 ====================

    def build_level1_index(self):
        """构建一级知识库的向量索引"""
        print("\n" + "=" * 70)
        print(" Building Level 1 Vector Index")
        print("=" * 70)

        if not self.embeddings:
            raise ValueError("Embeddings not set! Call set_embeddings() first.")

        if not self.level1_dir.exists():
            print(f" Level 1 data directory not found: {self.level1_dir}")
            print(" Creating empty Level 1 index...")
            self.level1_dir.mkdir(parents=True, exist_ok=True)
            return

        documents = []
        category_files = sorted(self.level1_dir.glob("*.json"))

        if not category_files:
            print(" No category files found in Level 1")
            return

        for category_file in category_files:
            try:
                with category_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                embedding_text = self._format_category_for_embedding(data)

                doc = Document(
                    page_content=embedding_text,
                    metadata={
                        "id": data.get("id", 0),
                        "category": data.get("category", "Unknown"),
                        "apps": data.get("apps", []),
                        "source": "level1"
                    }
                )
                documents.append(doc)

                print(f"  Processed: {data.get('category', category_file.stem)}")
            except Exception as e:
                print(f"  ⚠️ Error processing {category_file.name}: {e}")

        if documents:
            vector_store = FAISS.from_documents(documents, self.embeddings)
            self.level1_index_dir.mkdir(parents=True, exist_ok=True)
            vector_store.save_local(str(self.level1_index_dir))
            print(f"\n✅ Level 1 index saved: {self.level1_index_dir}")
            print(f"   Total documents: {len(documents)}")
        else:
            print(" No documents to index")

    def _format_category_for_embedding(self, data: Dict) -> str:
        """格式化category数据为embedding文本"""
        category = data.get("category", "Unknown")
        tasks = ", ".join(data.get("typical_tasks", []))
        apps = data.get("apps", [])
        if isinstance(apps, list):
            apps = ", ".join(apps)

        text = f"Category: {category}\n"
        if tasks:
            text += f"Typical Tasks: {tasks}\n"
        text += f"Apps: {apps}"

        return text

    # ==================== 二级索引构建 ====================

    def build_level2_index(self):
        """构建二级知识库的向量索引"""
        print("\n" + "=" * 70)
        print(" Building Level 2 Vector Index")
        print("=" * 70)

        if not self.embeddings:
            raise ValueError("Embeddings not set! Call set_embeddings() first.")

        if not self.level2_dir.exists():
            print(f" Level 2 data directory not found: {self.level2_dir}")
            print(" Creating empty Level 2...")
            self.level2_dir.mkdir(parents=True, exist_ok=True)
            return

        all_documents = []
        app_dirs = [d for d in self.level2_dir.iterdir() if d.is_dir()]

        if not app_dirs:
            print(" No app directories found in Level 2")
            return

        for app_dir in sorted(app_dirs):
            app_name = app_dir.name
            print(f"\n  Processing: {app_name}")

            try:
                # 1. App Info Document
                info_doc = self._build_app_info_doc(app_dir)
                if info_doc:
                    all_documents.append(info_doc)
                    print(f"    ✓ App Info")

                # 2. Workflow Documents
                workflow_docs = self._build_workflow_docs(app_dir)
                all_documents.extend(workflow_docs)
                print(f"    ✓ Workflows: {len(workflow_docs)} docs")

                # 3. Preference Document
                pref_doc = self._build_preference_doc(app_dir)
                if pref_doc:
                    all_documents.append(pref_doc)
                    print(f"    ✓ Preferences")

                # 4. Usage Stats Document
                stats_doc = self._build_usage_stats_doc(app_dir)
                if stats_doc:
                    all_documents.append(stats_doc)
                    print(f"    ✓ Usage Stats")
            except Exception as e:
                print(f"    ⚠️ Error processing {app_name}: {e}")

        if all_documents:
            vector_store = FAISS.from_documents(all_documents, self.embeddings)
            self.level2_index_dir.mkdir(parents=True, exist_ok=True)
            vector_store.save_local(str(self.level2_index_dir))
            print(f"\n✅ Level 2 index saved: {self.level2_index_dir}")
            print(f"   Total documents: {len(all_documents)}")
        else:
            print(" No documents to index")

    def _build_app_info_doc(self, app_dir: Path) -> Optional[Document]:
        """构建app info文档"""
        meta_file = app_dir / "meta.json"
        if not meta_file.exists():
            return None

        try:
            with meta_file.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            text = (
                f"App: {meta.get('app_name', app_dir.name)}\n"
                f"Category: {meta.get('category', 'Unknown')}\n"
                f"Description: {meta.get('description', '')}"
            )

            return Document(
                page_content=text,
                metadata={
                    "app_name": meta.get("app_name", app_dir.name),
                    "category": meta.get("category", "Unknown"),
                    "doc_type": "app_info",
                    "source": "level2"
                }
            )
        except Exception as e:
            print(f"      ⚠️ Error building app_info doc: {e}")
            return None

    def _build_workflow_docs(self, app_dir: Path) -> List[Document]:
        """构建workflow文档"""
        workflows_file = app_dir / "workflows.json"
        if not workflows_file.exists():
            return []

        try:
            with workflows_file.open("r", encoding="utf-8") as f:
                workflows = json.load(f)

            if not workflows or not isinstance(workflows, list):
                return []

            # 读取meta获取app_name
            meta_file = app_dir / "meta.json"
            if meta_file.exists():
                with meta_file.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                app_name = meta.get("app_name", app_dir.name)
            else:
                app_name = app_dir.name

            docs = []
            for idx, workflow in enumerate(workflows):
                if not isinstance(workflow, dict):
                    continue

                task = workflow.get('task', '')
                task_summary = workflow.get('task_summary', '')

                # Embedding文本 - 只包含task和summary
                embedding_text = (
                    f"App: {app_name}\n"
                    f"Task: {task}\n"
                    f"Summary: {task_summary}"
                )

                # 格式化steps
                steps_list = workflow.get("steps", [])
                if isinstance(steps_list, list):
                    if steps_list and isinstance(steps_list[0], dict):
                        steps_text = "\n".join([
                            f"{step.get('step', i)}. {step.get('action', '')} - {step.get('thought', '')}"
                            for i, step in enumerate(steps_list, 1)
                        ])
                    else:
                        steps_text = "\n".join([f"{i}. {step}" for i, step in enumerate(steps_list, 1)])
                else:
                    steps_text = ""

                # 格式化UI元素
                ui_elements_text = ""
                ui_elements = workflow.get("ui_elements", [])
                if ui_elements and isinstance(ui_elements, list):
                    ui_text_parts = []
                    for ui in ui_elements:
                        if isinstance(ui, dict):
                            element = ui.get("element", "")
                            position = ui.get("position", "")
                            action = ui.get("action", "")
                            ui_text_parts.append(f"- {element} at {position}: {action}")
                    ui_elements_text = "\n".join(ui_text_parts)

                doc = Document(
                    page_content=embedding_text,
                    metadata={
                        "app_name": app_name,
                        "workflow_id": f"{app_name}_wf_{idx}",
                        "task": task,
                        "task_summary": task_summary,
                        "steps": steps_text,
                        "ui_elements": ui_elements_text,
                        "success_count": workflow.get('success_count', 0),
                        "doc_type": "workflow",
                        "source": "level2"
                    }
                )
                docs.append(doc)

            return docs
        except Exception as e:
            print(f"      ⚠️ Error building workflow docs: {e}")
            return []

    def _build_preference_doc(self, app_dir: Path) -> Optional[Document]:
        """构建preference文档"""
        pref_file = app_dir / "preferences.json"
        if not pref_file.exists():
            return None

        try:
            with pref_file.open("r", encoding="utf-8") as f:
                pref_data = json.load(f)

            # 读取meta
            meta_file = app_dir / "meta.json"
            if meta_file.exists():
                with meta_file.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                app_name = meta.get("app_name", app_dir.name)
            else:
                app_name = app_dir.name

            # 格式化偏好数据
            pref_text_parts = []

            # 1. 任务偏好
            task_prefs = pref_data.get("task_preferences", {})
            if task_prefs:
                sorted_tasks = sorted(task_prefs.items(), key=lambda x: x[1], reverse=True)
                tasks_text = ", ".join([f"{k} ({v})" for k, v in sorted_tasks[:5]])
                pref_text_parts.append(f"Task Preferences: {tasks_text}")

            # 2. 内容偏好
            content_prefs = pref_data.get("content_preferences", {})
            if content_prefs:
                sorted_content = sorted(content_prefs.items(), key=lambda x: x[1], reverse=True)
                content_text = ", ".join([f"{k} ({v})" for k, v in sorted_content[:10]])
                pref_text_parts.append(f"Content Preferences: {content_text}")

            # 3. 表达模式
            expr_patterns = pref_data.get("expression_patterns", {})
            if expr_patterns:
                sorted_patterns = sorted(expr_patterns.items(), key=lambda x: x[1], reverse=True)
                patterns_text = ", ".join([f"{k} ({v})" for k, v in sorted_patterns[:5]])
                pref_text_parts.append(f"Expression Patterns: {patterns_text}")

            if not pref_text_parts:
                return None

            text = (
                f"App: {app_name}\n"
                f"User Preferences:\n"
                + "\n".join(pref_text_parts) + "\n"
                f"Total Usage: {pref_data.get('total_usage', 0)}"
            )

            return Document(
                page_content=text,
                metadata={
                    "app_name": app_name,
                    "doc_type": "preferences",
                    "source": "level2"
                }
            )
        except Exception as e:
            print(f"      ⚠️ Error building preference doc: {e}")
            return None

    def _build_usage_stats_doc(self, app_dir: Path) -> Optional[Document]:
        """构建usage stats文档"""
        stats_file = app_dir / "usage_stats.json"
        if not stats_file.exists():
            return None

        try:
            with stats_file.open("r", encoding="utf-8") as f:
                stats = json.load(f)

            if stats.get("usage_count", 0) == 0:
                return None

            # 读取meta
            meta_file = app_dir / "meta.json"
            if meta_file.exists():
                with meta_file.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                app_name = meta.get("app_name", app_dir.name)
            else:
                app_name = app_dir.name

            success_rate = (stats.get("success_count", 0) / max(stats["usage_count"], 1)) * 100
            text = (
                f"App: {app_name}\n"
                f"Usage Count: {stats.get('usage_count', 0)}\n"
                f"Success Count: {stats.get('success_count', 0)}\n"
                f"Success Rate: {success_rate:.1f}%\n"
                f"Average Steps: {stats.get('avg_steps', 0):.1f}\n"
                f"Last Used: {stats.get('last_used', 'Never')}"
            )

            return Document(
                page_content=text,
                metadata={
                    "app_name": app_name,
                    "doc_type": "usage_stats",
                    "source": "level2"
                }
            )
        except Exception as e:
            print(f"      ⚠️ Error building usage_stats doc: {e}")
            return None

    # ==================== 增量更新 ====================

    def incremental_update_app(self, app_name: str):
        """增量更新单个app的索引"""
        print(f"\n Incrementally updating: {app_name}")

        if not self.embeddings:
            raise ValueError("Embeddings not set!")

        if not self.level2_index_dir.exists():
            print(" No existing index, building from scratch...")
            self.build_level2_index()
            return

        try:
            vector_store = FAISS.load_local(
                str(self.level2_index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f" Failed to load index, rebuilding: {e}")
            self.build_level2_index()
            return

        app_dir = self.level2_dir / app_name
        if not app_dir.exists():
            print(f" App directory not found: {app_name}")
            return

        new_docs = []

        info_doc = self._build_app_info_doc(app_dir)
        if info_doc:
            new_docs.append(info_doc)

        workflow_docs = self._build_workflow_docs(app_dir)
        new_docs.extend(workflow_docs)

        pref_doc = self._build_preference_doc(app_dir)
        if pref_doc:
            new_docs.append(pref_doc)

        stats_doc = self._build_usage_stats_doc(app_dir)
        if stats_doc:
            new_docs.append(stats_doc)

        if new_docs:
            new_store = FAISS.from_documents(new_docs, self.embeddings)
            vector_store.merge_from(new_store)
            vector_store.save_local(str(self.level2_index_dir))
            print(f"✅ Updated {len(new_docs)} documents for {app_name}")
        else:
            print(f" No documents to update for {app_name}")

    def incremental_update_apps(self, app_names: List[str]):
        """增量更新多个app的索引"""
        print(f"\n Incrementally updating {len(app_names)} apps...")

        for app_name in app_names:
            self.incremental_update_app(app_name)

    # ==================== 完整重建 ====================

    def rebuild_all_indices(self):
        """完整重建所有索引"""
        print("\n" + "=" * 70)
        print(" Rebuilding All Indices")
        print("=" * 70)

        self.build_level1_index()
        self.build_level2_index()

        print("\n" + "=" * 70)
        print("✅ All Indices Rebuilt Successfully")
        print("=" * 70)


def init_index_builder(
        data_dir: str = None,
        embeddings: Embeddings = None
) -> RAGIndexBuilder:
    """
    初始化索引构建器（支持自定义目录）

    Args:
        data_dir: 数据目录（如果为None则使用Config）
        embeddings: Embeddings对象

    Returns:
        RAGIndexBuilder实例
    """
    if data_dir is None:
        from config import Config
        data_dir = Config.RAG_DATA_DIR

    return RAGIndexBuilder(
        data_dir=data_dir,
        index_dir=os.path.join(data_dir, "indices"),
        embeddings=embeddings
    )


def init_index_builder_from_config(embeddings: Embeddings = None) -> RAGIndexBuilder:
    """从Config初始化索引构建器（向后兼容）"""
    return init_index_builder(data_dir=None, embeddings=embeddings)