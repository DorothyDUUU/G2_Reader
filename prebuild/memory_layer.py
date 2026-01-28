# memory_system.py

from ast import Str
from typing import List, Dict, Optional, Literal, Any, Union
import json
from datetime import datetime
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path
import pickle
from openai import OpenAI
import copy
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import asyncio
from tqdm.asyncio import tqdm_asyncio
from prebuild.usage_tracker import add_chat_usage, add_embed_usage, add_single_call_duration
import time
import re

from config.config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    EMBED_BASE_URL,
    EMBED_API_KEY,
    MODELS,
    MEMORY_SYSTEMS_DIR,
    PROMPTS,
    MAX_CONCURRENCY,
)
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI


client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
aclient = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

qwen_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
qwen_aclient = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

embed_client = OpenAI(api_key=EMBED_API_KEY, base_url=EMBED_BASE_URL)
embed_aclient = AsyncOpenAI(api_key=EMBED_API_KEY, base_url=EMBED_BASE_URL)

# ======================================================================
# MemoryNote Class
# ======================================================================
class MemoryNote:
    """Basic memory unit with metadata"""
    def __init__(self, 
                 content: str,
                 context: str, 
                 keywords: List[str],
                 tags: List[str],
                 text_content:  str = "",
                 visual: bool = False, 
                 id: Optional[str] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = 0,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 ):
        
        self.content = content
        self.context = context
        if isinstance(self.context, list):
            self.context = " ".join(self.context)

        self.keywords = keywords
        self.tags = tags
        self.visual = visual
        self.text_content = text_content
        
        # Defaults
        self.id = id or str(uuid.uuid4())
        self.links = links or []
        self.category = category or "Uncategorized"


# ======================================================================
# SimpleEmbeddingRetriever
# ======================================================================
class SimpleEmbeddingRetriever:
    """Simple retrieval system using only text embeddings."""
    
    def __init__(self, model_name: str = MODELS["embed"]):
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}
        self.model_name = model_name
        
    def add_documents(self, documents: List[str], pre_embeddings):
        start_idx = len(self.corpus)
        self.corpus.extend(documents)

        if self.embeddings is None:
            self.embeddings = pre_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, pre_embeddings])

        for idx, doc in enumerate(documents):
            self.document_ids[doc] = start_idx + idx
    
    def search_original(self, query: str, k: int = 5):
        resp = embed_client.embeddings.create(
            model=self.model_name, 
            input=query
        )
        query_embedding = resp.data[0].embedding
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices
    
    async def search(self, query: str, k: int = 5):
        resp = await embed_aclient.embeddings.create(
            model=self.model_name, 
            input=query
        )

        query_embedding = resp.data[0].embedding
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices


# ======================================================================
# AgenticMemorySystem
# ======================================================================
class AgenticMemorySystem:
    def __init__(self, 
                 model_name: str = MODELS["embed"],
                 llm_model: str = MODELS["chat"]):
        self.model_name = model_name
        self.llm_model = llm_model
        self.memories = {}
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.evolution_system_prompt = PROMPTS["evolve"]
        
        self._evolve_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


    # ==========================================================
    def add_note(self, content: str, time: str = None, visual: bool = False, pre_embeddings=None, **kwargs) -> str:
        note = MemoryNote(content=content, timestamp=time, visual=visual, **kwargs)
        self.memories[note.id] = note

        self.retriever.add_documents(
            [note.context + " keywords: " + ", ".join(note.keywords)], 
            pre_embeddings
        )
        return note.id
    
    # ==========================================================
    def reset_retriever(self):
        self.retriever = SimpleEmbeddingRetriever(self.model_name)



    async def _call_llm_evolve(self, messages, response_format, temperature=0.7, max_tokens=20480):
        # 带重试的 LLM 调用，确保 JSON 可解析；解析失败也会触发重试
        async with self._evolve_semaphore: 
            call_start = time.time()  # 记录单次调用开始时间
            try:
                response = await qwen_aclient.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    response_format=response_format,
                    temperature=temperature,
                    max_tokens=2048
                )
            except Exception as e:
                print(f"[evolve] API error: {e}")
                await asyncio.sleep(1)
                raise  # ❌ API 失败，不记录此次调用时间
            
            # ✅ 只有成功的调用才计算耗时
            call_duration = time.time() - call_start
        
        # 记录演化阶段的token使用
        try:
            add_chat_usage(
                getattr(response, "usage", None),
                {"model": self.llm_model, "qkind": "memory_evolution"}
            )
            # 只记录成功调用的最大耗时
            add_single_call_duration("memory_evolution", call_duration)
        except Exception:
            pass
        
        return json.loads(response.choices[0].message.content)

    async def process_memory_single(self, note: MemoryNote):
        try: 
            memories_list = list(self.memories.values())
            note_original = copy.deepcopy(note)

            indices_embed, _ = await self.find_related_notes(
                note.context + " keywords: " + ", ".join(note.keywords),
                k=5,
                include_neighbors=False
            )

            indices_embed = [int(i) for i in indices_embed]

            neighbors_embed = [(i, memories_list[i]) for i in indices_embed]
            neighbors_link = [(i, memories_list[i]) for i in note.links]
            neighbors = list(set(neighbors_embed + neighbors_link))

            neighbor_str = ""
            neighbor_images = []

            for i, neighbor in neighbors:
                if not neighbor.visual:
                    neighbor_str += (
                        f"memory id: {i}, memory content: {neighbor.content}, "
                        f"memory summary: {neighbor.context}, memory keywords: {neighbor.keywords}\n"
                    )
                else:
                    neighbor_str += (
                        f"memory id: {i}, memory content: image {len(neighbor_images)+1} attached below, "
                        f"memory summary: {neighbor.context}, memory keywords: {neighbor.keywords}\n"
                    )
                    neighbor_images.append(neighbor.content)

            # 将 format 改为定点替换，避免 JSON 花括号触发 KeyError
            template = self.evolution_system_prompt
            prompt_memory = (
                template.replace("{context}", str(note.context))
                        .replace("{content}", "image 0 attached below" if note.visual else str(note.content))
                        .replace("{keywords}", ", ".join(note.keywords))
                        .replace("{neighbors}", neighbor_str)
                        .replace("{neighbor_number}", str(len(neighbors)))
            )

            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "suggested_connections": {
                                "type": "array",
                                "items": {"type": "integer"}
                            },
                            "should_update": {"type": "boolean"},
                            "new_summary": {"type": "string"},
                            "new_keywords": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["suggested_connections", "should_update", "new_summary", "new_keywords"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }

            messages = [{"role": "system", "content": "You must respond with a JSON object."}]
            messages.append({"role": "user", "content": [{"type": "text","text": prompt_memory}]})

            if note.visual:
                messages[1]["content"].append({"type": "text", "text": "Here is image 0:"})
                messages[1]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{note.content}"}})

            for i, image in enumerate(neighbor_images):
                messages[1]["content"].append({"type": "text", "text": f"Here is image {i+1}: "})
                messages[1]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})

            # 使用带重试的统一调用（替换原先的 client.chat.completions.create + json.loads）
            response_json = await self._call_llm_evolve(messages, response_format, temperature=0.7, max_tokens=2048)

            note.links = [
                i for i in response_json["suggested_connections"] 
                if i < len(memories_list)
            ][:5]

            if response_json["should_update"]:
                note.context = response_json["new_summary"]
                note.keywords = response_json["new_keywords"]
                
            return note

        except Exception as e:
            print("Error when evolving memory note", e)
            # if raw:
            #     print("Raw response", raw)
            return note_original



    async def process_memory_all(self):
        tasks = [
            self.process_memory_single(note) 
            for note in self.memories.values()
        ]
        results = await tqdm_asyncio.gather(*tasks)
        return results
              
    async def find_related_notes(self, query: str, k: int = 5, include_neighbors: bool = True, modality: str = "all"):

        if not self.memories:
            return []

        indices = await self.retriever.search(query, 30)
        indices_return = []
        
        all_memories = list(self.memories.values())
        notes = []
        j = 0

        for i in indices:
            if all_memories[i] not in notes and (modality == "all" or all_memories[i].category == modality):
                notes.append(all_memories[i])
                indices_return.append(i)
                j += 1
            if j >= k:
                return indices_return, notes

            if include_neighbors:
                neighborhood = all_memories[i].links[:3]
            else:
                neighborhood = []

            for neighbor in neighborhood:
                if all_memories[neighbor] not in notes and (modality == "all" or all_memories[neighbor].category == modality):
                    notes.append(all_memories[neighbor])
                    indices_return.append(neighbor)
                    j += 1
                if j >= k:
                    return indices_return, notes

        return indices_return, notes
    
    def find_related_notes_original(self, query: str, k: int = 5, include_neighbors: bool = True, modality: str = "all"):

        if not self.memories:
            return []

        indices = self.retriever.search_original(query, 30)
        indices_return = []
        
        all_memories = list(self.memories.values())
        notes = []
        j = 0

        for i in indices:
            if all_memories[i] not in notes and (modality == "all" or all_memories[i].category == modality):
                notes.append(all_memories[i])
                indices_return.append(i)
                j += 1
            if j >= k:
                return indices_return, notes

            if include_neighbors:
                neighborhood = all_memories[i].links[:3]
            else:
                neighborhood = []

            for neighbor in neighborhood:
                if all_memories[neighbor] not in notes and (modality == "all" or all_memories[neighbor].category == modality):
                    notes.append(all_memories[neighbor])
                    indices_return.append(neighbor)
                    j += 1
                if j >= k:
                    return indices_return, notes

        return indices_return, notes


    # ======================================================================
    # SAVE
    # ======================================================================
    def save_memory_system(self, name: str):
        try:
            system_dir = Path(MEMORY_SYSTEMS_DIR) / name
            system_dir.mkdir(parents=True, exist_ok=True)

            if len(self.memories) == 0:
                raise ValueError("无法保存：memories字典为空！")

            pkl_path = system_dir / "memories.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(self.memories, f)

            pkl_size = pkl_path.stat().st_size
            if pkl_size <= 10:
                raise RuntimeError(f"保存异常：memories.pkl 文件过小 ({pkl_size} 字节)")

            npy_path = system_dir / "retriever_embeddings.npy"
            np.save(npy_path, self.retriever.embeddings)

            npy_size = npy_path.stat().st_size

            print(f"保存成功:")
            print(f"  - memories.pkl: {pkl_size / 1024:.2f} KB ({len(self.memories)} 条记忆)")
            print(f"  - embeddings.npy: {npy_size / 1024:.2f} KB")

            return True
        
        except Exception as e:
            print(f"save_memory_system 失败: {e}")
            raise

    # ======================================================================
    # LOAD
    # ======================================================================
    def load_memory_system(self, name: str):
        try:
            system_dir = Path(MEMORY_SYSTEMS_DIR) / name

            if not system_dir.exists():
                raise FileNotFoundError(f"Memory system 目录不存在: {system_dir}")

            pkl_path = system_dir / "memories.pkl"
            npy_path = system_dir / "retriever_embeddings.npy"

            if not pkl_path.exists():
                raise FileNotFoundError(f"memories.pkl 不存在: {pkl_path}")
            if not npy_path.exists():
                raise FileNotFoundError(f"retriever_embeddings.npy 不存在: {npy_path}")

            pkl_size = pkl_path.stat().st_size
            if pkl_size <= 10:
                raise ValueError(f"memories.pkl 文件过小 ({pkl_size} 字节)")

            # --- 兼容旧版 pickle：为 'memory_layer' 建立模块别名 ---
            import importlib, sys
            if "memory_layer" not in sys.modules:
                try:
                    sys.modules["memory_layer"] = importlib.import_module("prebuild.memory_layer")
                except Exception as e:
                    print(f"load_memory_system 失败: {e}")
                    raise

            with open(pkl_path, "rb") as f:
                self.memories = pickle.load(f)

            if not isinstance(self.memories, dict):
                raise TypeError("memories 应为 dict")

            if len(self.memories) == 0:
                raise ValueError("加载的 memories 为空")

            self.retriever.embeddings = np.load(npy_path)

            print(f"加载成功:")
            print(f"  - 记忆数量: {len(self.memories)}")
            print(f"  - 嵌入向量形状: {self.retriever.embeddings.shape}")

            return True
        
        except Exception as e:
            print(f"load_memory_system 失败: {e}")
            raise

    def _phrase_rank_texts_optimized(self, docs: List[str], query_terms: List[str], k: int = 5, return_indices: bool = False) -> Union[List[str], List[int]]:
        """
        短语匹配打分（优化版）。
        
        Args:
            docs: 文本列表
            query_terms: 查询词列表
            k: 返回前 K 个
            return_indices: 
                - False (默认): 返回匹配的文本字符串列表 [str]
                - True: 返回匹配的文档在 docs 中的索引列表 [int]
                
        逻辑：优先奖励命中更多"不同"的关键词。
        """
        # 1. 预处理 Query Terms
        valid_terms = [qt.strip() for qt in (query_terms or []) if qt and qt.strip()]
        valid_terms.sort(key=len, reverse=True)     
        if not valid_terms:
            return []

        # 2. 构建正则
        clean_terms = [re.escape(t) for t in valid_terms]
        pattern_str = r"\b(?:" + "|".join(clean_terms) + r")\b"
        regex = re.compile(pattern_str, re.IGNORECASE)

        # 3. 扫描文档并打分
        scores = []
        for i, doc in enumerate(docs):
            if not doc: continue           
            matches = regex.findall(doc)            
            if not matches:
                continue

            # --- 核心算分优化 ---         
            # A. 覆盖度得分 (Unique Hits)
            unique_hits = {m.lower() for m in matches}
            count_unique = len(unique_hits)            
            # B. 频次得分 (Total Hits)
            count_total = len(matches)            
            # C. 综合打分: 覆盖度权重 1.0, 频次权重 0.01
            score = count_unique * 1.0 + count_total * 0.01         
            scores.append((i, score))

        # 4. 排序（过滤0分）
        scores = [t for t in scores if t[1] > 0]
        if not scores:
            return [] if return_indices else []
        scores.sort(key=lambda x: x[1], reverse=True)
        top_idx = [i for i, _ in scores[:k]]
        # 5. 根据 flag 返回不同结果
        if return_indices:
            return top_idx
        else:
            return [docs[i] for i in top_idx]

    def search_keyword(self, keywords: List[str], modality: str = "all", top_k_text: int = 5, top_k_image: int = 5) -> List[MemoryNote]:
        all_memories = list(self.memories.values())
        text_docs: List[str] = []
        text_map: List[int] = []
        image_docs: List[str] = []
        image_map: List[int] = []
        for idx, n in enumerate(all_memories):
            c = getattr(n, "content", None)
            ctx = getattr(n, "text_content", None)
            v = getattr(n, "visual", False)

            if not v and isinstance(c, str) and c.strip():
                text_docs.append(c.strip())
                text_map.append(idx)
            elif v and isinstance(ctx, str) and ctx.strip():
                image_docs.append(ctx.strip())
                image_map.append(idx)
        selected: List[int] = []
        if modality in ("all", "text") and top_k_text > 0 and text_docs:
            top_text_idx = self._phrase_rank_texts_optimized(text_docs, keywords, k=int(top_k_text), return_indices=True)
            for di in top_text_idx:
                selected.append(text_map[di])

        if modality in ("all", "image") and top_k_image > 0 and image_docs:
            top_img_idx = self._phrase_rank_texts_optimized(image_docs, keywords, k=int(top_k_image), return_indices=True)
            for di in top_img_idx:
                selected.append(image_map[di])

        notes: List[MemoryNote] = []
        for mi in selected:
            n = all_memories[mi]
            notes.append(n)
        return notes

    